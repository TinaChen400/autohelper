"""Local screenshot-only recognition for the sample TTS evaluation layout."""
from dataclasses import dataclass
import re
import time
import cv2
import numpy as np
from pathlib import Path


@dataclass
class Target:
    name: str
    label: tuple
    button: tuple
    state: str
    confidence: float


def frame_bgr(frame):
    w, h, rgb = frame
    return cv2.cvtColor(np.frombuffer(rgb, np.uint8).reshape(h, w, 3), cv2.COLOR_RGB2BGR)


def glyph_state(image, box):
    x, y, w, h = box
    crop = image[y:y+h, x:x+w]
    if crop.size == 0: return 'unknown'
    # At 12-18 px a pause bar may be only one pixel wide: contour area
    # would be zero. Normalize before contour extraction, without inventing ink.
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_NEAREST)
    w = h = 64
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yy, xx = np.ogrid[:h, :w]
    inner = ((xx-(w-1)/2)/(w*.34))**2 + ((yy-(h-1)/2)/(h*.34))**2 < 1
    for threshold in (180, 150, 120):
        mask = ((gray > threshold) & inner).astype(np.uint8)*255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pieces = [c for c in contours if cv2.contourArea(c) > max(1, w*h*.009)]
        if len(pieces) == 2:
            rects = sorted([cv2.boundingRect(c) for c in pieces])
            if (all(r[3] > r[2]*1.4 for r in rects)
                    and abs(rects[0][1]-rects[1][1]) < h*.12
                    and abs(rects[0][3]-rects[1][3]) < h*.15):
                return 'active'
            # A remote cursor's small white outline can overlap the play
            # icon. It is not a second pause bar. Admit the dominant central
            # triangle only when the extra piece is substantially smaller.
            dominant=sorted(pieces,key=cv2.contourArea,reverse=True)
            if cv2.contourArea(dominant[1])<cv2.contourArea(dominant[0])*.5:
                a,b,cw,ch=cv2.boundingRect(dominant[0])
                if abs(a+cw/2-w/2)<w*.15:pieces=[dominant[0]]
        if len(pieces) == 1:
            c = pieces[0]
            a, b, cw, ch = cv2.boundingRect(c)
            if cw < w*.2 or ch < h*.25: continue
            # A right-pointing triangle has much more ink in its left half.
            region = mask[b:b+ch, a:a+cw]
            left = np.count_nonzero(region[:, :max(1, cw//2)])
            right = np.count_nonzero(region[:, cw//2:])
            polygon = cv2.approxPolyDP(c, cv2.arcLength(c, True)*.07, True)
            if 3 <= len(polygon) <= 5 and left > right*1.25:
                return 'idle'
    return 'unknown'


def player_buttons(image, threshold=85):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < threshold).astype(np.uint8)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL if threshold == 85 else cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (12 <= w <= 80 and 12 <= h <= 80 and .78 < w/h < 1.28): continue
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area/(w*h) < .55 or perimeter == 0 or 4*np.pi*area/perimeter**2 < .6: continue
        state = glyph_state(image, (x, y, w, h))
        if state != 'unknown': result.append(((x, y, w, h), state))
    return result


def prerequisite_name(text, require_heading=False):
    """Exact player/prerequisite identity; one OCR edit only in 'first'."""
    text=re.sub(r'[^a-z0-9]','',text.lower())
    for name,heading,action in (('A',r'a?responsea',r'playinputs?'),
                                ('B',r'b?responseb',r'playresponsea')):
        prefix=heading+r'\d{0,2}'
        if not require_heading:prefix='(?:'+prefix+')?'
        match=re.fullmatch(prefix+action+r'([a-z]{4,6})',text)
        if not match:continue
        word=match[1];expected='first'
        if len(word)==5:
            close=sum(a!=b for a,b in zip(word,expected))<=1
        elif len(word)==4:
            close=any(expected[:i]+expected[i+1:]==word for i in range(5))
        else:
            close=any(word[:i]+word[i+1:]==expected for i in range(6))
        if close:return name
    return None


def label_name(text):
    # A lock icon between the response heading and its prerequisite can be
    # recognized as 8/0. Admit only that bounded, exact header structure;
    # numbered quick-replay labels and digits elsewhere remain excluded.
    locked_name=prerequisite_name(text,require_heading=True)
    if locked_name:return locked_name
    # Numbered quick-replay controls are not player section headers.
    if re.search(r'\d', text): return None
    text = re.sub(r'[^a-z]', '', text.lower())
    if text in ('responseaplayinputfirst', 'responseaplayinputsfirst', 'aresponseaplayinputfirst', 'aresponseaplayinputsfirst'): return 'A'
    if text in ('responsebplayresponseafirst', 'bresponsebplayresponseafirst'): return 'B'
    if text in ('responsea', 'aresponsea'): return 'A'
    if text in ('responseb', 'bresponseb'): return 'B'
    if text in ('inputspeech', 'inputspeechvoice', 'generatespeechusingthisvoice', 'inputaudio'):
        return 'Input'
    # Restrict typo tolerance to the long reference heading; never fuzzy-match
    # A vs B. Small remote text commonly reads this/voice as thls/voloe.
    reference = 'generatespeechusingthisvoice'
    if abs(len(text)-len(reference)) <= 3:
        costs = list(range(len(reference)+1))
        for i, char in enumerate(text, 1):
            updated = [i]
            for j, other in enumerate(reference, 1):
                updated.append(min(updated[-1]+1, costs[j]+1, costs[j-1]+(char != other)))
            costs = updated
        if costs[-1] <= 3: return 'Input'
    return None


def associate(lines, buttons, shape):
    """Only bind header labels to the left-hand control of a player below it."""
    height, width = shape[:2]
    labels = []
    for quad, text, confidence in lines:
        name = label_name(text)
        # Some reference tasks label the container only "Inputs". Bind it
        # only to a real player above BOTH response headings; a text-only
        # Inputs container must never manufacture a reference player.
        if not name and re.sub(r'[^a-z]','',text.lower())=='inputs':
            response_tops=[float(np.asarray(q)[:,1].min()) for q,t,s in lines
                           if label_name(t) in ('A','B') and s>=.85]
            if len(response_tops)>=2:
                q=np.asarray(quad)
                candidates=[b for b,_ in buttons if q[:,1].max()<b[1] and
                            b[1]+b[3]<min(response_tops) and
                            abs(b[0]-q[:,0].min())<max(30,(q[:,1].max()-q[:,1].min())*4)]
                if len(candidates)==1:name='Input'
        if not name or confidence < .85: continue
        q = np.asarray(quad)
        x, y = q.min(axis=0); right, bottom = q.max(axis=0)
        labels.append((name, (int(x), int(y), int(right-x), int(bottom-y)), float(confidence)))
    candidates = {n: [] for n in ('Input', 'A', 'B')}
    for name, label, confidence in labels:
        lx, ly, lw, lh = label
        # Response labels in rating rows have no full player below them;
        # quick replay labels are explicitly excluded by label_name.
        for button, state in buttons:
            bx, by, bw, bh = button
            dy = by + bh/2 - (ly + lh)
            dx = bx + bw/2 - lx
            if not (lh*2 < dy < min(height*.42, lh*20)): continue
            if not (-lh*4 < dx < lh*3): continue
            # Another header on the same column between label and control
            # makes ownership ambiguous. Do not cross it.
            crossed = any(other[1][1] > ly + lh and other[1][1] < by
                          and abs(other[1][0]-lx) < max(lw, lh*5)
                          for other in labels if other[1] != label)
            if not crossed: candidates[name].append(Target(name, label, button, state, confidence))
    targets, issues = {}, []
    for name, found in candidates.items():
        if len(found) == 1: targets[name] = found[0]
        else: issues.append(f'{name}: ' + ('未找到完整标签与播放图标' if not found else '存在多个对应，拒绝定位'))
    # Different labels must never own the same button.
    for a in list(targets):
        if any(a != b and targets[a].button == targets[b].button for b in targets):
            issues.append('标签指向同一按钮，拒绝点击')
            return {}, issues
    return targets, issues


class Detector:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        model=Path(__file__).resolve().parent/'audio_lab'/'models'/'english-ocr'/'en_PP-OCRv4_rec_mobile.onnx'
        options={'rec_model_path':str(model)} if model.exists() else {}
        self.ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2,rec_batch_num=1,**options)
        self.english_page_ocr=bool(options)
        if options:self.text_ocr=self.ocr
        self.fast_search=True
        self.fast_ui_ocr=True

    def ui_lines(self,image):
        from screen_ocr import ui_text_lines
        return ui_text_lines(self.ocr,image)

    def rating_lines(self,image):
        from screen_ocr import ui_text_lines
        # Rating labels/options are short. Explanatory paragraphs do not
        # determine click targets and need not enter the recognition batch.
        return ui_text_lines(self.ocr,image,max_aspect=24)

    def include_locked(self,image,lines,targets,issues):
        if {'A','B'}<=targets.keys():return targets,issues
        gray_targets,_=associate(lines,player_buttons(image,threshold=170),image.shape)
        for name in ('A','B'):
            if name in targets or name not in gray_targets:continue
            target=gray_targets[name]
            lx,ly,lw,lh=target.label
            locked=any(prerequisite_name(text)==name
                       and abs(float(np.asarray(q)[:,1].min())-ly)<max(12,lh*1.5)
                       and lx-lh*2<=float(np.asarray(q)[:,0].min())<=lx+image.shape[1]*.25
                       and score>=.8 for q,text,score in lines)
            if locked:
                from dataclasses import replace
                targets[name]=replace(target,state='locked')
        return targets,[issue for issue in issues if issue.split(':')[0] not in targets]

    def read_question_marker(self,image):
        from rating_reader import question_marker
        height,width=image.shape[:2]
        crop=image[:round(height*.55),round(width*.62):]
        return question_marker(self.ui_lines(crop))

    def refine_marker_lines(self,image,lines):
        """Read tiny task IDs from original pixels, not the reduced page."""
        from rating_reader import question_marker
        labels=[np.asarray(q) for q,text,score in lines
                if question_marker([(q,text,score)]) or
                re.search(r'^(?:Prompt\b|Run(?:\s*#?\s*\d+)?\s*$)',text.strip(),re.I)]
        if not labels:return lines
        label_points=np.concatenate(labels)
        top,bottom=label_points[:,1].min(),label_points[:,1].max()
        left=label_points[:,0].min()
        indices={i for i,(q,text,score) in enumerate(lines)
                 if abs(float(np.asarray(q)[:,1].mean())-(top+bottom)/2)<=max(12,bottom-top)
                 and float(np.asarray(q)[:,0].min())>=left-12}
        anchors=[entry for i,entry in enumerate(lines) if i in indices]
        points=np.concatenate([np.asarray(q) for q,_,_ in anchors])
        # A reduced-page pass may retain only Run or misread "Prompt" as
        # "Pzompt". Include its neighboring field in the native pixel retry.
        x=max(0,int(points[:,0].min())-max(12,round((bottom-top)*12)));y=max(0,int(points[:,1].min())-8)
        right=min(image.shape[1],int(points[:,0].max())+12)
        bottom=min(image.shape[0],int(points[:,1].max())+8)
        crop=image[y:bottom,x:right]
        if not crop.size:return lines
        refined=[(np.asarray(q)/2+np.array([x,y]),text,score)
                 for q,text,score in self.ui_lines(cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC))]
        before,after=dict(question_marker(anchors)),dict(question_marker(refined))
        if not after or not before.keys()<=after.keys():return lines
        return [entry for i,entry in enumerate(lines) if i not in indices]+refined

    def detect(self, image):
        start = time.perf_counter()
        scale = min(1, 1280/image.shape[1], 900/image.shape[0])
        reduced = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        lines = self.ui_lines(reduced)
        lines = [(np.asarray(q)/scale, text, score) for q, text, score in (lines or [])]
        buttons = player_buttons(image)
        targets, issues = associate(lines, buttons, image.shape)
        # Pale, explicitly locked players are already located. Their color is
        # not evidence of a missing label requiring another whole-page pass.
        targets,issues=self.include_locked(image,lines,targets,issues)
        # Downscaling can erase one otherwise visible response heading. Retry
        # at native resolution before reporting that its player is absent.
        if scale < 1 and not {'A', 'B'} <= targets.keys():
            native = self.ui_lines(image)
            if native:
                lines = native
                targets, issues = associate(lines, buttons, image.shape)
                targets,issues=self.include_locked(image,lines,targets,issues)
        # Reference headings are much smaller than A/B headings on this page.
        # Retry only the upper content area at higher resolution, not the whole
        # desktop. Preserve coordinates in the original screenshot.
        if 'Input' not in targets and image.shape[1] < 1500:
            boundary = min((t.label[1] for n, t in targets.items() if n in ('A', 'B')),
                           default=int(image.shape[0]*.6))
            anchor = targets.get('A')
            left = max(0, anchor.label[0]-anchor.label[3]*8) if anchor else 0
            right = min(image.shape[1], left+int(image.shape[1]*.62))
            upper = image[:max(1, boundary), left:right]
            # Tiny reference headings need the full recognizer's normal path.
            # This fallback is limited to the upper crop, not another page OCR.
            extra,_ = self.ocr(cv2.resize(upper, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),use_cls=False)
            retry = [(np.asarray(q)/2+np.array([left, 0]), text, score) for q, text, score in (extra or [])
                     if label_name(text) == 'Input']
            # Replace an existing uncertain Input anchor rather than duplicate it.
            if retry:
                combined = [item for item in lines if label_name(item[1]) != 'Input'] + retry
                lines = combined
                targets, issues = associate(combined, buttons, image.shape)
        # Locked players are lighter and can include an unlock instruction in
        # the heading. Admit their geometry only with an explicit nearby lock
        # instruction, and mark them locked rather than clickable idle.
        targets,issues=self.include_locked(image,lines,targets,issues)
        if 'Input' not in targets:
            # The reference button can turn blue on hover, putting its gray
            # value above the dark-button threshold. Require the same header,
            # circular control and readable play/pause glyph at the lighter
            # threshold; never infer a player from text alone.
            lighter,_=associate(lines,player_buttons(image,threshold=170),image.shape)
            if 'Input' in lighter:
                targets['Input']=lighter['Input']
                issues=[issue for issue in issues if not issue.startswith('Input:')]
        lines=self.refine_marker_lines(image,lines)
        self.last_task_lines=lines
        self.last_input_plain_text=bool('Input' in targets and any(
            re.sub(r'[^a-z]','',text.lower())=='inputs' and
            abs(float(np.asarray(q)[:,1].min())-targets['Input'].label[1])<3 for q,text,_ in lines))
        return targets, issues, time.perf_counter()-start


def stable_target(first, second, tolerance=5):
    return first.name == second.name and first.state == second.state and all(
        abs(a-b) <= tolerance for a, b in zip(first.button + first.label, second.button + second.label))


def classify_task(targets, lines, shape):
    """Recognize AB-only from positive page evidence, never a missing Input alone."""
    if {'Input','A','B'} <= targets.keys():
        return ('Input','A','B'), ['已确认三个音频播放器']
    if not {'A','B'} <= targets.keys(): return None, ['A/B 播放器尚未完整确认']
    texts=[re.sub(r'[^a-z]','',text.lower()) for _,text,score in lines if score>=.8]
    if any(label_name(text)=='Input' or 'playinput' in text for text in texts):
        return None, ['页面含参考音频提示，继续寻找 Input']
    a,b=targets['A'],targets['B']
    if a.state=='locked': return None, ['A 尚锁定，不能排除需要先播放 Input']
    side_by_side=(abs(a.label[1]-b.label[1])<=max(a.label[3],b.label[3])*2 and
                  b.label[0]-a.label[0]>shape[1]*.12)
    elo=any('omnittselo' in text or 'ttseloevaluation' in text for text in texts)
    inputs=any(text=='inputs' for text in texts)
    transcript=any('hereisthetranscript' in text or 'youshouldadjustthespeaking' in text for text in texts)
    if side_by_side and ((elo and (inputs or transcript)) or (inputs and transcript)):
        return ('A','B'), ['并排 A/B 播放器','已确认 ELO/Inputs/正文提示结构','未发现参考音频或先播放 Input 提示']
    return None, ['尚无足够页面证据确认两段题']


def locate_players(detector, snapshot, mouse, status, required=('Input','A','B')):
    """Find the audio area, including responses below a long transcript."""
    previous = None
    previous_down = False
    downward_steps = 0
    stationary = 0
    ab_confirmations=0
    detector.last_search_profile=[]
    for attempt in range(31):
        began=time.monotonic()
        image = snapshot()
        # At the bottom of the form no player controls exist. A cheap glyph
        # probe avoids two full-page OCR retries on every upward scroll.
        probe=player_buttons(image) or player_buttons(image,threshold=170) if getattr(detector,'fast_search',False) is True else None
        clipped=bool(probe and max(box[1] for box,_ in probe)<image.shape[0]*.28)
        if getattr(detector,'fast_search',False) is True and stationary==0 and (not probe or clipped):
            targets,issues,seconds={},['尚未看到播放器控件'],0
            detector.last_task_lines=[]
        else:
            targets, issues, seconds = detector.detect(image)
        mouse.check()
        detector.last_search_profile.append({'attempt':attempt,'seconds':round(time.monotonic()-began,3),
                                             'players':list(targets),'ocr_seconds':round(seconds,3)})
        if required is None:
            order,evidence=classify_task(targets,getattr(detector,'last_task_lines',[]),image.shape)
            ab_confirmations=ab_confirmations+1 if order==('A','B') else 0
            if order==('Input','A','B') or ab_confirmations>=2:
                detector.last_mode_evidence=evidence
                return image,targets,issues,seconds
            if order==('A','B'):
                status('已发现仅 A/B 题型证据，正在再次核对。')
                mouse.wait(.2)
                verified=snapshot()
                # A second unchanged frame confirms the same complete OCR
                # evidence. Any changed page goes through fresh OCR next time.
                if scene_difference(image,verified)<.15 and all(
                        glyph_state(verified,target.button)==target.state for target in targets.values()):
                    detector.last_mode_evidence=evidence
                    return verified,targets,issues,seconds
                continue
        elif set(required) <= targets.keys():
            return image, targets, issues, seconds
        # A long transcript can push A/B controls below the viewport even
        # while Input is fully visible. Scrolling up then gets stuck at the
        # page top. Move down in small steps while Input still anchors this
        # frame; never merge coordinates from different scroll positions.
        down = 'Input' in targets and not {'A','B'} <= targets.keys()
        if down != previous_down:
            previous = None
            stationary = 0
        if previous is not None:
            stationary = stationary+1 if scene_difference(previous, image) < .5 else 0
        if stationary >= 2 or attempt == 30 or (down and downward_steps >= 6):
            reason = ('连续'+('下滚' if down else '上滚')+'后画面不再移动，可能已到页面边缘或滚轮未生效'
                      if stationary >= 2 else '未能在同一画面确认全部播放器，请缩小远程页面显示比例'
                      if down and downward_steps >= 6 else '达到音频区搜索上限')
            return image, targets, issues+[reason], seconds
        missing = '/'.join(n for n in (required or ('Input','A','B')) if n not in targets)
        status(f'正在向'+('下' if down else '上')+f'寻找音频区（{attempt+1}/30），尚未确认 {missing}；找到后自动开始。')
        previous = image.copy()
        previous_down = down
        downward_steps += int(down)
        # Smaller steps near any known player avoid scrolling past its header.
        notches=12 if not targets and getattr(detector,'fast_search',False) is True else 6 if required is None and {'A','B'}<=targets.keys() else 3 if not targets else 1
        mouse.scroll(image.shape, down=down,notches=1 if down else notches)
    return image,{},issues+['题型连续核对不足，没有确认播放器'],seconds


def scene_difference(first, second):
    if first.shape != second.shape: return float('inf')
    a = cv2.resize(first, (320, 180)).astype(np.float32)
    b = cv2.resize(second, (320, 180)).astype(np.float32)
    return float(np.abs(a-b).mean())


def annotate(image, targets):
    result = image.copy()
    for name, target in targets.items():
        for box, color in ((target.label, (0, 200, 0)), (target.button, (0, 165, 255))):
            x, y, w, h = box
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
        x, y, _, _ = target.button
        cv2.putText(result, f'{name}: {target.state}', (x, max(15, y-8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 100, 220), 2)
    return result
