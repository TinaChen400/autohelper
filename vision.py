"""Local screenshot-only recognition for the sample TTS evaluation layout."""
from dataclasses import dataclass
import re
import time
import cv2
import numpy as np


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


def label_name(text):
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
        self.ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=2)

    def detect(self, image):
        start = time.perf_counter()
        scale = min(1, 1280/image.shape[1], 900/image.shape[0])
        reduced = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        lines, _ = self.ocr(reduced)
        lines = [(np.asarray(q)/scale, text, score) for q, text, score in (lines or [])]
        buttons = player_buttons(image)
        targets, issues = associate(lines, buttons, image.shape)
        # Downscaling can erase one otherwise visible response heading. Retry
        # at native resolution before reporting that its player is absent.
        if scale < 1 and not {'A', 'B'} <= targets.keys():
            native, _ = self.ocr(image)
            if native:
                lines = native
                targets, issues = associate(lines, buttons, image.shape)
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
            extra, _ = self.ocr(cv2.resize(upper, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
            retry = [(np.asarray(q)/2+np.array([left, 0]), text, score) for q, text, score in (extra or [])
                     if label_name(text) == 'Input']
            # Replace an existing uncertain Input anchor rather than duplicate it.
            if retry:
                combined = [item for item in lines if label_name(item[1]) != 'Input'] + retry
                targets, issues = associate(combined, buttons, image.shape)
        # Locked players are lighter and can include an unlock instruction in
        # the heading. Admit their geometry only with an explicit nearby lock
        # instruction, and mark them locked rather than clickable idle.
        if not {'A', 'B'} <= targets.keys():
            gray_targets, _ = associate(lines, player_buttons(image, threshold=170), image.shape)
            for name in ('A', 'B'):
                if name in targets or name not in gray_targets: continue
                target = gray_targets[name]
                lx, ly, lw, lh = target.label
                marker = 'playinput' if name == 'A' else 'playresponseafirst'
                locked = any(marker in re.sub(r'[^a-z]', '', text.lower())
                             and abs(float(np.asarray(q)[:,1].min())-ly) < max(12, lh*1.5)
                             and lx-lh*2 <= float(np.asarray(q)[:,0].min()) <= lx+image.shape[1]*.25
                             and score >= .8 for q,text,score in lines)
                if locked:
                    from dataclasses import replace
                    targets[name] = replace(target, state='locked')
            issues = [issue for issue in issues if issue.split(':')[0] not in targets]
        return targets, issues, time.perf_counter()-start


def stable_target(first, second, tolerance=5):
    return first.name == second.name and first.state == second.state and all(
        abs(a-b) <= tolerance for a, b in zip(first.button + first.label, second.button + second.label))


def locate_players(detector, snapshot, mouse, status):
    """Search upward from any starting scroll position before giving up."""
    previous = None
    stationary = 0
    for attempt in range(31):
        image = snapshot()
        targets, issues, seconds = detector.detect(image)
        mouse.check()
        if {'Input', 'A', 'B'} <= targets.keys():
            return image, targets, issues, seconds
        if previous is not None:
            stationary = stationary+1 if scene_difference(previous, image) < .5 else 0
        if stationary >= 2 or attempt == 30:
            reason = '连续上滚后画面不再移动，可能已到顶部或滚轮未生效' if stationary >= 2 else '达到上滚搜索上限'
            return image, targets, issues+[reason], seconds
        missing = '/'.join(n for n in ('Input','A','B') if n not in targets)
        status(f'正在向上寻找音频区（{attempt+1}/30），尚未确认 {missing}；找到后自动开始。')
        previous = image.copy()
        # Smaller steps near any known player avoid scrolling past its header.
        mouse.scroll(image.shape, down=False, notches=1 if targets else 3)


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
