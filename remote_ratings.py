"""Screenshot rating fill, review window, guarded submit and observation."""
from dataclasses import dataclass
import queue
import threading
import time
import random
import cv2

from rating_reader import read_rating_screen,refresh_selections,judgment_name,submit_enabled,enclosed_form_boxes,marker_changed
from vision import scene_difference


class RatingRecheck(RuntimeError):
    """Pre-press manual choice/uncertainty; no mouse-down has occurred."""


class SubmissionIncomplete(RuntimeError):
    def __init__(self,missing):
        self.missing=missing
        super().__init__('原题仍有未选必填项，补选后重新提交：'+'/'.join(missing))


class ReviewPending(RuntimeError):
    """No new evidence to generate; observe this form without replaying it."""


def preference_conflict(choices):
    factor=choices.get('factor');overall=choices.get('overall')
    preferred=choices.get(factor)
    return bool(overall in ('A','B') and preferred in ('A','B') and overall!=preferred)


def consistent_preference(judgments,choices,sources):
    """Return corrections to assistant-owned overall/factor choices only."""
    overall=choices.get('overall');factor=choices.get('factor')
    model=judgments.get('overall',{})
    factor=factor or model.get('dominant_factor')
    corrections={}
    if sources.get('overall')=='model_auto' and choices.get(factor) in ('A','B'):
        overall=choices[factor]
        if overall!=choices.get('overall'):corrections['overall']=overall
    if sources.get('factor')=='model_auto' and choices.get(factor) in ('A','B') and choices[factor]!=overall:
        weights={key:strength for key,_,strength in model.get('evidence',{}).get('contrasts',[])}
        supported=[key for key in ('quality','speaker','pronunciation','naturalness') if choices.get(key)==overall]
        if supported:
            factor=max(supported,key=lambda key:weights.get(key,0))
            corrections['factor']=factor
    return corrections


def same_layout(before,after,rows,guard_boxes=()):
    if before.shape!=after.shape: return False
    for row in rows:
        x,y,w,h=row['label_box']
        if scene_difference(before[y:y+h,x:x+w],after[y:y+h,x:x+w])>4: return False
    for x,y,w,h in guard_boxes:
        if scene_difference(before[y:y+h,x:x+w],after[y:y+h,x:x+w])>.15: return False
    return bool(rows)


def option_for_judgment(row,judgment):
    return next((name for name in row['options'] if judgment_name(name)==judgment),None)


def ordered_rows(rows):
    return sorted(rows,key=lambda row:(row['label_box'][1],row['label_box'][0]))


def submission_missing_rows(before,current):
    expected={r['key'] for r in before if r.get('selection_state')=='selected'}
    visible=[r for r in current if r['key'] in expected]
    if not visible or any(r.get('selection_state')=='ambiguous' for r in visible):return []
    # A newly loaded task clears every answer. A rejected incomplete form
    # retains some answers and leaves only the missing rows blank.
    if not any(r.get('selection_state')=='selected' for r in visible):return []
    return sorted(r['key'] for r in visible if r.get('selection_state')=='none')


def hidden_earlier_row(rows,record,judgments,observed_keys=None):
    order=['quality','speaker','pronunciation','naturalness','overall','factor']
    visible={row['key'] for row in rows}
    if not visible:return None
    first=min(order.index(key) for key in visible if key in order)
    known=record.data.get('remote_ratings',{})
    for key in order[:first]:
        if key=='speaker' and judgments.get(key,{}).get('judgment')=='不适用':continue
        if key in judgments and (known.get(key,{}).get('judgment') in (None,'需复核')
                                  or observed_keys is not None and key not in observed_keys):
            return key
    return None


def choose_action(row,judgments,attempted):
    key=row.get('key')
    if key in attempted or row.get('selection_state')!='none': return None
    judgment=judgments.get(key,{}).get('judgment') if key!='factor' else judgments.get('overall',{}).get('dominant_factor')
    allowed=('quality','speaker','pronunciation','naturalness') if key=='factor' else ('A','B','Both Good','Both Bad','不适用')
    if judgment not in allowed: return None
    option=option_for_judgment(row,judgment)
    if not option: return None
    return key,judgment,option


def rating_pause(row,judgments,rng,long_audio=False):
    """A bounded reading pause; uncertain evidence gets a little more time."""
    key=row.get('key')
    item=judgments.get('overall' if key=='factor' else key,{})
    evidence=item.get('evidence',{})
    a,b=evidence.get('A'),evidence.get('B')
    close=isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b)<(.02 if key=='speaker' else .15)
    longer=close or long_audio and key in ('pronunciation','overall','factor')
    return rng.uniform(.45,.85) if longer else rng.uniform(.2,.5)


def short_ready_task(flow,judgments):
    readings=getattr(flow,'known_readings',{})
    durations=[getattr(readings.get(name),'total',None) for name in ('A','B')]
    return (all(value is not None and 0<value<=12 for value in durations)
            and all(judgments.get(key,{}).get('judgment') in ('A','B','Both Good','Both Bad','不适用')
                    for key in ('quality','speaker','pronunciation','naturalness','overall'))
            and not any(item.get('uncertainty')=='high_uncertainty' for item in judgments.values()))


def retry_candidates(first,second,judgments,counts,manual_keys):
    """Two clear empty observations, never a selected/ambiguous/manual row."""
    prior={row['key']:row for row in first}
    return [row for row in second if row['key'] not in manual_keys
            and counts.get(row['key'],0)<3
            and prior.get(row['key'],{}).get('selection_state')=='none'
            and row['selection_state']=='none'
            and choose_action(row,judgments,set()) is not None]


def rating_completion(record,screen):
    judgments=(record.data.get('model') or {}).get('judgments',{})
    if not judgments:return None
    chosen={key:value['judgment'] for key,value in record.data.get('remote_ratings',{}).items() if value['judgment']!='需复核'}
    if 'observed_keys' in screen:
        chosen={key:value for key,value in chosen.items() if key in screen['observed_keys']}
    for row in screen['rows']:
        if row.get('selection_state')!='selected':chosen.pop(row['key'],None)
        elif row.get('selected'):chosen[row['key']]=judgment_name(row['selected'])
    required=set(judgments)|{'factor'}
    if judgments.get('speaker',{}).get('judgment')=='不适用' and 'speaker' not in {row['key'] for row in screen['rows']}:
        required.discard('speaker')
    return {'all_required_observed':not (required-chosen.keys()),'submit_enabled_observed':screen.get('submit_enabled',False),
            'missing':sorted(required-chosen.keys()),'submitted_by_assistant':False}


@dataclass
class SelectionTracker:
    expected: dict
    require_click_evidence: bool=False

    def __post_init__(self):
        self.samples={}
        self.previous={}

    def observe(self,rows,manual_clicks=None):
        observations={}
        for row in rows:
            key=row.get('key')
            if not key: continue
            state=row.get('selection_state')
            value=judgment_name(row['selected']) if state=='selected' else '需复核' if state=='none' else None
            if value is None:
                self.samples.pop(key,None);continue
            old,count=self.samples.get(key,(None,0))
            count=count+1 if old==value else 1
            self.samples[key]=(value,count)
            if count<2: continue
            if value=='需复核' and key not in self.previous and key not in self.expected: continue
            clicked=(manual_clicks or {}).get(key)
            source=('model_auto' if self.expected.get(key)==value and clicked!=value else
                    'human_remote' if not self.require_click_evidence or clicked==value or
                        value=='需复核' and key in (manual_clicks or {}) else 'existing_remote')
            item={'judgment':value,'source':source}
            if self.previous.get(key)!=item:
                observations[key]=item
                self.previous[key]=item
        return observations


class RatingController:
    def __init__(self,flow):
        self.flow=flow
        self.record=flow.comparison_record
        manual={key for key,value in self.record.data.get('remote_ratings',{}).items() if value['source']!='model_auto'}
        self.expected={action['dimension']:action['judgment'] for action in self.record.data.get('auto_actions',[]) if action['dimension'] not in manual}
        self.attempted=set()
        self.attempt_counts={}
        self.rng=getattr(flow,'rng',None) or random.Random()
        self.input_revision=getattr(flow.mouse,'interference_count',0) if hasattr(flow,'mouse') else 0
        self.tracker=SelectionTracker(self.expected,getattr(flow,'strict_source_attribution',False) is True)

    def evidence(self,image,prefix):
        path=self.flow.audio_folder/f'{prefix}_{time.time_ns()}.png'
        cv2.imwrite(str(path),image)
        return path

    def store(self,image,rows):
        if getattr(self.flow,'require_current_rating_scan',False) is True:
            self.observed_keys=getattr(self,'observed_keys',set())|{row['key'] for row in rows}
        observations=self.tracker.observe(rows)
        latest={action['dimension']:action['state'] for action in self.record.data.get('auto_actions',[])}
        observations={key:value for key,value in observations.items()
                      if not (value['judgment']=='需复核' and latest.get(key) in ('planned','unverified')
                              and self.record.data.get('remote_ratings',{}).get(key,{}).get('judgment') in (None,'需复核'))}
        if observations:
            evidence=self.evidence(image,'remote_choice')
            self.record.save_remote(observations,evidence)
            for key,value in observations.items():
                if value['source']=='model_auto' and latest.get(key) in ('planned','unverified'):
                    self.record.save_auto_action(key,value['judgment'],'verified',evidence)

    def publish(self,screen,image):
        callback=getattr(self.flow,'publish_rating_screen',None)
        if callback:callback(screen,image)

    def completion(self,screen):
        if getattr(self.flow,'require_current_rating_scan',False) is True:
            screen={**screen,'observed_keys':getattr(self,'observed_keys',set())}
        return rating_completion(self.record,screen)

    def guard_question(self):
        guard=getattr(self.flow,'guard_rating_action',None)
        if guard:guard()

    def apply_row(self,image,row,judgments,defer_verification=False,retry=False):
        prior=self.record.data.get('remote_ratings',{}).get(row['key'],{})
        replacement=None
        if row.get('selection_state')=='selected' and prior.get('source')=='model_auto':
            proposed=choose_action({**row,'selection_state':'none'},judgments,set())
            if proposed and judgment_name(row['selected'])!=proposed[1]:replacement=row['selected']
        action=choose_action({**row,'selection_state':'none'} if replacement else row,
                             judgments,set() if retry or replacement else self.attempted)
        if action is None: return image
        key,judgment,option=action
        if self.attempt_counts.get(key,0)>=3:return image
        if retry and (getattr(self.flow.mouse,'interference_count',0)!=self.input_revision or
                getattr(self.flow,'repair_empty_required',False) is not True and
                self.record.data.get('remote_ratings',{}).get(key,{}).get('source')=='human_remote'):
            return image
        self.attempted.add(key)
        evidence=self.evidence(image,'before_auto_rating')
        # Journal before mouse-down: a crash must not turn our click into a
        # supposedly independent human label when the recording is reopened.
        self.record.save_auto_action(key,judgment,'planned',evidence)
        def before_press():
            self.guard_question()
            latest=self.flow.snapshot()
            if retry and getattr(self.flow.mouse,'interference_count',0)!=self.input_revision:
                raise RatingRecheck('人工操作后已取消本次补点，先复查当前网页选择。')
            if scene_difference(image,latest)>2:
                raise RuntimeError('评价按下前画面变化，取消代选，请核对当前题目。')
            current=refresh_selections(latest,[row])[0]
            if (replacement and (current['selection_state']!='selected' or current['selected']!=replacement)) or (
                    not replacement and current['selection_state']!='none'):
                raise RatingRecheck('你已选择或选项状态变化，取消本项代选，保留网页上的选择。')
        text_box=row['options'][option]
        candidate=row.get('selection_evidence',{}).get(option,{}).get('button_box',text_box)
        # A detected border must contain the text and exclude neighboring
        # options. Otherwise use the smaller text interior for this click.
        x,y,w,h=candidate
        tx,ty,tw,th=text_box
        isolated=all(name==option or bx>=x+w or bx+bw<=x or by>=y+h or by+bh<=y
                     for name,(bx,by,bw,bh) in row['options'].items())
        box=candidate if isolated and x<=tx and y<=ty and x+w>=tx+tw and y+h>=ty+th else text_box
        self.flow.mouse.click_box(box,image.shape,before_press=before_press,
                                  quick=short_ready_task(self.flow,judgments))
        self.attempt_counts[key]=self.attempt_counts.get(key,0)+1
        self.flow.log('model_rating_click',dimension=key,judgment=judgment,
                      click=getattr(self.flow.mouse,'last_click',None))
        if replacement:self.flow.log('model_rating_correction',dimension=key,before=judgment_name(replacement),after=judgment)
        if retry:self.flow.log('model_rating_retry',dimension=key,attempt=self.attempt_counts[key],reason='两次核对均为空选')
        self.expected[key]=judgment
        longest=max((getattr(r,'total',0) or 0 for r in getattr(self.flow,'known_readings',{}).values()),default=0)
        pause=rating_pause(row,judgments,self.rng,long_audio=longest>=15)
        self.flow.log('rating_reading_pause',dimension=key,seconds=round(pause,3))
        self.flow.mouse.wait(pause)
        if defer_verification:
            # Independent rating rows can be clicked consecutively. Before
            # each press, still check that row is blank and the page is stable.
            # Keep the planned journal until the shared two-frame verification.
            return self.flow.snapshot()
        # Move off the button before checking its appearance; hover is not a
        # selected state, and the remote pointer can cover the text/border.
        # Clear hover near the row rather than traversing the whole page to a
        # fixed parking point after every choice.
        label=row['label_box']
        rng=getattr(self.flow,'rng',None)
        fraction=rng.uniform(.45,.58) if rng else .52
        self.flow.mouse.move((image.shape[1]*fraction,label[1]+label[3]/2),image.shape,quick=True)
        deadline=time.monotonic()+1
        matched=0
        while time.monotonic()<deadline:
            latest=self.flow.snapshot()
            if not same_layout(image,latest,[row]): break
            current=refresh_selections(latest,[row])[0]
            matched=matched+1 if current['selected']==option and current['selection_state']=='selected' else 0
            if matched>=2:
                self.expected[key]=judgment
                self.tracker.previous[key]={'judgment':judgment,'source':'model_auto'}
                self.record.save_remote({key:self.tracker.previous[key]},self.evidence(latest,'auto_rating_verified'))
                self.record.save_auto_action(key,judgment,'verified',evidence)
                return latest
            self.flow.mouse.wait(.12)
        self.record.save_auto_action(key,judgment,'unverified',self.evidence(latest,'auto_rating_unverified'))
        # Unconfirmed clicks remain attributable to the assistant. Do not retry
        # toggles or mislabel a delayed remote render as the user's choice.
        self.expected[key]=judgment
        self.flow.remote_status=f'{row["name"]} 已点选但未看清选中状态；请在网页核对，不重复点击。'
        return latest

    def recover_rows(self,image,screen,judgments):
        """Bounded read/retry after the shared verification, with no submit path."""
        rows=screen['rows']
        for recovery in range(2):
            missing=self.completion({**screen,'rows':rows})['missing']
            missing=[key for key in missing if any(r['key']==key and
                choose_action({**r,'selection_state':'none'},judgments,set()) for r in rows)]
            if not missing:break
            revision=getattr(self.flow.mouse,'interference_count',0)
            self.flow.mouse.move((image.shape[1]*self.rng.uniform(.4,.6),
                                  image.shape[0]*self.rng.uniform(.08,.12)),image.shape)
            self.flow.mouse.wait(self.rng.uniform(.25,.45))
            latest=self.flow.snapshot()
            if not same_layout(image,latest,rows,screen.get('guard_boxes',())):
                raise RuntimeError('补点核对时页面变化，停止本题。')
            first=refresh_selections(latest,rows)
            # Re-detect unclear text/borders once per recovery pass. A fresh
            # OCR result supplies actual geometry, never a guessed offset.
            if any(row['key'] in missing and row['selection_state']=='ambiguous' for row in first):
                refreshed=read_rating_screen(self.flow.detector,latest)
                before={r['key'] for r in rows}
                after={r['key'] for r in refreshed['rows']}
                if not before<=after:
                    self.flow.log('rating_retry_skipped',reason='重新识别的评价行不完整')
                    break
                refreshed={**refreshed,'rows':[row for row in refreshed['rows'] if row['key'] in before]}
                expected=dict(getattr(self.flow,'question_marker',()))
                if marker_changed(expected,refreshed.get('question_marker',())):
                    raise RuntimeError('补点前题目编号变化，停止本题。')
                rows=refreshed['rows']
                first=refresh_selections(latest,rows)
                screen=refreshed
            self.store(latest,first)
            self.flow.mouse.wait(self.rng.uniform(.25,.45))
            second_image=self.flow.snapshot()
            if not same_layout(latest,second_image,rows,screen.get('guard_boxes',())):
                raise RuntimeError('补点按下前页面变化，停止本题。')
            second=refresh_selections(second_image,rows)
            self.store(second_image,second)
            image=second_image
            if getattr(self.flow.mouse,'interference_count',0)!=revision:
                self.flow.log('rating_retry_skipped',reason='复查期间人工仍在操作，等待下一次复查')
                rows=second
                continue
            # Two current observations after the normal mouse guard has
            # waited for release/stability. Selected or manually cleared rows
            # remain protected; moving the pointer alone must not disable all
            # remaining empty rows for the rest of this question.
            if revision!=self.input_revision:
                self.flow.log('rating_manual_input_rechecked',revision=revision)
                self.input_revision=revision
            empty={row['key'] for row in second if row['selection_state']=='none'} if getattr(self.flow,'repair_empty_required',False) is True else set()
            manual={k for k,v in self.record.data.get('remote_ratings',{}).items() if v['source']!='model_auto' and k not in empty}
            candidates=ordered_rows(retry_candidates(first,second,judgments,self.attempt_counts,manual))
            for row in candidates:
                try:image=self.apply_row(image,row,judgments,defer_verification=True,retry=True)
                except RatingRecheck as exc:
                    self.flow.log('rating_row_recheck',dimension=row['key'],reason=str(exc))
                    image=self.flow.snapshot()
                    if not same_layout(second_image,image,rows,screen.get('guard_boxes',())):
                        raise RuntimeError('人工操作后评价布局变化，停止本题。') from exc
            rows=second
            self.flow.mouse.move((image.shape[1]*self.rng.uniform(.055,.075),
                                  image.shape[0]*self.rng.uniform(.4,.6)),image.shape)
            for sample in range(2):
                self.flow.mouse.wait(.25)
                latest=self.flow.snapshot()
                if not same_layout(image,latest,rows):raise RuntimeError('补点后页面变化，停止本题。')
                rows=refresh_selections(latest,rows)
                self.store(latest,rows)
                image=latest
            self.flow.log('rating_recovery_check',pass_number=recovery+1,remaining=self.completion({**screen,'rows':rows})['missing'])
        return image,{**screen,'rows':rows,'submit_enabled':submit_enabled(image,screen['submit_boxes'])}

    def reconcile_preferences(self,image,screen,judgments):
        rows=refresh_selections(image,screen['rows'])
        self.store(image,rows)
        self.flow.mouse.wait_observation(.12)
        latest=self.flow.snapshot()
        if not same_layout(image,latest,rows,screen.get('guard_boxes',())):
            raise RuntimeError('评价一致性核对时页面变化，重新识别当前题。')
        image=latest;rows=refresh_selections(image,rows);self.store(image,rows)
        remote=self.record.data.get('remote_ratings',{})
        choices={key:item['judgment'] for key,item in remote.items()}
        sources={key:item['source'] for key,item in remote.items()}
        for row in rows:
            if row.get('selection_state')=='selected':choices[row['key']]=judgment_name(row['selected'])
        corrections=consistent_preference(judgments,choices,sources)
        revised={key:dict(value) for key,value in judgments.items()}
        for key,value in corrections.items():
            if key=='factor':revised.setdefault('overall',{})['dominant_factor']=value
            else:revised.setdefault(key,{})['judgment']=value
        for row in rows:
            if row['key'] in corrections:
                image=self.apply_row(image,row,revised)
                rows=refresh_selections(image,rows)
                choices[row['key']]=judgment_name(next(r for r in rows if r['key']==row['key'])['selected'])
        conflicts=preference_conflict(choices)
        self.record.update(preference_consistency={'checked_at':time.time(),'choices':choices,
            'consistent':not conflicts,'corrections':corrections})
        self.flow.log('preference_consistency',consistent=not conflicts,corrections=corrections,choices=choices)
        return image,{**screen,'rows':rows},conflicts

    def run(self):
        flow=self.flow
        while not flow.analysis_done.is_set():
            flow.status('补听预算已用完，等待后台分析；不会继续循环点击。')
            flow.mouse.wait(.25)
        early=getattr(flow,'rating_observer',None)
        began=time.monotonic()
        if early:
            early.stopped.set()
            early.thread.join(timeout=5)
            if early.thread.is_alive():
                flow.remote_status='网页记录进程尚未停止，本轮不代选，已保留模型结果。';return
        flow.log('rating_phase_timing',phase='observer_stop',seconds=round(time.monotonic()-began,3))
        if getattr(flow,'remote_submitted',False) or getattr(flow,'remote_question_changed',False):
            flow.status('本题已经提交或页面已换题，保留记录，不再代选。');return
        identity=getattr(flow,'ensure_rating_identity',None)
        if identity:identity()
        judgments=(self.record.data.get('model') or {}).get('judgments',{})
        if not any(item.get('judgment') in ('A','B','Both Good','Both Bad','不适用') for item in judgments.values()):
            flow.remote_status='没有可代选的模型结果；'+(self.record.data.get('model') or {}).get('reason','请查看录音与分析状态。')
            flow.status(flow.remote_status)
            # Continue read-only human feedback without scrolling/clicking a
            # form that has no model votes available.
            if getattr(flow,'last_image',None) is not None:
                observer=RemoteRatingObserver(flow,{'rows':[],'submit_boxes':[]},flow.last_image,{})
                flow.rating_observer=observer;observer.start()
            return
        # Move the verified audio controls toward the upper part of the view
        # before OCR. This is navigation only: every option is located anew.
        # Scanning at each tiny wheel movement dominated the old fill time.
        targets=getattr(flow,'targets',{})
        if targets:
            began=time.monotonic()
            image=flow.snapshot()
            bottom=max(t.button[1]+t.button[3] for t in targets.values())
            height=max(t.button[3] for t in targets.values())
            notches=max(0,min(12,round((bottom-image.shape[0]*.20)/max(1,height*2))))
            if notches:flow.mouse.scroll(image.shape,down=True,notches=notches)
            if getattr(flow,'fast_navigation',False) is True and not enclosed_form_boxes(flow.snapshot()):
                # A clipped form needs more navigation, not a full OCR pass.
                # This bounded visual check never supplies option coordinates.
                flow.mouse.scroll(image.shape,down=True,notches=6)
            flow.log('rating_phase_timing',phase='navigation',seconds=round(time.monotonic()-began,3))
        stationary=0
        layout_retries=0
        review_done=False
        short_ready=short_ready_task(flow,judgments)
        review_planned=(not short_ready and getattr(flow,'mid_rating_review_enabled',False) is True
                        and self.rng.random()<.35)
        flow.log('rating_path_selected',short_ready=short_ready,mid_review_planned=review_planned)
        for step in range(flow.settings.max_scrolls+1):
            if getattr(flow,'fast_navigation',False) is True:
                # The wheel anchor can hover over Response B and obscure its
                # label in the remote image. Clear it before every OCR pass.
                shape=flow.last_image.shape
                flow.mouse.move((shape[1]*flow.rng.uniform(.055,.075),
                                 shape[0]*flow.rng.uniform(.4,.6)),shape,quick=True)
                flow.mouse.wait(.1)
            image=flow.snapshot()
            began=time.monotonic()
            screen=read_rating_screen(flow.detector,image)
            self.publish(screen,image)
            flow.log('rating_phase_timing',phase='recognition',seconds=round(time.monotonic()-began,3),step=step)
            expected=getattr(flow,'question_marker',())
            if marker_changed(expected,screen.get('question_marker',())):
                flow.remote_question_changed=True
                flow.remote_status='题目编号变化，已停止代选，保留上一题记录。';return
            rows=ordered_rows(screen['rows'])
            absent=[row['key'] for row in rows if row.get('selection_state')=='none'
                    and (wanted:=judgments.get(row['key'],{}).get('judgment')) in ('A','B','Both Good','Both Bad','不适用')
                    and option_for_judgment(row,wanted) is None]
            if absent:
                flow.log('rating_option_ocr_recheck',dimensions=absent)
                flow.mouse.wait(.25)
                image=flow.snapshot()
                screen=read_rating_screen(flow.detector,image)
                self.publish(screen,image)
                rows=ordered_rows(screen['rows'])
            if not rows:
                if step==flow.settings.max_scrolls or stationary>=2: break
                flow.mouse.scroll(image.shape,down=True,notches=flow.rng.randint(1,3))
                after=flow.snapshot()
                stationary=stationary+1 if scene_difference(image,after)<.5 else 0
                continue
            earlier=hidden_earlier_row(rows,self.record,judgments,
                getattr(self,'observed_keys',set()) if getattr(flow,'require_current_rating_scan',False) is True else None) if getattr(flow,'strict_rating_order',False) is True else None
            if earlier:
                flow.log('rating_order_navigation',missing=earlier)
                flow.mouse.scroll(image.shape,down=False,notches=2)
                continue
            flow.mouse.wait(.12)
            latest=flow.snapshot()
            if not same_layout(image,latest,rows):
                self.evidence(image,'rating_layout_before');self.evidence(latest,'rating_layout_after')
                flow.log('rating_layout_recheck',attempt=layout_retries+1)
                layout_retries+=1
                if layout_retries<=2:continue
                flow.remote_status='评价区域连续变化，已停止代选；请重新查看当前题目。';return
            layout_retries=0
            rows=refresh_selections(latest,rows)
            self.store(image,screen['rows']);self.store(latest,rows)
            image=latest
            began=time.monotonic()
            return_to_form=False
            for row in rows:
                try:
                    image=self.apply_row(image,row,judgments,defer_verification=False)
                    if row['key'] in self.attempted:
                        checked=refresh_selections(image,[row])[0]
                        if checked['selection_state']!='selected':
                            image,recovered=self.recover_rows(image,{**screen,'rows':[row]},judgments)
                            if recovered['rows'][0]['selection_state']!='selected':
                                raise RuntimeError(f'{row["name"]} 尚未确认选中，停止后续评价和提交。')
                except RatingRecheck as exc:
                    flow.log('rating_row_recheck',dimension=row['key'],reason=str(exc))
                    image=flow.snapshot()
                    if not same_layout(latest,image,rows,screen.get('guard_boxes',())):
                        raise RuntimeError('人工操作后评价布局变化，停止本题。') from exc
                    if getattr(flow,'strict_rating_order',False) is True:
                        image,recovered=self.recover_rows(image,{**screen,'rows':[row]},judgments)
                        if recovered['rows'][0]['selection_state']!='selected':
                            raise RuntimeError(f'{row["name"]} 还未确认选中，先重新识别本项。')
                if review_planned and not review_done and len(self.attempted)>=2 and step<flow.settings.max_scrolls:
                    # Confirm prior selections before navigating away. Never
                    # reuse form coordinates after returning from the players.
                    review_done=True
                    flow.mid_rating_review()
                    self.tracker.samples.clear()
                    return_to_form=True
                    break
            if return_to_form:continue
            flow.log('rating_phase_timing',phase='click_dispatch',seconds=round(time.monotonic()-began,3),step=step)
            verification_began=time.monotonic()
            if getattr(flow,'fast_navigation',False) is True:
                flow.mouse.move((image.shape[1]*flow.rng.uniform(.055,.075),
                                 image.shape[0]*flow.rng.uniform(.4,.6)),image.shape,quick=True)
            # Delayed remote paints are checked together, rather than blocking
            # the next independent row for three seconds after every click.
            deadline=time.monotonic()+3
            while True:
                latest=flow.snapshot()
                if not same_layout(image,latest,rows):
                    raise RuntimeError('最终核对时评价区域变化，已停止本题。')
                rows=refresh_selections(latest,rows)
                self.store(latest,rows)
                image=latest
                # Require two current observations, including when resuming a
                # record that already contains a previously verified choice.
                # A stable human override is also resolved and never retried.
                pending={row['key'] for row in rows if row['key'] in self.expected
                         and (row['selection_state']!='selected'
                              or self.tracker.samples.get(row['key'],(None,0))[1]<2)}
                if not pending or time.monotonic()>=deadline:break
                flow.mouse.wait(.12)
            flow.log('rating_phase_timing',phase='click_and_verify',seconds=round(time.monotonic()-began,3),step=step)
            flow.log('rating_phase_timing',phase='selection_verification',seconds=round(time.monotonic()-verification_began,3),step=step)
            recovery_began=time.monotonic()
            image,screen=self.recover_rows(image,{**screen,'rows':rows},judgments)
            rows=screen['rows']
            flow.log('rating_phase_timing',phase='recovery',seconds=round(time.monotonic()-recovery_began,3),step=step)
            screen.update(rows=refresh_selections(image,rows),submit_enabled=submit_enabled(image,screen['submit_boxes']))
            self.publish(screen,image)
            flow.send('dimensions',screen['rows'])
            flow.log('rating_page',step=step,dimensions=screen['rows'],submit_visible=bool(screen['submit_boxes']))
            if step==flow.settings.max_scrolls or stationary>=2: break
            if screen['submit_boxes']:
                # A taller/new layout may hide upper required rows after the
                # initial scroll. Recover them instead of claiming completion.
                missing=self.completion(screen)['missing']
                order=['quality','speaker','pronunciation','naturalness','overall','factor']
                visible=[order.index(r['key']) for r in rows if r.get('key') in order]
                hidden=[order.index(key) for key in missing if key in order and key not in self.attempted]
                if not visible or not hidden or min(hidden)>=min(visible):break
                flow.mouse.scroll(image.shape,down=False,notches=3)
            else:
                flow.mouse.scroll(image.shape,down=True,notches=flow.rng.randint(3,5))
            after=flow.snapshot()
            stationary=stationary+1 if scene_difference(image,after)<.5 else 0
        if not screen['rows']:
            flow.remote_status='未看清评价区域，已保存模型结果；本轮未代选。';return
        verified=len({item['dimension'] for item in self.record.data.get('auto_actions',[]) if item['state']=='verified'})
        completion=self.completion(screen)
        if getattr(flow,'require_current_rating_scan',False) is True:
            screen={**screen,'observed_keys':getattr(self,'observed_keys',set())}
        missing=completion['missing']
        ready=completion['all_required_observed'] and completion['submit_enabled_observed']
        conflict=False
        if ready:
            image,screen,conflict=self.reconcile_preferences(image,screen,judgments)
            ready=ready and not conflict
        self.record.update(rating_completion=completion)
        from audio_lab.performance import write_performance
        if (flow.audio_folder/'pipeline_status.json').exists() and (flow.directory/'events.jsonl').exists():
            timing=write_performance(flow.audio_folder,flow.directory)
            if timing:self.record.update(performance=timing)
        flow.remote_status=(f'本题必填项已选齐并核对（本轮代选 {verified} 项）；3–5 秒后自动 Submit，无需确认。' if ready else
            f'本轮确认代选 {verified} 项；尚未核对齐：'+('/'.join(sorted(missing)) or 'Submit 未启用')+'。已保存依据，不会提交。')
        observer=RemoteRatingObserver(flow,screen,image,self.expected)
        flow.rating_observer=observer
        observer.start()
        flow.status(flow.remote_status)
        unresolved=[key for key in missing if judgments.get(key,{}).get('judgment')=='需复核']
        model=self.record.data.get('model') or {}
        errors=model.get('pipeline',{}).get('stage_errors',{})
        if unresolved and errors:
            # A failed/missing input is recoverable work, not a completed
            # independent review. Let the supervisor re-read this question
            # and rebuild from saved audio instead of waiting for the user.
            raise RuntimeError('评价尚未生成完整，正在恢复：'+'；'.join(f'{key}: {value}' for key,value in errors.items()))
        if conflict or unresolved and getattr(flow,'bounded_review',False) is True:
            flow.pending_review=True
            flow.remote_status=('最终偏好与主导因素不一致，保留人工选择，等待核对。' if conflict else
                '已有证据仍无法判断：'+('/'.join(sorted(unresolved)))+'；保留录音与已选项，等待这些项确认。')
            raise ReviewPending(flow.remote_status)
        if ready and getattr(flow,'auto_submit',False):
            self.submit_after_review(observer)

    def submit_after_review(self,observer):
        flow=self.flow
        delay=flow.rng.uniform(3,5)
        flow.remote_status=f'答案已选齐，{delay:.1f} 秒后自动 Submit；现在可以直接修改网页答案。'
        flow.status(flow.remote_status)
        flow.log('submit_review_wait',seconds=delay)
        deadline=time.monotonic()+delay
        while time.monotonic()<deadline:
            flow.mouse.wait_observation(min(.1,max(0,deadline-time.monotonic())))
            if flow.remote_submitted or flow.remote_question_changed:break
        observer.stopped.set()
        observer.thread.join(timeout=5)
        observer.input_thread.join(timeout=5)
        if observer.thread.is_alive() or observer.input_thread.is_alive():
            raise RuntimeError('答案观察尚未停止，未自动提交。')
        if flow.remote_submitted:return
        if flow.remote_question_changed:
            raise RuntimeError('提交前页面已变化，未自动提交。')
        flow.mouse.check()
        image=flow.snapshot()
        screen={**observer.screen}
        if not same_layout(observer.image,image,observer.screen['rows'],observer.screen.get('guard_boxes',())):
            raise RuntimeError('提交前评价布局变化，未自动提交。')
        # Selected buttons use white text on a dark fill. A new OCR pass can
        # omit that label; retain verified option geometry on this unchanged
        # layout and read its actual pixels rather than treating it as empty.
        # Save the current human choices without applying any model votes again.
        for _ in range(2):
            image=flow.snapshot()
            rows=refresh_selections(image,screen['rows'])
            self.store(image,rows)
            flow.mouse.wait_observation(.12)
        if not self.completion({**screen,'rows':rows,'submit_enabled':submit_enabled(image,screen['submit_boxes'])})['all_required_observed']:
            image,repaired=self.recover_rows(image,{**screen,'rows':rows},
                (self.record.data.get('model') or {}).get('judgments',{}))
            rows=repaired['rows']
            if not self.completion(repaired)['all_required_observed']:
                raise RuntimeError('正在补选未确认的必填项，选齐后自动提交。')
        image,reconciled,conflict=self.reconcile_preferences(image,{**screen,'rows':rows},
            (self.record.data.get('model') or {}).get('judgments',{}))
        rows=reconciled['rows']
        if conflict:
            following=RemoteRatingObserver(flow,reconciled,image,self.expected)
            flow.rating_observer=following;following.start()
            flow.pending_review=True
            raise ReviewPending('网页最终偏好与主导因素不一致，等待核对，不提交矛盾答案。')
        boxes=screen['submit_boxes']
        if len(boxes)!=1:raise RuntimeError('Submit 按钮未唯一确认，未自动提交。')
        def before_press():
            self.guard_question()
            latest=flow.snapshot()
            if not same_layout(image,latest,rows,screen.get('guard_boxes',())) or not submit_enabled(latest,boxes):
                raise RuntimeError('Submit 点击前页面变化或按钮未启用，未提交。')
            if any(a.get('selected')!=b.get('selected') or b.get('selection_state')!='selected'
                   for a,b in zip(rows,refresh_selections(latest,rows))):
                raise RuntimeError('答案仍在变化，未自动提交。')
            note=getattr(flow,'note_submission',None)
            if note:note('assistant',{**screen,'rows':rows},latest,'verified_auto_submit_press')
        flow.mouse.click_box(boxes[0],image.shape,before_press=before_press)
        flow.submitted_image=image.copy()
        flow.submitted_screen={**screen,'rows':rows}
        flow.remote_submitted=True
        if not callable(getattr(flow,'note_submission',None)):
            self.record.save_remote({},self.evidence(image,'auto_submit'),submit_clicked=True,confirm_auto=False)
        self.record.update(submission_observation={**self.record.data.get('submission_observation',{}),
            'actor':'assistant','note':'助手点击 Submit；等待新题刷新确认处理结果。'})
        self.record.update(rating_completion={**self.record.data.get('rating_completion',{}),'submitted_by_assistant':True})
        flow.log('auto_submit_clicked')


class RemoteRatingObserver:
    """No mouse/keyboard injection; cached row pixels avoid repeated OCR."""
    def __init__(self,flow,screen,image,expected):
        self.flow=flow;self.screen=screen;self.image=image
        self.tracker=SelectionTracker(dict(expected),getattr(flow,'strict_source_attribution',False) is True)
        self.clicks=queue.Queue()
        self.manual_clicks={}
        self.detector=None
        self.geometry=None
        self.stopped=threading.Event()
        self.guard=threading.Lock()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.input_thread=threading.Thread(target=self.watch_clicks,daemon=True)

    def start(self):
        self.input_thread.start();self.thread.start()

    def watch_clicks(self):
        from assistant import u,client
        from dynamic_app import foreground
        from human_mouse import HumanMouse
        held=False
        while not self.stopped.wait(.02) and not self.flow.event.is_set():
            if u.GetAsyncKeyState(0x1B)&0x8000:
                self.flow.event.set();self.stopped.set()
                self.flow.remote_status='已停止网页答案观察（Esc）。'
                return
            down=bool(u.GetAsyncKeyState(0x01)&0x8000)
            if down and not held and foreground(self.flow.hwnd):
                x,y,w,h=client(self.flow.hwnd)
                px,py=HumanMouse.position()
                with self.guard:
                    if self.geometry==(x,y,w,h):
                        shape=self.image.shape
                        point=((px-x)/w*shape[1],(py-y)/h*shape[0])
                        self.clicks.put((point,time.monotonic(),self.screen))
            held=down

    @staticmethod
    def inside(point,box,pad=0):
        x,y,w,h=box;px,py=point
        return x-pad<=px<=x+w+pad and y-pad<=py<=y+h+pad

    def run(self):
        flow=self.flow
        last_evidence=flow.audio_folder/'page.png'
        try:
            while not self.stopped.is_set() and not flow.event.is_set():
                submitted=False
                while not self.clicks.empty():
                    point,clicked,screen=self.clicks.get_nowait()
                    if screen.get('submit_enabled') and any(self.inside(point,box,8) for box in screen['submit_boxes']): submitted=True
                    for row in screen['rows']:
                        for option,box in row['options'].items():
                            if self.inside(point,box,3): self.manual_clicks[row['key']]=judgment_name(option)
                if submitted:
                    pending={key:value for key,value in self.manual_clicks.items()
                             if self.tracker.previous.get(key,{}).get('judgment')!=value}
                    note=getattr(flow,'note_submission',None)
                    if callable(note):note('user',screen,self.image,'rating_observer_click')
                    else:flow.comparison_record.save_remote({},last_evidence,submit_clicked=True,confirm_auto=not pending)
                    if pending: flow.comparison_record.update(unverified_user_clicks=pending)
                    flow.remote_status='已观察到你点击 Submit，最后已看清的答案已记录；本题观察结束。'
                    flow.remote_submitted=True
                    flow.submitted_screen=screen
                    flow.submitted_image=self.image.copy()
                    break
                try:
                    image,geometry=flow.grab(flow.hwnd)
                except Exception:
                    # Read-only observation resumes when the selected window is
                    # visible again; it never reactivates or clicks the window.
                    self.stopped.wait(.25);continue
                if same_layout(self.image,image,self.screen['rows'],self.screen.get('guard_boxes',())):
                    screen={**self.screen,'rows':refresh_selections(image,self.screen['rows']),
                            'submit_enabled':submit_enabled(image,self.screen['submit_boxes'])}
                else:
                    if self.detector is None:
                        from vision import Detector
                        self.detector=Detector()
                    screen=read_rating_screen(self.detector,image)
                    marker=screen.get('question_marker')
                    expected=getattr(flow,'question_marker',None)
                    if marker_changed(expected,marker):
                        flow.remote_question_changed=True
                        flow.remote_status='题目编号已变化，本题记录结束，未继承旧答案。';break
                    if not screen['rows']:
                        self.stopped.wait(.5);continue
                    self.tracker.samples.clear()
                previous=flow.comparison_record.data.get('remote_ratings',{})
                visible=[row for row in screen['rows'] if row['key'] in previous and previous[row['key']]['judgment']!='需复核']
                if len(visible)>=2 and all(row['selection_state']=='none' for row in visible):
                    flow.remote_question_changed=True
                    flow.remote_status='多项评价同时清空，可能已换题；本题观察结束，未继承旧答案。';break
                observations=self.tracker.observe(screen['rows'],self.manual_clicks)
                if observations:
                    last_evidence=flow.audio_folder/f'remote_human_{time.time_ns()}.png'
                    cv2.imwrite(str(last_evidence),image)
                    flow.comparison_record.save_remote(observations,last_evidence)
                    latest={action['dimension']:action['state'] for action in flow.comparison_record.data.get('auto_actions',[])}
                    for key,value in observations.items():
                        if value['source']=='model_auto' and latest.get(key) in ('planned','unverified'):
                            flow.comparison_record.save_auto_action(key,value['judgment'],'verified',last_evidence)
                    flow.remote_status='已自动记录网页答案及分歧；你可直接修改，助手会自动 Submit，无需确认。'
                completion=rating_completion(flow.comparison_record,screen)
                if completion and completion!=flow.comparison_record.data.get('rating_completion'):
                    flow.comparison_record.update(rating_completion=completion)
                if completion and completion['all_required_observed'] and completion['submit_enabled_observed']:
                    flow.remote_status='本题必填项已核对；3–5 秒后助手自动 Submit，你也可以提前提交。'
                with self.guard: self.screen,self.image,self.geometry=screen,image,geometry
                publisher=getattr(flow,'publish_rating_screen',None)
                if publisher:publisher(screen,image)
                self.stopped.wait(.2)
        except Exception as exc:
            flow.remote_status='远程答案观察已停止：'+str(exc)
        finally:
            self.stopped.set()
