"""Optional integration entry point; frozen click implementation stays unchanged."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import threading
import time
import os
from tkinter import ttk
from tkinter import filedialog
from types import SimpleNamespace
import tkinter as tk
import cv2

from dynamic_app import DynamicApp, grab, foreground
from workflow import Workflow, Settings
from playback import PlaybackEvidenceError
from audio_lab.resilient_tracker import ResilientTracker
from audio_lab.comparison_record import ComparisonRecord, DIMENSIONS
from audio_lab.durable_io import write_json

ROOT=Path(__file__).resolve().parent
AUDIO=ROOT/'audio_lab'


class RecordedWorkflow(Workflow):
    tracker_type=ResilientTracker
    auto_submit=True
    mid_rating_review_enabled=True
    strict_rating_order=True
    repair_empty_required=True
    require_current_rating_scan=True
    strict_source_attribution=True
    bounded_review=True

    def log(self,kind,**data):
        folder=getattr(self,'audio_folder',None)
        if folder:data.setdefault('recording_session',folder.name)
        super().log(kind,**data)
        progress=getattr(self,'question_progress',None)
        if progress is not None:progress.observe(kind,**data)
        if kind in ('continuous_recovery','capture_repair_pending','review_pending') and getattr(self,'last_image',None) is not None:
            cv2.imwrite(str(self.directory/f'{kind}_last_frame.png'),self.last_image)
        from health_monitor import EVENTS
        if kind in EVENTS:
            self.send('health_event',{'event':kind,'data':{**data,
                'task_id':getattr(self,'task_tracking',{}).get('task_id')},'at':time.monotonic()})

    def status(self,text):
        self.worker_status=text
        super().status(text)

    def response_identity(self):
        from audio_lab.task_context import waveform_profiles,response_identity_evidence,WAVEFORM_METHOD
        record=getattr(self,'comparison_record',None)
        stored=record.data.get('question_visual_signature',{}) if record else {}
        if not stored:
            stored=(getattr(self,'recovery_identity',None) or {}).get('waveforms',{})
        method=record.data.get('question_visual_signature_method') if record else WAVEFORM_METHOD
        if (not stored or method!=WAVEFORM_METHOD) and getattr(self,'audio_folder',None):
            # Older saved sessions have a complete original screenshot.
            original=cv2.imread(str(self.audio_folder/'page.png'))
            if original is not None:
                lines=getattr(self.detector,'last_task_lines',[])
                plain=getattr(self.detector,'last_input_plain_text',False)
                try:
                    targets,_,_=self.detector.detect(original)
                    if {'A','B'}<=targets.keys():
                        stored=waveform_profiles(original,targets,('A','B'))
                        self.comparison_record.update(question_visual_signature={k:v.tolist() for k,v in stored.items()},
                            question_visual_signature_method=WAVEFORM_METHOD)
                finally:
                    self.detector.last_task_lines=lines
                    self.detector.last_input_plain_text=plain
        if not stored:return 'unknown'
        return response_identity_evidence(self.last_image,self.targets,stored)

    def question_has_changed(self,expected,observed):
        return self.identity_relation(expected,observed)=='different'

    def identity_relation(self,expected,observed):
        from question_identity import relation
        before,after=dict(expected or ()),dict(observed or ())
        mismatch=any(before[key]!=after[key] for key in before.keys()&after.keys())
        evidence=self.response_identity() if mismatch else 'unknown'
        result=relation(before,after,evidence)
        if mismatch:self.log('question_identity_check',expected=before,observed=after,
                             response_evidence=evidence,result=result)
        return result

    def guard_rating_action(self):
        if getattr(self,'remote_submitted',False):
            from question_progress import SubmissionPending
            raise SubmissionPending('已记录 Submit，正在确认结果；暂停本题后续点击。')
        if (getattr(self,'rating_identity_dirty',False) or
                getattr(self,'rating_input_revision',None)!=self.mouse.interference_count):
            raise RuntimeError('人工操作后重新核对题目身份，保留已有评价，不重播首轮。')

    def ensure_rating_identity(self):
        from rating_reader import question_marker,marker_changed,marker_matches
        expected=self.comparison_record.data.get('verified_question_marker') or self.comparison_record.data.get('question_marker') or self.question_marker
        # A complete, unchanged audio view already has current player geometry.
        # Verify its header directly before paying for another full-page OCR.
        fast=False
        if getattr(self,'targets',None) and callable(getattr(self.detector,'read_question_marker',None)):
            image=self.snapshot()
            from vision import glyph_state
            visible=all(0<=target.button[0]<image.shape[1] and 0<=target.button[1]<image.shape[0]
                        and glyph_state(image,target.button) in ('idle','active','locked')
                        for target in self.targets.values())
            if visible:
                observed=self.detector.read_question_marker(image)
                fast=isinstance(observed,(tuple,list)) and self.identity_relation(expected,observed)=='same'
        for _ in range(5):
            if fast:break
            self.identify()
            observed=question_marker(getattr(self.detector,'last_task_lines',[]))
            if self.identity_relation(expected,observed)!='unknown':break
            self.mouse.scroll(self.last_image.shape,down=False,notches=2)
        if self.question_has_changed(expected,observed):
            self.remote_question_changed=True
            raise RuntimeError('当前题目已变化，旧模型结果不再用于网页选择。')
        if self.identity_relation(expected,observed)!='same':
            raise RuntimeError('题号尚未读清，保留评价并重新识别。')
        self.rating_input_revision=self.mouse.interference_count
        self.rating_identity_dirty=False
        self.question_marker=tuple(observed)
        self.comparison_record.update(verified_question_marker=observed)
        self.log('rating_question_verified',expected=expected,observed=observed,
                 verification='current_header_and_visible_controls' if fast else 'relocated_players')

    def verify_recovery_identity(self):
        from rating_reader import question_marker,marker_matches
        from workflow import LocationUnconfirmed
        from dataclasses import replace
        from vision import stable_target
        image=self.snapshot()
        self.check_page(image)
        targets,_,_=self.detector.detect(image)
        observed=dict(question_marker(getattr(self.detector,'last_task_lines',[])))
        expected=dict((getattr(self,'comparison_record',None).data.get('verified_question_marker',())
                       if getattr(self,'comparison_record',None) else ()) or
                      (getattr(self,'recovery_identity',None) or {}).get('marker',())
                      or getattr(self,'question_marker',()))
        if not expected or not observed:
            raise LocationUnconfirmed('正在重新读取题号，保留已生成评价。')
        if self.identity_relation(expected,observed)!='same':
            raise LocationUnconfirmed('已检测到不同题号，重新确认当前题目。')
        if any(name not in targets or not stable_target(replace(old,state=targets[name].state),targets[name])
               for name,old in self.targets.items()):
            raise LocationUnconfirmed('播放器位置正在重新定位，保留已生成评价。')
        self.log('recovery_identity_verified',marker=list(observed.items()),basis='question_marker_and_page_geometry')

    def refresh_observation_context(self):
        from assistant import client
        from human_mouse import HumanMouse
        self.mouse.geometry=client(self.hwnd)
        self.mouse.expected=HumanMouse.position()

    def stop_recovery_observers(self):
        observer=getattr(self,'rating_observer',None)
        if observer:
            observer.stopped.set()
            for thread in (observer.thread,observer.input_thread):
                if thread.ident is not None:thread.join(timeout=5)
            if observer.thread.is_alive() or observer.input_thread.is_alive():
                raise RuntimeError('旧答案观察尚未退出，等待后再恢复。')

    def recover_saved_decisions(self):
        """Repair text-dependent decisions without rerunning audio models."""
        from audio_lab.decision_engine import build_decisions
        from audio_lab.comparison_record import model_judgments
        model=self.comparison_record.data.get('model') or {}
        pipeline=model.get('pipeline',{})
        # Failed inference must still use its normal model retry path.
        if any(stage!='decisions' for stage in pipeline.get('stage_errors',{})):return False
        try:
            decisions=build_decisions(self.audio_folder)
            review_path=self.audio_folder/'pronunciation_review.json'
            if decisions['judgments']['pronunciation']['judgment']=='需复核' and not review_path.exists():
                self.status('使用已保存 A/B 音频进行一次独立转写复核，不重播整题。')
                review_path.write_text(json.dumps({'state':'running'}),encoding='utf-8')
                with (self.audio_folder/'pronunciation_review.log').open('w',encoding='utf-8') as output:
                    worker=subprocess.Popen([str(AUDIO/'.venv'/'Scripts'/'python.exe'),str(AUDIO/'pronunciation_review.py'),str(self.audio_folder)],
                        stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    deadline=time.monotonic()+150
                    while worker.poll() is None and time.monotonic()<deadline:self.mouse.wait_observation(.2)
                    if worker.poll() is None:worker.terminate();worker.wait(timeout=5)
                outcome=json.loads(review_path.read_text(encoding='utf-8'))
                if outcome.get('state')=='running':review_path.write_text(json.dumps({'state':'failed','reason':'worker_failed_or_timed_out'}),encoding='utf-8')
                decisions=build_decisions(self.audio_folder)
        except (OSError,ValueError,KeyError) as exc:
            self.log('saved_decisions_unavailable',reason=str(exc));return False
        pipeline={**pipeline,'state':'draft_ready','reason':'','stage_errors':{}}
        (self.audio_folder/'pipeline_status.json').write_text(json.dumps(pipeline,ensure_ascii=False,indent=2),encoding='utf-8')
        self.comparison_record.update(model={**model,'state':'draft_ready','reason':'',
            'judgments':model_judgments(self.audio_folder),'pipeline':pipeline})
        self.log('saved_decisions_recovered',audio_models_rerun=False)
        return True

    def resume_pending_review(self):
        observer=getattr(self,'rating_observer',None)
        if self.remote_question_changed or not observer or observer.stopped.is_set():
            self.pending_review=False;return True
        completion=self.comparison_record.data.get('rating_completion',{})
        if completion.get('all_required_observed') and completion.get('submit_enabled_observed'):
            self.pending_review=False;return True
        model=self.comparison_record.data.get('model') or {}
        if model.get('pipeline',{}).get('stage_errors'):
            self.pending_review=False;return True
        return False

    def recover_current_question(self):
        from workflow import LocationUnconfirmed
        from rating_reader import question_marker,marker_changed,marker_matches
        self.stop_recovery_observers()
        # Window/focus recovery uses fresh geometry. All controls are then
        # located again; no old screen coordinates survive the recovery.
        self.refresh_observation_context()
        if getattr(self,'submission_repair_pending',False) is True:
            from remote_ratings import same_layout,submission_missing_rows
            from rating_reader import refresh_selections
            image=self.snapshot()
            old=getattr(self,'submitted_screen',{}).get('rows',[])
            reader=getattr(self.detector,'read_question_marker',None)
            marker=reader(image) if reader else ()
            previous=dict(getattr(self,'question_marker',()))
            if isinstance(marker,(tuple,list)) and marker_changed(previous,marker):return False
            current=refresh_selections(image,old)
            if submission_missing_rows(old,current) and same_layout(self.submitted_image,image,old):
                self.submission_repair_pending=False
                self.inspect_ratings()
                return True
        thread=getattr(self,'analysis_thread',None)
        if thread and thread.is_alive():
            self.status('等待本题已有分析完成，不重复启动模型。')
            while thread.is_alive():
                self.mouse.wait_observation(.2)
            self.stop_recovery_observers()
        previous=dict(self.comparison_record.data.get('verified_question_marker') or self.comparison_record.data.get('question_marker') or getattr(self,'question_marker',()))
        for _ in range(8):
            image=self.snapshot()
            try:
                self.identify()
                marker=question_marker(getattr(self.detector,'last_task_lines',[]))
                self.log('recovery_question_marker',expected=previous,observed=dict(marker))
                if self.question_has_changed(previous,marker):
                    self.mouse.wait_observation(.5)
                    self.identify()
                    confirmation=question_marker(getattr(self.detector,'last_task_lines',[]))
                    if marker==confirmation and self.question_has_changed(previous,confirmation):
                        self.log('recovery_new_question',marker=marker)
                        progress=getattr(self,'question_progress',None)
                        if progress is not None:progress.page_changed(marker,submission_observed=False)
                        return False
                    raise RuntimeError('题目编号暂不稳定，等待重新识别。')
                if previous and self.identity_relation(previous,marker)!='same':
                    self.mouse.scroll(image.shape,down=False,notches=2)
                    continue
                self.verify_recovery_identity()
                break
            except LocationUnconfirmed:
                self.mouse.scroll(image.shape,down=False,notches=3)
        else:raise RuntimeError('尚未定位音频区，稍后继续识别。')
        model=self.comparison_record.data.get('model') or {}
        if model.get('state') not in ('draft_ready','partial_ready') or any(
                item.get('judgment')=='需复核' for item in model.get('judgments',{}).values()):
            self.status('本题评价生成不完整，使用已保存音频重新生成；保留网页已有选择。')
            from audio_lab.task_context import capture_context
            context_path=self.audio_folder/'task_context.json'
            try:context=json.loads(context_path.read_text(encoding='utf-8'))
            except (OSError,ValueError):context={}
            # Identity was verified above. Keep this question's previously
            # accepted transcript instead of degrading it with a later OCR
            # pass over compressed/settling remote pixels.
            if not context.get('supported_english_text'):
                context=capture_context(self.detector,self.last_image,self.identity,self.audio_folder)
            self.context_saved=context.get('supported_english_text',False)
            self.log('rating_context_recovered',supported=self.context_saved,
                     target_text=context.get('target_text',''))
            if not self.recover_saved_decisions():
                errors=model.get('pipeline',{}).get('stage_errors',{})
                if errors and set(errors)=={'decisions'}:
                    raise RuntimeError('目标正文或评价依据仍未读清，保留录音和模型结果，继续重新识别正文。')
                self.analysis_done.clear()
                self.analysis_thread=threading.Thread(target=self.finish_audio_and_analyze,daemon=True)
                self.analysis_thread.start()
                while not self.analysis_done.is_set():self.mouse.wait_observation(.2)
        self.remote_question_changed=False
        self.inspect_ratings()
        return True

    def recover_failed_capture(self):
        """Revalidate retained PCM after bounded retries, without new playback."""
        from audio_lab.segment_session import segment
        from playback import CaptureRepairPending
        from rating_reader import question_marker
        self.stop_recovery_observers()
        self.stop_audio()
        self.refresh_observation_context()
        expected=self.comparison_record.data.get('question_marker') or self.question_marker
        self.identify()
        observed=question_marker(getattr(self.detector,'last_task_lines',[]))
        # identify updates pre-capture metadata; restore the immutable origin.
        self.question_marker=expected
        self.comparison_record.update(question_marker=expected)
        relation=self.identity_relation(expected,observed)
        if relation=='different':
            self.mouse.wait_observation(.5)
            self.identify()
            confirmation=question_marker(getattr(self.detector,'last_task_lines',[]))
            self.question_marker=expected
            self.comparison_record.update(question_marker=expected)
            if confirmation==observed and self.identity_relation(expected,confirmation)=='different':return False
        if relation!='same':raise CaptureRepairPending('录音已保留，正在等待明确的本题身份；未重播。')
        try:segment(self.audio_folder)
        except (OSError,ValueError) as exc:
            raise CaptureRepairPending('录音仍无法可靠切分：'+str(exc)+' 保留原录音，观察题目变化；不再循环重录。') from exc
        self.log('capture_repair_validated',recording_preserved=True)
        self.on_first_round_complete()
        return True

    def mid_rating_review(self):
        from workflow import LocationUnconfirmed
        from rating_reader import question_marker
        self.status('前几项已核对，回到本题音频区短暂复听，再继续按顺序评价。')
        self.log('mid_rating_review_start')
        proof=getattr(self,'recovery_identity',None)
        if not proof:raise RuntimeError('缺少本题身份依据，未执行中途复听。')
        for _ in range(8):
            image=self.snapshot()
            self.mouse.scroll(image.shape,down=False,notches=self.rng.randint(3,5))
            try:
                self.identify()
                expected=dict(proof.get('marker',()))
                marker=dict(question_marker(getattr(self.detector,'last_task_lines',[])))
                if expected and not expected.keys()<=marker.keys():continue
                break
            except LocationUnconfirmed:continue
        else:raise RuntimeError('返回音频区后未确认播放器，停止本题。')
        self.verify_recovery_identity()
        original=(self.settings.short_count,self.settings.short_min,self.settings.short_max)
        self.review_in_progress=True
        try:
            self.settings.short_count=1
            self.settings.short_min=.6
            self.settings.short_max=1.4
            self._opening_previews()
        finally:
            self.settings.short_count,self.settings.short_min,self.settings.short_max=original
            self.review_in_progress=False
        self.log('mid_rating_review_end')

    def wait_for_next_question(self):
        from rating_reader import question_marker,refresh_selections,marker_changed
        from rating_reader import submit_enabled
        from remote_ratings import SubmissionIncomplete,same_layout,submission_missing_rows
        from vision import scene_difference
        self.status('Submit 已点击，正在观察自动换题。')
        previous=dict(getattr(self,'question_marker',()))
        began=time.monotonic();deadline=began+90
        candidate=None;candidate_marker=None;stable_since=None;reset_samples=0;navigating=False
        form_reset_confirmed=False
        verified_marker=None
        partial_missing=None;partial_since=None
        while time.monotonic()<deadline:
            self.mouse.wait_observation(.15)
            image=self.grab(self.hwnd)[0]
            reader=getattr(self.detector,'read_question_marker',None)
            marker=reader(image) if reader else None
            targets=None
            if not isinstance(marker,(tuple,list)):
                targets,_,_=self.detector.detect(image)
                marker=question_marker(getattr(self.detector,'last_task_lines',[]))
            changed=marker_changed(previous,marker,form_reset=form_reset_confirmed)
            if changed:
                if verified_marker!=marker:
                    if targets is None:targets,_,_=self.detector.detect(image)
                    if not {'A','B'}<=targets.keys():
                        if navigating:self.mouse.scroll(image.shape,down=False,notches=4)
                        continue
                    verified_marker=marker
                if candidate is not None and candidate_marker==marker and candidate.shape==image.shape and scene_difference(candidate,image)<2:
                    if time.monotonic()-stable_since>=.5:
                        self.log('next_question_ready',marker=marker)
                        observation=self.comparison_record.data.get('submission_observation',{})
                        self.comparison_record.update(submission_observation={**observation,'server_success_verified':True,
                            'result':'next_page_observed','next_question_confirmed':True,
                            'verification':'changed_question_marker_and_stable_players',
                            'note':'根据提交后题号变化和稳定播放器确认进入下一题；未直接读取服务器响应。'})
                        progress=getattr(self,'question_progress',None)
                        if progress is not None:progress.page_changed(marker,submission_observed=True)
                        self.save_completion_timing(refresh_complete=True)
                        return
                else:stable_since=time.monotonic()
                candidate=image.copy();candidate_marker=marker
                continue
            candidate=None;candidate_marker=None;stable_since=None;verified_marker=None
            old=getattr(self,'submitted_screen',{}).get('rows',[])
            if old and not navigating:
                current=refresh_selections(image,old)
                selected=[r['key'] for r in old if r.get('selection_state')=='selected']
                cleared=[r for r in current if r['key'] in selected and r.get('selection_state')=='none']
                reset_samples=reset_samples+1 if len(cleared)>=2 and len(cleared)==len(selected) else 0
                if reset_samples>=2:
                    navigating=True
                    form_reset_confirmed=True
                    self.log('refresh_form_reset_confirmed',rows=len(cleared))
                    self.mouse.scroll(image.shape,down=False,notches=8)
                    continue
                missing=submission_missing_rows(old,current)
                if (missing and not submit_enabled(image,self.submitted_screen.get('submit_boxes',[]))
                        and same_layout(getattr(self,'submitted_image',image),image,old)):
                    if partial_missing!=missing:partial_missing=missing;partial_since=time.monotonic()
                    elif time.monotonic()-partial_since>=1.0:raise SubmissionIncomplete(missing)
                else:partial_missing=None;partial_since=None
            incomplete_marker=not marker or 'prompt' in previous and 'prompt' not in dict(marker)
            if incomplete_marker and not navigating and partial_missing is None and time.monotonic()-began>=3:
                # A manual refresh/scroll can move every old row. Search the
                # header without mistaking a missing marker for an old page.
                navigating=True
                self.log('refresh_header_search',reason='question_marker_out_of_view')
                self.mouse.scroll(image.shape,down=False,notches=6)
                continue
            if navigating and incomplete_marker:
                self.mouse.scroll(image.shape,down=False,notches=4)
        raise RuntimeError('页面尚未确认换题，继续观察自动刷新，不重复提交。')

    def save_completion_timing(self,refresh_complete=False):
        from task_tracking import timestamp
        record=self.comparison_record
        pipeline=record.data.get('model') or {}
        stages=pipeline.get('pipeline',{}).get('stage_timings',{})
        timings={**record.data.get('timings',{}),
                 'model_evaluation_seconds':pipeline.get('elapsed_seconds'),
                 'transcription_seconds':stages.get('analyzing_asr',stages.get('asr')),
                 'translation_seconds':None,'translation_state':'not_run'}
        key='total_until_next_question_seconds' if refresh_complete else 'total_until_submit_seconds'
        if key not in timings:
            timings[key]=round(time.monotonic()-getattr(self,'task_started_monotonic',record.started),3)
        tracking={**record.data.get('task_tracking',{}),**getattr(self,'task_tracking',{})}
        tracking.setdefault('next_question_confirmed_at' if refresh_complete else 'submit_recorded_at',timestamp())
        record.update(timings=timings,task_tracking=tracking)
        summary={'session':self.audio_folder.name,'started_at':record.data['started_at'],
                 'timings':timings,'task_tracking':tracking,
                 'submission':record.data.get('submission_observation',{})}
        (self.audio_folder/'completion_timing.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        with (AUDIO/'sessions'/'question_timings.jsonl').open('a',encoding='utf-8') as output:
            output.write(json.dumps(summary,ensure_ascii=False)+'\n')
        self.append_task_timeline('next_question_confirmed' if refresh_complete else 'submit_clicked',tracking)

    def append_task_timeline(self,event,tracking=None):
        from task_tracking import timestamp
        entry={'event':event,'recorded_at':timestamp(),'session':self.audio_folder.name,
            'question_marker':getattr(self,'question_marker',()),
            'task_tracking':tracking or getattr(self,'task_tracking',{})}
        with (AUDIO/'sessions'/'task_timeline.jsonl').open('a',encoding='utf-8') as output:
            output.write(json.dumps(entry,ensure_ascii=False)+'\n')

    def __init__(self,*args,**kwargs):
        self.submission_guard=threading.RLock()
        self.question_progress=None
        self.submit_monitor=None
        self.audio_process=None
        self.audio_folder=None
        self.audio_log=None
        self.pipeline_process=None
        self.last_capture_at=None
        self.round_captured=False
        self.analysis_started_at=None
        self.chain_finished_at=None
        self.failure_reason=None
        self.analysis_done=threading.Event()
        self.comparison_record=None
        self.analysis_thread=None
        self.finished=threading.Event()
        self.rating_observer=None
        self.remote_submitted=False
        self.remote_question_changed=False
        self.remote_status='网页答案将在评价区自动识别并记录，理由默认留空。'
        super().__init__(*args,**kwargs)
        self.fast_navigation=True
        # Playback sampling stays read-only. Text-guidance mouse movements
        # would invoke the blocking manual-input guard and miss autoplay.
        self.observation_only_playback=True
        self.automatic_recovery=True
        self.settings.replay_min=1
        self.settings.replay_max=2

    def stop_replays(self):
        return getattr(self,'remote_submitted',False) or self.analysis_done.is_set() and not getattr(self,'review_in_progress',False)

    def replay_count(self):
        longest=max((reading.total or 0 for reading in getattr(self,'known_readings',{}).values()),default=0)
        # Allow a second comparison for longer material while analysis is
        # pending. Completion still interrupts this budget immediately.
        if longest>=15:return min(2,self.settings.replay_max)
        return min(2,super().replay_count())

    def inspect_ratings(self):
        from remote_ratings import RatingController
        RatingController(self).run()

    def gaze_point(self,image):
        import math
        zones=((.05,.15,.22,.82),(.34,.65,.15,.28),(.78,.94,.22,.76),(.35,.70,.60,.86))
        previous=getattr(self,'last_park_point',None)
        for _ in range(30):
            zone=self.rng.randrange(len(zones));x1,x2,y1,y2=zones[zone]
            point=(image.shape[1]*self.rng.uniform(x1,x2),image.shape[0]*self.rng.uniform(y1,y2))
            obstructs=any(t.button[0]-15<=point[0]<=t.button[0]+t.button[2]*8 and
                           t.button[1]-20<=point[1]<=t.button[1]+t.button[3]*2 for t in self.targets.values())
            obstructs=obstructs or any(t.label[0]-35<=point[0]<=t.label[0]+t.label[2]+35 and
                t.label[1]-35<=point[1]<=t.label[1]+t.label[3]+35 for t in self.targets.values())
            if not obstructs and (previous is None or math.dist(point,previous)>image.shape[1]*.15):break
        self.last_park_point=point
        return point,zone

    def parked(self,image,target=None):
        point,zone=self.gaze_point(image)
        if target is None:self.mouse.move(point,image.shape,slow=True)
        else:self.mouse.observation_glide(point,image.shape)
        self.next_gaze_at=time.monotonic()+self.rng.uniform(3,7)
        self.log('mouse_park',point=list(point),zone=zone,target=target.name if target else 'start')

    def playback_guidance(self,image,readings,now):
        # Input is a reference voice, often a different sentence. Only map
        # actual A/B speech to the target. No OCR or mouse wait in this poll.
        self.mouse.observation_glide(None,image.shape,now)
        active=next(((name,r) for name,r in readings.items() if name in ('A','B') and r.state=='active'),None)
        if active and getattr(self,'audio_folder',None) and hasattr(self,'audio_zero') and not self.round_captured:
            name,reading=active
            if reading.current is not None and reading.current>=2 and now>=getattr(self,'next_follow_request',0):
                audio_time=now-self.audio_zero
                start=max(0,audio_time-reading.current-.5,audio_time-10)
                request={'label':name,'start_seconds':start,'end_seconds':audio_time,
                         'progress_seconds':reading.current,'sequence':time.time_ns()}
                try:
                    temporary=self.audio_folder/'follow_request.tmp'
                    temporary.write_text(json.dumps(request),encoding='utf-8')
                    temporary.replace(self.audio_folder/'follow_request.json')
                except OSError:pass
                self.next_follow_request=now+3
            try:
                follow=json.loads((self.audio_folder/'live_follow.json').read_text(encoding='utf-8'))
                if (follow['label']==name and now-follow['decoded_at']<5 and
                        follow['sequence']!=getattr(self,'last_follow_sequence',None)):
                    self.last_follow_sequence=follow['sequence']
                    context=getattr(self,'transcript_context',{})
                    from text_follow import phrase_position
                    previous=getattr(self,'follow_word_index',0) if getattr(self,'follow_label',None)==name else 0
                    index=phrase_position(context.get('target_text',''),follow['text'],previous)
                    word=next((item for item in context.get('transcript_words',[]) if item['target_index']==index),None)
                    if word:
                        x,y,w,h=word['box'];point=(x+w*.6,y+h+3)
                        moved=self.mouse.observation_glide(point,image.shape,now)
                        self.follow_label=name;self.follow_word_index=index
                        self.next_gaze_at=now+self.rng.uniform(3,6)
                        self.log('transcript_follow' if moved else 'transcript_match',target=name,word=word['text'],word_index=index,point=list(point),
                                 source=follow['source'],recognition_lag_seconds=round(now-self.audio_zero-follow['audio_end_seconds'],2))
            except (OSError,ValueError,KeyError):pass
        if now>=getattr(self,'next_gaze_at',now+1) and not getattr(self.mouse,'observation_path',None):
            point,zone=self.gaze_point(image)
            if self.mouse.observation_glide(point,image.shape,now):
                self.log('mouse_gaze',point=list(point),zone=zone)
            self.next_gaze_at=now+self.rng.uniform(3,7)

    def capture_recovery_identity(self,image):
        super().capture_recovery_identity(image)
        proof=getattr(self,'recovery_identity',None)
        if proof and self.comparison_record:
            from audio_lab.task_context import WAVEFORM_METHOD
            self.comparison_record.update(question_visual_signature={
                key:value.tolist() for key,value in proof.get('waveforms',{}).items() if key in ('A','B')},
                question_visual_signature_method=WAVEFORM_METHOD)

    def click(self, name, fast=False):
        if getattr(self,'remote_submitted',False):
            from question_progress import SubmissionPending
            raise SubmissionPending('已记录 Submit，先确认结果；不再点击旧题音频。')
        result=super().click(name,fast=fast)
        if name==self.settings.player_order[0] and not self.round_captured and self.audio_folder:
            # Discard observations of a round already in progress at startup.
            # The WAV retains its original time origin; only the complete
            # restarted round is used as segmentation evidence.
            (self.audio_folder/'timeline.jsonl').write_text('',encoding='utf-8')
        return result

    def snapshot(self):
        from assistant import u
        if (u.GetAsyncKeyState(0x74)&0x8000 or
                u.GetAsyncKeyState(0x11)&0x8000 and u.GetAsyncKeyState(0x52)&0x8000):
            self.rating_identity_dirty=True
        self.mouse.check_observation()
        image,geometry=self.grab(self.hwnd)
        monitor=getattr(self,'submit_monitor',None)
        if monitor is not None:monitor.observe_frame(image,geometry)
        self.last_image=image
        self.last_capture_at=time.monotonic()
        return image

    def publish_rating_screen(self,screen,image):
        monitor=getattr(self,'submit_monitor',None)
        if monitor is None:return
        from assistant import client
        monitor.publish(screen,image,client(self.hwnd))

    def note_submission(self,actor,screen,image,evidence):
        """One shared boundary for manual and automatic Submit attempts."""
        with self.submission_guard:
            if self.remote_submitted or self.remote_question_changed:return False
            progress=getattr(self,'question_progress',None)
            if progress is not None and not progress.submitted(actor,evidence):return False
            self.submitted_screen={**screen};self.submitted_image=image.copy()
            # Mark pending before any further worker can dispatch an old action.
            self.remote_submitted=True
            self.capture_repair_pending=False;self.pending_review=False
            self.comparison_record.save_remote({},str(evidence),submit_clicked=True,confirm_auto=False)
            self.comparison_record.update(submission_observation={
                **self.comparison_record.data.get('submission_observation',{}),
                'actor':actor,'result':'pending','click_evidence':evidence,
                'note':'已记录提交尝试；等待页面结果，尚未确认处理成功。'})
            self.remote_status='已记录'+('你的' if actor=='user' else '助手的')+' Submit，正在确认结果和下一题。'
            self.status(self.remote_status)
            self.log('submission_pending',actor=actor,evidence=evidence)
            self.save_completion_timing()
            return True

    def prepare_audio_session(self):
        self.audio_folder=AUDIO/'sessions'/('integrated_'+self.directory.name)
        self.audio_folder.mkdir(parents=True)
        write_json(self.audio_folder/'session.json',{'player_order':list(self.settings.player_order),
            'task_type':'ab_only' if len(self.settings.player_order)==2 else 'reference_ab'})
        self.comparison_record=ComparisonRecord(self.audio_folder,AUDIO/'sessions'/'feedback_library.jsonl')
        from question_progress import QuestionProgress
        from submit_monitor import ManualSubmitMonitor
        self.question_progress=QuestionProgress(self.comparison_record)
        self.submit_monitor=ManualSubmitMonitor(self)
        self.submit_monitor.thread.start()
        self.comparison_record.update(task_tracking=getattr(self,'task_tracking',{}))
        self.append_task_timeline('attempt_started')
        write_json(self.audio_folder/'capture_status.json',{
            'state':'preparing','seconds':0,'note':'正在确认题型、播放器和计时，尚未启动录音。'})

    def on_players_ready(self):
        # A missing Prompt is not a valid original identity. Locate its header
        # before capture, rather than bind all later questions to a Run alone.
        for _ in range(5) if self.audio_folder else ():
            if dict(getattr(self,'question_marker',())).get('prompt'):break
            self.mouse.scroll(self.last_image.shape,down=False,notches=2)
            self.identify()
        else:
            if self.audio_folder:raise RuntimeError('正在读取完整 Prompt 题号，尚未开始录音。')
        if self.audio_folder:self.capture_recovery_identity(self.last_image)
        self.status('播放器已确认，正在准备录音；设备就绪后开始完整首轮播放。')
        self.start_audio()
        self.log('recording_ready',player_order=list(self.settings.player_order))

    def start_audio(self):
        for name in ('FOLLOW_STOP','follow_request.json','live_follow.json'):
            (self.audio_folder/name).unlink(missing_ok=True)
        self.audio_log=(self.audio_folder/'audio_error.log').open('w',encoding='utf-8')
        self.audio_process=subprocess.Popen([str(AUDIO/'.venv'/'Scripts'/'python.exe'),str(AUDIO/'live_capture.py'),
            '--folder',str(self.audio_folder)],stdout=subprocess.DEVNULL,stderr=self.audio_log,creationflags=subprocess.CREATE_NO_WINDOW)
        self.follow_log=(self.audio_folder/'live_follow.log').open('w',encoding='utf-8')
        self.follow_process=subprocess.Popen([str(AUDIO/'.venv'/'Scripts'/'python.exe'),str(AUDIO/'live_follow.py'),str(self.audio_folder)],
            stdout=self.follow_log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            if self.event.is_set(): raise RuntimeError('录音启动已取消。')
            try:
                data=json.loads((self.audio_folder/'capture_status.json').read_text(encoding='utf-8'))
                if data['state']=='failed': raise RuntimeError(data.get('error','录音失败'))
                if 'audio_zero_monotonic' in data:
                    self.audio_zero=data['audio_zero_monotonic']; return
            except (OSError,ValueError): pass
            if self.audio_process.poll() is not None: raise RuntimeError('录音进程退出，请检查audio_error.log。')
            time.sleep(.1)
        raise RuntimeError('录音设备启动未完成，未启动点击。')

    def preview(self,image,readings,message):
        super().preview(image,readings,message)
        if self.audio_folder and hasattr(self,'audio_zero') and not self.round_captured and readings:
            for name,r in readings.items():
                if r.state=='active' and r.current is None:
                    path=self.audio_folder/f'timer_unreadable_{name}.png'
                    if not path.exists():
                        cv2.imwrite(str(path),image)
                        self.log('timer_unreadable',target=name,attempts=getattr(self.detector,'timer_failures',{}).get(name,[]))
            active=[n for n,r in readings.items() if r.state=='active']
            label=active[0] if len(active)==1 else ('idle' if all(r.state=='idle' for r in readings.values()) else 'unknown')
            row={'audio_seconds':max(0,(self.last_capture_at or time.monotonic())-self.audio_zero),
                 'label':label,'readings':{n:asdict(r) for n,r in readings.items()}}
            with (self.audio_folder/'timeline.jsonl').open('a',encoding='utf-8') as out:
                out.write(json.dumps(row,ensure_ascii=False)+'\n')
            self.check_capture_signal(readings)

    def check_capture_signal(self,readings):
        """Warn from recorded PCM evidence; a quiet opening alone is not fatal."""
        active=[r for r in readings.values() if r.state=='active' and r.current is not None and r.current>=3]
        if not active or not self.audio_folder:return
        try:info=json.loads((self.audio_folder/'capture_status.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):return
        if info.get('nonzero_samples_total')==0:
            warning='画面正在播放，但录音尚未收到声音：'+info.get('device','当前输出设备')+'。请核对向日葵远程声音；开头静音会继续观察。'
            if not getattr(self,'capture_signal_warning',False):
                self.log('capture_signal_warning',device=info.get('device'),captured_seconds=info.get('seconds'))
            self.capture_signal_warning=True
            self.status(warning)
        elif info.get('nonzero_samples_total',0)>0:
            self.capture_signal_warning=False

    def identify(self):
        image,readings=super().identify()
        if self.audio_folder and not self.round_captured and not getattr(self,'capture_repair_pending',False):
            from rating_reader import question_marker
            self.question_marker=question_marker(getattr(self.detector,'last_task_lines',[]))
            metadata={'player_order':list(self.settings.player_order),
                'task_type':'ab_only' if len(self.settings.player_order)==2 else 'reference_ab',
                'mode_detection':'automatic','evidence':getattr(self.detector,'last_mode_evidence',[]),
                'question_marker':self.question_marker,'task_tracking':getattr(self,'task_tracking',{})}
            write_json(self.audio_folder/'session.json',metadata)
            self.comparison_record.update(task_type=metadata['task_type'],player_order=metadata['player_order'],
                mode_evidence=metadata['evidence'],question_marker=self.question_marker)
            if not getattr(self,'mode_saved',False):
                self.append_task_timeline('question_identified')
                self.mode_saved=True
            cv2.imwrite(str(self.audio_folder/'page.png'),image)
            if self.identity:
                x,y,w,h=self.identity
                cv2.imwrite(str(self.audio_folder/'target_text_crop.png'),image[y:y+h,x:x+w])
            if not getattr(self,'context_saved',False):
                from audio_lab.task_context import capture_context
                context=capture_context(self.detector,image,self.identity,self.audio_folder)
                self.transcript_context=context
                self.context_saved=context.get('supported_english_text',False)
                self.log('task_context',target_text=context.get('target_text',''),supported=context.get('supported_english_text',False))
        return image,readings

    def on_first_round_complete(self):
        if not self.round_captured:
            self.round_captured=True
            self.chain_finished_at=time.monotonic()
            # Closing the recorder and model startup must not delay replay clicks.
            self.analysis_thread=threading.Thread(target=self.finish_audio_and_analyze,daemon=True)
            self.analysis_thread.start()
            # Independent read-only observer catches answers during analysis;
            # its own OCR instance never shares mutable playback OCR state.
            from remote_ratings import RemoteRatingObserver
            self.rating_observer=RemoteRatingObserver(self,{'rows':[],'submit_boxes':[]},self.last_image,{})
            self.rating_observer.start()

    def validate_first_round(self):
        if not self.audio_folder:return
        from audio_lab.segment_session import segment,CaptureUnavailable
        began=time.monotonic()
        # Leave actual recorded tail silence before closing the WAV. Do not
        # start model workers on a recording that cannot even be segmented.
        self.mouse.wait_observation(.5)
        self.stop_audio()
        self.mouse.check_observation()
        try:
            segment(self.audio_folder)
        except CaptureUnavailable:
            raise
        except ValueError as exc:
            self.capture_validation_failure=str(exc)
            self.log('capture_validation_failed',reason=str(exc))
            raise PlaybackEvidenceError('录音完整性核对未通过：'+str(exc)) from exc
        self.capture_validation_failure=None
        self.log('capture_validation_passed',seconds=round(time.monotonic()-began,3))

    def on_first_round_retry(self,attempt,reason):
        self.stop_audio()
        self.mouse.check_observation()
        archive=self.audio_folder/'recovery_attempts'/f'failed_{attempt}'
        archive.mkdir(parents=True,exist_ok=False)
        # Preserve every failed capture with its own original time origin.
        # No old PCM/timeline is spliced into the new full round.
        for name in ('whole.wav','timeline.jsonl','capture_status.json','audio_error.log',
                     'STOP','clip_manifest.json','Input.wav','A.wav','B.wav'):
            path=self.audio_folder/name
            if path.exists():path.replace(archive/name)
        (archive/'failure.json').write_text(json.dumps({'attempt':attempt,'reason':reason},
            ensure_ascii=False,indent=2),encoding='utf-8')
        self.audio_process=None
        self.audio_log=None
        self.last_capture_at=None
        self.capture_signal_warning=False
        self.comparison_record.update(recovery={'attempt':attempt,'reason':reason,'failed_capture':str(archive)})
        self.mouse.check_observation()
        self.start_audio()
        self.log('recovery_recording_ready',attempt=attempt,archive=str(archive))

    def finish_audio_and_analyze(self):
        try:
            self.analysis_started_at=time.monotonic()
            self.log('analysis_started')
            self.stop_audio()
            log=(self.audio_folder/'pipeline.log').open('w',encoding='utf-8')
            try:
                self.pipeline_process=subprocess.Popen([str(AUDIO/'.venv'/'Scripts'/'python.exe'),str(AUDIO/'pipeline.py'),str(self.audio_folder)],
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                self.pipeline_process.wait()
            finally: log.close()
        except Exception as exc:
            (self.audio_folder/'pipeline_status.json').write_text(json.dumps({'state':'needs_review','reason':str(exc)}),encoding='utf-8')
        finally:
            try:
                try:
                    status=json.loads((self.audio_folder/'pipeline_status.json').read_text(encoding='utf-8'))
                except (OSError,ValueError):
                    status={'state':'needs_review','reason':'分析进程未生成有效状态；请检查 pipeline.log'}
                if status.get('state') not in ('draft_ready','partial_ready','needs_review'):
                    status={'state':'needs_review','reason':'分析进程退出，未生成完整结果；请检查 pipeline.log'}
                    (self.audio_folder/'pipeline_status.json').write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8')
                self.comparison_record.save_model(status,self.chain_finished_at,self.analysis_started_at)
                self.log('analysis_completed',state=status['state'],reason=status.get('reason',''))
                from audio_lab.question_history import find_history
                history=find_history(self.audio_folder)
                self.comparison_record.update(question_history=history)
                reused=history['reused_judgments']
                if reused:
                    model=self.comparison_record.data['model']
                    self.comparison_record.update(model={**model,'judgments':{
                        **model['judgments'],**{k:v for k,v in reused.items() if k in DIMENSIONS}}})
            finally:
                observer=getattr(self,'rating_observer',None)
                if observer:observer.stopped.set()
                self.analysis_done.set()

    def stop_audio(self):
        if self.audio_folder:
            (self.audio_folder/'FOLLOW_STOP').touch()
        follower=getattr(self,'follow_process',None)
        if follower and follower.poll() is None:
            try:follower.wait(timeout=2)
            except subprocess.TimeoutExpired:
                follower.terminate();follower.wait(timeout=5)
        follow_log=getattr(self,'follow_log',None)
        if follow_log and not follow_log.closed:follow_log.close()
        if self.audio_process and self.audio_process.poll() is None:
            (self.audio_folder/'STOP').touch()
            self.audio_process.wait(timeout=15)
        if self.audio_log and not self.audio_log.closed: self.audio_log.close()

    def run(self):
        try:
            self.prepare_audio_session()
            super().run()
        except Exception as exc:
            self.failure_reason=str(exc)
            raise
        finally:
            self.finished.set()
            if not self.round_captured:
                self.stop_audio()
                if self.audio_folder:
                    (self.audio_folder/'pipeline_status.json').write_text(json.dumps({
                        'state':'needs_review','reason':'首轮未完整完成，模型分析未启动：'+(self.failure_reason or '流程已取消'),
                        'analysis_seconds':0,'analysis_started':False},ensure_ascii=False,indent=2),encoding='utf-8')


class IntegratedApp(DynamicApp):
    defer_detector_loading=True
    def __init__(self):
        super().__init__()
        self.audio_workflow=None
        self.human_started_at=None
        self.model_seen=False
        self.human_fields={}
        self.notified_record=None
        self.replay_min.set('1');self.replay_max.set('2')
        self.short_enabled.set(True);self.scroll_after.set(True)
        from compact_ui import build_dashboard
        build_dashboard(self)
        from health_monitor import HealthMonitor, interrupted_capture
        self.health_monitor=HealthMonitor()
        self.last_health_alert=None
        self.startup_health_alert=interrupted_capture(AUDIO/'sessions')
        self.root.after(500,self.check_audio)
        self.root.after(200,self.update_dashboard)
        self.root.after(500,self.write_runtime_status)

    def write_runtime_status(self):
        """Expose our own task state without needing desktop UI inspection."""
        try:
            flow=self.audio_workflow
            future=getattr(self,'task_future',None)
            record=getattr(flow,'comparison_record',None)
            write_json(ROOT/'artifacts'/'assistant_runtime_status.json',{
                'pid':os.getpid(),'updated_at':datetime.now().astimezone().isoformat(),
                'running':future is not None and not future.done(),
                'status':self.status.get(),'worker_status':getattr(flow,'worker_status',None),
                'target_hwnd':getattr(flow,'hwnd',None),
                'session':str(getattr(flow,'directory','')),
                'recording':str(getattr(flow,'audio_folder','')),
                'progress':record.data.get('question_progress') if record else None,
                'submit_pending':getattr(flow,'remote_submitted',False) is True,
                'capture_pending':getattr(flow,'capture_repair_pending',False) is True,
                'review_pending':getattr(flow,'pending_review',False) is True})
        except OSError:
            pass
        finally:self.root.after(2000,self.write_runtime_status)

    def update_dashboard(self):
        future=getattr(self,'task_future',None)
        running=future is not None and not future.done()
        self.start_button.configure(state='disabled' if running else 'normal',
                                    text='自动答题中…' if running else '开始自动答题')
        flow=self.audio_workflow
        record=getattr(flow,'comparison_record',None)
        if record:
            progress=record.data.get('question_progress',{})
            sequence=getattr(flow,'task_tracking',{}).get('task_sequence',1)
            self.question_stage.set(f'第 {sequence} 题 · '+progress.get('label','识别当前题'))
            timings=record.data.get('timings',{})
            total=timings.get('total_until_submit_seconds')
            if total is None:total=max(0,time.monotonic()-getattr(flow,'task_started_monotonic',record.started))
            submitted=' · 已确认下一题' if progress.get('phase')=='page_changed' else ' · 等待提交结果'
            self.question_duration.set(f'{total:.1f} 秒'+(submitted if 'total_until_submit_seconds' in timings else ''))
            model=record.data.get('model')
            if model:
                self.model_duration.set(f'{model.get("elapsed_seconds",0):.1f} 秒')
            elif getattr(flow,'analysis_started_at',None) is not None:
                self.model_duration.set(f'{time.monotonic()-flow.analysis_started_at:.1f} 秒 · 生成中')
            elif getattr(flow,'capture_repair_pending',False):self.model_duration.set('录音待核验')
            else:self.model_duration.set('首轮录音中' if getattr(flow,'audio_zero',None) is not None else '准备播放')
        self.update_health_alert(running,flow)
        self.root.after(200,self.update_dashboard)

    def handle_health_event(self,payload):
        current=getattr(self.audio_workflow,'task_tracking',{}).get('task_id')
        if current and payload['data'].get('task_id')!=current:return
        self.health_monitor.observe(payload['event'],payload['data'],payload['at'])

    def update_health_alert(self,running,flow):
        from compact_ui import show_health_alert
        now=time.monotonic()
        task=getattr(flow,'task_tracking',{}).get('task_id')
        if task:self.health_monitor.begin(task,now)
        folder=getattr(flow,'audio_folder',None)
        note=' 本题资料保存在本机。' if folder else ''
        analysis_started=getattr(flow,'analysis_started_at',None)
        done=getattr(flow,'analysis_done',None)
        if done is None or done.is_set():analysis_started=None
        alert=self.health_monitor.snapshot(now,running,self.status.get(),
            capture_pending=getattr(flow,'capture_repair_pending',False) is True,
            review_pending=getattr(flow,'pending_review',False) is True,record_note=note,
            analysis_started=analysis_started)
        if running:self.startup_health_alert=None
        if not flow and not alert:alert=self.startup_health_alert
        show_health_alert(self,alert)
        # Persist changes in fault kind, not the per-second countdown.
        signature=(alert['level'],alert['detail']) if alert else None
        if signature!=self.last_health_alert:
            self.last_health_alert=signature
            try:
                (ROOT/'artifacts').mkdir(exist_ok=True)
                with (ROOT/'artifacts'/'health_alerts.jsonl').open('a',encoding='utf-8') as output:
                    output.write(json.dumps({'time':datetime.now().astimezone().isoformat(),
                        'task_id':task,'alert':alert},ensure_ascii=False)+'\n')
            except OSError:
                pass  # A failed diagnostic log must not stop UI updates.

    def refresh_task_labels(self):
        first='A' if self.task_type.get().startswith('仅') else 'Input'
        order='A/B' if first=='A' else 'Input/A/B'
        def visit(parent):
            for widget in parent.winfo_children():
                if isinstance(widget,ttk.LabelFrame) and str(widget.cget('text')).startswith('联合模式：'):
                    widget.configure(text=f'联合模式：首轮完整 {order}；最多补听两次＋有限短听；结果就绪后代选评价')
                if isinstance(widget,ttk.Label):
                    text=str(widget.cget('text'))
                    if text.startswith('首轮 B 结束后，重播'):
                        widget.configure(text=f'首轮 B 结束后，重播 {first} 的停顿（秒）：')
                    elif text.startswith('重听次数包含第一遍'):
                        widget.configure(text=f'补听第一遍 {first}；后续随机 A/B；按实际计时与图标确认。')
                visit(widget)
        visit(self.root)

    def start_human_timer(self):
        self.human_started_at=time.monotonic()
        self.human_status.set('计时中。请直接在远程网页选择或修改，助手自动记录，理由默认留空。')

    def show_transcripts(self):
        flow=self.audio_workflow
        if not flow or not flow.audio_folder:
            self.audio_status.set('请先启动本题联合流程，或载入已保存题目的结果。'); return
        from tkinter.scrolledtext import ScrolledText
        from audio_lab.transcript_report import transcript_report
        window=tk.Toplevel(self.root)
        window.title('本题转写与耗时 · '+flow.audio_folder.name)
        text=ScrolledText(window,width=100,height=24,wrap='word')
        text.pack(fill='both',expand=True,padx=8,pady=8)
        text.insert('1.0',transcript_report(flow.audio_folder))
        text.configure(state='disabled')

    def load_saved_result(self):
        current=self.audio_workflow
        if current and (not current.finished.is_set() or
                current.round_captured and not current.analysis_done.is_set()):
            self.audio_status.set('当前题目仍在运行，请等待结束再查看旧题。'); return
        folder=filedialog.askdirectory(title='选择 integrated_session 开头的题目目录',initialdir=str(AUDIO/'sessions'))
        if not folder: return
        observer=getattr(current,'rating_observer',None)
        if observer: observer.stopped.set()
        try:
            record=ComparisonRecord.load(folder,AUDIO/'sessions'/'feedback_library.jsonl')
            status=json.loads((Path(folder)/'pipeline_status.json').read_text(encoding='utf-8'))
            if status['state'] not in ('draft_ready','partial_ready','needs_review'):
                raise ValueError('此题仍在分析，请稍后再查看。')
            elapsed=status.get('elapsed_seconds',0)
            status.setdefault('completed_at',datetime.fromtimestamp((Path(folder)/'pipeline_status.json').stat().st_mtime).astimezone().isoformat())
            first_finished=record.started+record.data.get('timings',{}).get('first_round_seconds',0)
            record.save_model(status,first_finished,time.monotonic()-elapsed,recovered=True)
            done=threading.Event(); done.set()
            self.audio_workflow=SimpleNamespace(audio_folder=Path(folder),comparison_record=record,
                finished=done,analysis_done=done,round_captured=True,analysis_started_at=None)
            self.task_type.set('仅 A＋B（已保存题目）' if record.data['model']['judgments']['speaker']['judgment']=='不适用' else 'Input＋A＋B（已保存题目）')
            self.refresh_task_labels()
            self.model_seen=True
            self.human_started_at=time.monotonic()
            for key,(answer,reason) in self.human_fields.items():
                item=(record.data.get('human') or {}).get('judgments',{}).get(key,{})
                answer.set(item.get('judgment','')); reason.set(item.get('reason',''))
            self.human_status.set('已载入保存的题目，可查看模型结果或修改自己的判断。')
        except Exception as exc:
            self.audio_status.set('无法载入：'+str(exc))

    def save_human(self):
        flow=self.audio_workflow
        if not flow or not flow.comparison_record or self.human_started_at is None:
            self.human_status.set('请先启动本题联合流程。'); return
        try:
            flow.comparison_record.save_human({key:{'judgment':fields[0].get(),'reason':fields[1].get()}
                for key,fields in self.human_fields.items()},self.human_started_at,self.model_seen)
            self.human_status.set('已保存：以你的判断为准；每次修改均留历史。'+str(flow.audio_folder/'comparison.json'))
        except Exception as exc:
            self.human_status.set(str(exc))

    def resume_remote(self,folder=None):
        current=self.audio_workflow
        folder=Path(folder) if folder else getattr(current,'audio_folder',None)
        if folder is None:
            self.human_status.set('请先查看已保存题目的结果，再继续记录本题。');return
        if current and not current.finished.is_set():
            self.human_status.set('本题仍在运行，请等待结束。');return
        previous=getattr(current,'rating_observer',None)
        if previous: previous.stopped.set()
        try:
            record=ComparisonRecord.load(folder,AUDIO/'sessions'/'feedback_library.jsonl')
            metadata=json.loads((folder/'session.json').read_text(encoding='utf-8'))
            context=json.loads((folder/'task_context.json').read_text(encoding='utf-8'))
            from question_progress import retained_capture_available
            has_model=bool(record.data.get('model'))
            if not has_model and not retained_capture_available(folder):
                raise ValueError('此题尚无可恢复的模型记录或已保存录音。')
            hwnd=self.select_window();self.activate(hwnd)
        except Exception as exc:
            self.human_status.set('无法继续记录：'+str(exc));return
        def work(event,send):
            import re
            from audio_lab.task_context import capture_context
            from rating_reader import question_marker,marker_matches
            from continuous_runner import ContinuousRunner
            from assistant import u
            if self.detector is None:
                from vision import Detector
                self.detector=Detector()
            if not dict(metadata.get('question_marker',())).get('prompt'):
                original=cv2.imread(str(folder/'page.png'))
                if original is not None:
                    restored=self.detector.read_question_marker(original)
                    if dict(restored).get('prompt'):
                        metadata['question_marker']=restored
                        record.update(question_marker=restored)
            self.activate(hwnd)
            if event.wait(.6):return
            flow=RecordedWorkflow(hwnd,self.detector,event,send,grab,
                Settings(player_order=tuple(metadata['player_order']),auto_mode=False),ROOT)
            flow.round_captured=has_model
            if has_model:flow.analysis_done.set()
            flow.capture_repair_pending=not has_model
            flow.audio_folder=folder;flow.comparison_record=record
            from question_progress import QuestionProgress
            from submit_monitor import ManualSubmitMonitor
            flow.question_progress=QuestionProgress(record)
            flow.submit_monitor=ManualSubmitMonitor(flow)
            flow.submit_monitor.thread.start()
            flow.chain_finished_at=record.started+record.data.get('timings',{}).get('first_round_seconds',0)
            flow.mode_detected=True
            flow.question_marker=tuple(tuple(item) for item in record.data.get('verified_question_marker') or metadata.get('question_marker',[]))
            flow.recovery_identity={'marker':flow.question_marker}
            flow.task_tracking=record.data.get('task_tracking',{})
            flow.task_started_monotonic=record.started
            if flow.task_tracking.get('question_started_at'):
                began=datetime.fromisoformat(flow.task_tracking['question_started_at'])
                flow.task_started_monotonic=time.monotonic()-(datetime.now().astimezone()-began).total_seconds()
            observation=record.data.get('submission_observation',{})
            flow.remote_submitted=flow.question_progress.data['phase']=='submit_pending' or bool(
                observation.get('clicked_at') and observation.get('result')!='incomplete_form')
            self.audio_workflow=flow
            if not has_model:flow.log('retained_capture_resume',recording_preserved=True)
            def factory():
                next_flow=RecordedWorkflow(hwnd,self.detector,event,send,grab,
                    Settings(replay_min=1,replay_max=2,short_enabled=True,auto_mode=True),ROOT)
                self.audio_workflow=next_flow
                return next_flow
            def ready():
                try:grab(hwnd)
                except Exception:return False
                return True
            ContinuousRunner(event,send,factory,ready,lambda:bool(u.GetAsyncKeyState(0x1B)&0x8000)).run(
                initial_flow=flow)
        self.start_task(work)

    def check_audio(self):
        flow=self.audio_workflow
        if flow and getattr(flow,'mode_detected',False):
            label='仅 A＋B（自动识别，无参考音频）' if len(flow.settings.player_order)==2 else 'Input＋A＋B（自动识别，有参考音频）'
            if self.task_type.get()!=label:
                self.task_type.set(label)
                self.refresh_task_labels()
        if flow and flow.audio_folder:
            self.result_context.set('结果所属录音：'+flow.audio_folder.name+'；提交后等待自动刷新并继续下一题。')
            self.model_seen=True
            path=flow.audio_folder/'pipeline_status.json'
            # Show each model's evidence as soon as its file is available.
            try:
                comparison=json.loads((flow.audio_folder/'model_comparison.json').read_text(encoding='utf-8'))
                quality=comparison.get('quality',{})
                speaker=comparison.get('speaker_similarity',{})
                if all(n in quality for n in ('A','B')):
                    self.rating_table.item('quality',values=('音质',f'SigMOS A={quality["A"]["MOS_OVRL"]:.2f} / B={quality["B"]["MOS_OVRL"]:.2f}；需核对明显缺陷，不直接按分差选项'))
                if all(n in speaker for n in ('A','B')):
                    self.rating_table.item('speaker',values=('音色相似度',f'相似度 A={speaker["A"]:.3f} / B={speaker["B"]:.3f}；待核对'))
                if all(n in comparison.get('transcripts',{}) for n in ('A','B')):
                    self.rating_table.item('pronunciation',values=('发音忠实度','A/B转写已生成；目标正文待核对，不自动认定 Both Good'))
            except (OSError,ValueError,KeyError): pass
            try:
                scores=json.loads((flow.audio_folder/'naturalness.json').read_text(encoding='utf-8'))['scores']
                available=' / '.join(f'{n}={scores[n]["mos"]:.2f}' for n in ('A','B') if n in scores)
                if available: self.rating_table.item('naturalness',values=('自然度','UTMOSv2 '+available+'；待听感验证'))
            except (OSError,ValueError,KeyError): pass
            try:
                data=json.loads(path.read_text(encoding='utf-8'))
                labels={'segmenting':'正在切分本题录音','analyzing_quality_speaker_asr':'正在分析内容、音质、音色',
                        'analyzing_asr':'正在本地转写（PyTorch · 英语）','analyzing_quality':'正在分析音质','analyzing_speaker':'正在分析音色',
                        'analyzing_naturalness':'正在分析自然度','analyzing_parallel':'正在并行分析内容、自然度与音质',
                        'draft_ready':'模型结果已生成，等待核对',
                        'partial_ready':'部分结果已生成，失败维度需复核','needs_review':'需要复核'}
                self.audio_status.set(labels.get(data['state'],data['state'])+'；'+data.get('reason',str(flow.audio_folder)))
                if data['state'] not in ('needs_review','draft_ready','partial_ready'):
                    elapsed=0 if flow.analysis_started_at is None else time.monotonic()-flow.analysis_started_at
                    self.audio_status.set(labels.get(data['state'],data['state'])+f'；分析已用 {elapsed:.1f} 秒；最多补听两次。')
                if data['state']=='needs_review':
                    for key in self.rating_table.get_children():
                        title=self.rating_table.item(key,'values')[0]
                        self.rating_table.item(key,values=(title,'未生成：'+data.get('reason','需要复核录音或模型日志')))
                if data['state']=='draft_ready':
                    comparison=json.loads((flow.audio_folder/'model_comparison.json').read_text(encoding='utf-8'))
                    natural=json.loads((flow.audio_folder/'naturalness.json').read_text(encoding='utf-8'))
                    speaker=comparison['speaker_similarity']
                    scores=natural['scores']
                    quality=comparison['quality']
                    values={
                        'quality':f'SigMOS A={quality["A"]["MOS_OVRL"]:.2f} / B={quality["B"]["MOS_OVRL"]:.2f}；需核对明显缺陷，分差不直接决定选项',
                        'speaker':f'相似度 A={speaker["A"]:.3f} / B={speaker["B"]:.3f}；候选倾向 '+('A' if speaker['A']>speaker['B'] else 'B' if speaker['B']>speaker['A'] else '无差异')+'，未校准',
                        'pronunciation':'转写已保存；目标正文截图待核对，暂不判断 Both Good 或 A/B',
                        'naturalness':f'UTMOSv2 A={scores["A"]["mos"]:.2f} / B={scores["B"]["mos"]:.2f}；候选倾向 '+('A' if scores['A']['mos']>scores['B']['mos'] else 'B' if scores['B']['mos']>scores['A']['mos'] else '无差异')+'，待听感验证',
                        'overall':'待核对各维度与严重缺陷；尚未自动决定整体偏好或主要因素'}
                    for key,value in values.items():
                        title=self.rating_table.item(key,'values')[0]
                        self.rating_table.item(key,values=(title,value))
                    self.audio_status.set(f'本题模型证据已生成；分析共 {data.get("elapsed_seconds",0):.1f} 秒；请核对下表。目录：'+str(flow.audio_folder))
            except (OSError,ValueError,KeyError):
                self.audio_status.set('联合录音目录：'+str(flow.audio_folder))
            if not path.exists():
                try:
                    capture=json.loads((flow.audio_folder/'capture_status.json').read_text(encoding='utf-8'))
                    state=capture['state']
                    text=('正在定位题目和播放器，尚未开始录音或点击播放。' if state=='preparing' else
                          f'首轮录音：{capture.get("seconds",0):.1f} 秒；'+('正在录音，首轮完成后自动分析' if state=='recording' else state))
                    if getattr(flow,'capture_signal_warning',False):text+='；警告：画面播放中，录音尚未收到声音。'
                    self.audio_status.set(text)
                except (OSError,ValueError): pass
            if flow.analysis_started_at is not None and not flow.analysis_done.is_set() and not path.exists():
                self.audio_status.set(f'首轮已结束，正在保存录音并启动分析；已用 {time.monotonic()-flow.analysis_started_at:.1f} 秒。')
            record=flow.comparison_record
            if record and record.data['model']:
                model=record.data['model']
                for key,item in model['judgments'].items():
                    title=self.rating_table.item(key,'values')[0]
                    human=record.data['human']
                    comparison=''
                    if human and human['judgments'][key]['judgment']!='需复核':
                        choice=human['judgments'][key]['judgment']
                        comparison=f'；你选 {choice}（'+('一致' if choice==item['judgment'] else '以你为准')+'）'
                    elif record.data.get('remote_ratings',{}).get(key,{}).get('source')=='model_auto':
                        choice=record.data['remote_ratings'][key]['judgment']
                        comparison=f'；网页已代选 {choice}，待你复核'
                    self.rating_table.item(key,values=(title,f'模型：{item["judgment"]}{comparison}；{item["reason"]}'))
                outcome={'draft_ready':'模型结果已出','partial_ready':'部分模型结果已出；失败项需复核'}.get(model['state'],'分析结束，需要复核')
                timing=record.data.get('timings',{})
                elapsed_parts=[outcome,f'分析共 {model["elapsed_seconds"]:.1f} 秒']
                if 'model_ready_since_session_seconds' in timing:
                    elapsed_parts.append(f'本题开始后第 {timing["model_ready_since_session_seconds"]:.1f} 秒完成')
                if 'first_round_to_model_seconds' in timing:
                    elapsed_parts.append(f'从首轮结束到结果 {timing["first_round_to_model_seconds"]:.1f} 秒')
                self.audio_status.set('；'.join(elapsed_parts)+'。后续补听已取消。')
                if self.notified_record is not record:
                    self.notified_record=record
            if getattr(flow,'remote_status',None): self.human_status.set(flow.remote_status)
            if record:
                timing=record.data.get('timings',{})
                elapsed=timing.get('total_until_submit_seconds')
                if elapsed is not None:
                    model_seconds=(record.data.get('model') or {}).get('elapsed_seconds')
                    self.result_context.set(f'本题提交总耗时 {elapsed:.1f} 秒；模型生成评价耗时 '+
                        (f'{model_seconds:.1f} 秒' if isinstance(model_seconds,(int,float)) else '尚无有效记录'))
        self.root.after(500,self.check_audio)

    def run_integrated(self):
        future=getattr(self,'task_future',None)
        if future is not None and not future.done():return
        if self.audio_workflow and (not self.audio_workflow.finished.is_set() or
                (self.audio_workflow.round_captured and not self.audio_workflow.analysis_done.is_set())):
            self.audio_status.set('上一题仍在运行或分析，请等待完成再开启联合流程。'); return
        from question_progress import resumable_session
        checkpoint=resumable_session(AUDIO/'sessions')
        if checkpoint is not None:
            self.status.set('正在载入上次题目进度，先核对当前页面与提交结果。')
            self.resume_remote(checkpoint)
            return
        if self.audio_workflow and self.audio_workflow.comparison_record and self.audio_workflow.comparison_record.data['human'] is None:
            self.audio_workflow.comparison_record.update(review_state='not_confirmed')
        previous_observer=getattr(self.audio_workflow,'rating_observer',None)
        if previous_observer: previous_observer.stopped.set()
        try:
            settings=Settings(1,2,float(self.pause_min.get()),
                float(self.pause_max.get()),int(self.scroll_max.get()),float(self.after_b_pause.get()),
                short_enabled=self.short_enabled.get(),short_count=int(self.short_count.get()),
                short_min=float(self.short_min.get()),short_max=float(self.short_max.get()),scroll_after=self.scroll_after.get(),
                auto_mode=True)
            settings.validate(); hwnd=self.select_window(); self.activate(hwnd)
        except Exception as exc:
            self.status.set(str(exc)); return
        for key in self.rating_table.get_children():
            title=self.rating_table.item(key,'values')[0]
            self.rating_table.item(key,values=(title,'等待本题录音与分析'))
        self.audio_status.set('正在确认题型和播放器；确认后准备录音并开始播放。')
        self.result_context.set('正在创建本题新录音；上题记录已保留，旧结果已清空。')
        self.task_type.set('自动识别：正在核对页面')
        self.refresh_task_labels()
        self.audio_workflow=None
        self.model_seen=False
        self.start_human_timer()
        for answer,reason in self.human_fields.values():
            answer.set(''); reason.set('')
        def work(event,send):
            if event.wait(.6): return
            # OCR loading may take several seconds after the UI-side activation.
            # Reacquire only at this explicit start, never during user takeover.
            self.activate(hwnd)
            if event.wait(.2): return
            if not foreground(hwnd):
                send('status','等待返回所选远程窗口；恢复后自动开始。')
            from dataclasses import replace
            from continuous_runner import ContinuousRunner
            from assistant import u
            def factory():
                if self.detector is None:
                    send('status','加载识别模型；加载失败会自动重试。')
                    from vision import Detector
                    self.detector=Detector()
                flow=RecordedWorkflow(hwnd,self.detector,event,send,grab,replace(settings),ROOT)
                self.audio_workflow=flow
                return flow
            def remote_ready():
                try:grab(hwnd)
                except Exception:return False
                return True
            ContinuousRunner(event,send,factory,remote_ready,
                lambda:bool(u.GetAsyncKeyState(0x1B)&0x8000)).run()
        self.start_task(work)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    app=IntegratedApp()
    if args.resume:app.root.after(500,lambda:app.resume_remote(args.resume))
    app.root.mainloop()
