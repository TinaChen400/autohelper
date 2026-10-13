"""Durable per-question phase and submission boundary shared by all workers."""
from datetime import datetime
import threading
import json
from pathlib import Path

PHASES={'identifying':'识别当前题','recording':'首轮录音','analyzing':'生成评价',
        'rating':'核对评价选项','reviewing':'提交前复核','submit_pending':'等待提交结果',
        'page_changed':'已确认下一题'}
ORDER={key:index for index,key in enumerate(PHASES)}
EVENT_PHASE={'recording_ready':'recording','capture_validation_passed':'analyzing',
             'analysis_started':'analyzing','analysis_completed':'rating',
             'rating_question_verified':'rating','model_rating_click':'rating',
             'submit_review_wait':'reviewing','next_question_ready':'page_changed',
             'recovery_new_question':'page_changed'}

class SubmissionPending(RuntimeError):
    """A click may have reached the remote page; resolve it before more input."""


def retained_capture_available(folder):
    """A saved signal may be revalidated; this is not proof of valid clips."""
    folder=Path(folder)
    try:
        capture=json.loads((folder/'capture_status.json').read_text(encoding='utf-8'))
        return (capture.get('state')=='saved' and capture.get('nonzero_samples_total',0)>0
                and (folder/'whole.wav').stat().st_size>44
                and (folder/'timeline.jsonl').stat().st_size>0)
    except (OSError,ValueError,TypeError):return False


def resumable_session(sessions):
    folders=sorted(Path(sessions).glob('integrated_session_*'),reverse=True)
    if not folders:return None
    folder=folders[0]
    try:data=json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):return None
    phase=data.get('question_progress',{}).get('phase')
    model_pending=phase in ('submit_pending','rating','reviewing') and data.get('model')
    capture_pending=phase in ('identifying','recording') and not data.get('model') and retained_capture_available(folder)
    if ((model_pending or capture_pending) and (folder/'session.json').exists()
            and (folder/'task_context.json').exists()):return folder
    return None


class QuestionProgress:
    def __init__(self,record):
        self.record=record
        self.lock=threading.RLock()
        self.data=dict(record.data.get('question_progress') or {})
        observed=record.data.get('submission_observation',{})
        inferred=('page_changed' if observed.get('next_question_confirmed') or observed.get('server_success_verified') else
                  'submit_pending' if observed.get('clicked_at') and observed.get('result')!='incomplete_form' else
                  'rating' if record.data.get('model') else 'identifying')
        self.data.setdefault('phase',inferred)
        self.data.setdefault('label',PHASES[self.data['phase']])
        self.data.setdefault('submit_attempt',0)
        self.data.setdefault('history',[])
        if not record.data.get('question_progress'):
            record.update(question_progress=self.data)

    def save(self,phase,**details):
        with self.lock:
            if self.data['phase']==phase and all(self.data.get(k)==v for k,v in details.items()):return
            stamp=datetime.now().astimezone().isoformat()
            self.data={**self.data,'phase':phase,'label':PHASES[phase],'updated_at':stamp,**details}
            self.data['history']=[*self.data['history'],{'at':stamp,'phase':phase,**details}][-80:]
            self.record.update(question_progress=self.data)

    def observe(self,event,**details):
        phase=EVENT_PHASE.get(event)
        if phase is None:return
        with self.lock:
            # Late background completion must never undo a submit/page boundary.
            if ORDER[phase]<ORDER[self.data['phase']]:return
            self.save(phase)

    def submitted(self,actor,evidence):
        with self.lock:
            if self.data['phase'] in ('submit_pending','page_changed'):return False
            self.save('submit_pending',actor=actor,submit_attempt=self.data['submit_attempt']+1,
                      click_evidence=evidence,result='unconfirmed')
            return True

    def incomplete(self,missing):
        with self.lock:
            self.save('rating',result='incomplete_form',missing=list(missing))

    def page_changed(self,marker,submission_observed):
        with self.lock:
            self.save('page_changed',next_marker=list(marker),
                      result='next_page_observed' if submission_observed else 'page_changed_without_observed_submit')
