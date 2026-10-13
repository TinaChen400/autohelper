"""One question identity across capture retries and page re-recognition."""
from datetime import datetime,timezone
import time
from uuid import uuid4


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class TaskTracking:
    def __init__(self,sequence=1):
        self.task_id=uuid4().hex
        self.sequence=sequence
        self.started_at=timestamp()
        self.started_monotonic=time.monotonic()
        self.attempt=0
        self.previous_session=None

    def attach(self,flow):
        self.attempt+=1
        flow.task_tracking={'task_id':self.task_id,'task_sequence':self.sequence,
            'question_started_at':self.started_at,'attempt_started_at':timestamp(),
            'attempt_number':self.attempt,'retry_of_session':self.previous_session,
            'boundary_basis':'confirmed_new_page_or_explicit_start'}
        flow.task_started_monotonic=self.started_monotonic
        directory=getattr(flow,'directory',None)
        self.previous_session=directory.name if directory is not None else None


def record_relation(current,prior):
    """Explicit identity takes priority; time alone never proves audio equality."""
    def tracking(record):return record.get('task_tracking',{})
    a,b=tracking(current),tracking(prior)
    if a.get('task_id') and b.get('task_id'):
        return 'same_question_retry' if a['task_id']==b['task_id'] else 'separate_question_occurrence'
    def marker(record):
        return dict(record.get('question_marker',[]) or [])
    ma,mb=marker(current),marker(prior)
    shared=ma.keys()&mb.keys()
    if any(ma[k]!=mb[k] for k in shared):return 'different_question_marker'
    try:
        ta=datetime.fromisoformat(a.get('attempt_started_at') or current['started_at'])
        tb=datetime.fromisoformat(b.get('attempt_started_at') or prior['started_at'])
        near=abs((ta-tb).total_seconds())<=15*60
    except (KeyError,ValueError,TypeError):near=False
    if near and ma.get('prompt') and ma.get('prompt')==mb.get('prompt') and ma.get('run')==mb.get('run'):
        return 'likely_same_question_retry'
    return 'unknown'
