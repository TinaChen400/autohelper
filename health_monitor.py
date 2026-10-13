"""UI-only fault signals. No input, focus changes, sound, or modal dialogs."""
from collections import Counter
import json
from pathlib import Path
import wave


def interrupted_capture(sessions):
    """Inspect latest capture on startup; never modify or resume damaged data."""
    folders=sorted(Path(sessions).glob('integrated_session_*'),reverse=True)
    if not folders:return None
    folder=folders[0]
    if not (folder/'whole.wav').exists():return None
    try:
        status=json.loads((folder/'capture_status.json').read_text(encoding='utf-8'))
        complete=status.get('state')=='saved'
    except (OSError,ValueError):complete=False
    try:
        with wave.open(str(folder/'whole.wav'),'rb') as audio:
            seconds=audio.getnframes()/audio.getframerate()
        usable=seconds>0
    except (OSError,EOFError,wave.Error,ZeroDivisionError):usable=False
    if complete and usable:return None
    return {'level':'error','title':'上次录音未正常完成',
            'detail':('找到部分录音，完整性仍需核验。' if usable else '上次录音为空或损坏，无法用它生成可靠评价。')+' 记录：'+folder.name,
            'action':'连接远程后点击开始，重新识别当前题目。原始记录保留供排查。'}

EVENTS={'recording_ready','playback_sample','capture_validation_passed','analysis_started','analysis_completed','submission_pending',
        'question_identity_check','rating_question_verified','rating_page',
        'saved_decisions_recovered','model_rating_click','auto_submit_clicked','next_question_ready',
        'continuous_recovery','first_round_retry','review_pending','capture_repair_pending','capture_signal_warning'}

class HealthMonitor:
    def __init__(self):
        self.task=None

    def begin(self,task,now):
        if task==self.task:return
        self.task=task;self.progress_at=now;self.phase='识别题目';self.signature=None
        self.model_ready=False;self.rating_signature=None
        self.awaiting_submit_outcome=False
        self.failures=Counter();self.reason='';self.fault_at=None;self.wait_key=None;self.wait_at=now

    def observe(self,event,data,now):
        task=data.get('task_id')
        if not task:return
        self.begin(task,now)
        if event in ('submission_pending','auto_submit_clicked'):
            self.awaiting_submit_outcome=True
        elif event=='next_question_ready':self.awaiting_submit_outcome=False
        elif self.awaiting_submit_outcome and event not in ('continuous_recovery',):
            return  # Late model/playback workers cannot undo the submit boundary.
        phases={'recording_ready':'首轮播放','capture_validation_passed':'录音已核验',
                'analysis_started':'生成评价','analysis_completed':'评价已生成，等待核对',
                'rating_question_verified':'核对选项','saved_decisions_recovered':'评价已恢复',
                'model_rating_click':'核对选项','submission_pending':'等待提交结果',
                'auto_submit_clicked':'等待换题','next_question_ready':'换题完成'}
        if event=='playback_sample':
            signature=json.dumps(data.get('readings',{}),sort_keys=True)
            if signature!=self.signature:
                self.signature=signature;self.progress_at=now
                if not self.model_ready:self.phase='播放观察'
        elif event=='question_identity_check' and data.get('result')=='unknown':
            self.phase='核对题号'
            self.reason='题号识别结果不一致，正在核对是否仍为同一题；不会使用未确认的题目结果。'
        elif event=='rating_page':
            # Re-reading the same selections is not new progress.
            selections={row['key']:row.get('selected') for row in data.get('dimensions',[])
                        if row.get('key') and row.get('selection_state')=='selected'}
            signature=json.dumps(selections,sort_keys=True)
            if selections and signature!=self.rating_signature:
                self.rating_signature=signature;self.progress_at=now;self.phase='核对选项'
        elif event in phases:
            self.phase=phases[event];self.progress_at=now
            if event in ('analysis_completed','saved_decisions_recovered'):
                self.model_ready=data.get('state','draft_ready') in ('draft_ready','partial_ready')
            if event in ('capture_validation_passed','saved_decisions_recovered','submission_pending','auto_submit_clicked','next_question_ready'):
                self.failures.clear();self.reason='';self.fault_at=None
        elif event in ('continuous_recovery','first_round_retry','review_pending','capture_repair_pending','capture_signal_warning'):
            reason=data.get('reason') or ('播放中尚未收到声音' if event=='capture_signal_warning' else event)
            # Inner failure and its immediate propagated outer exception are
            # one occurrence, not two independent retries.
            if reason!=self.reason or self.fault_at is None or now-self.fault_at>2:
                self.failures[reason]+=1
            self.reason=reason;self.fault_at=now

    def snapshot(self,now,running,status='',capture_pending=False,review_pending=False,record_note='',analysis_started=None):
        if self.task is None:return None
        if not running:
            if self.awaiting_submit_outcome:
                return {'level':'warning','title':'已停止 · 提交结果尚未确认',
                        'detail':'已经记录 Submit，尚未确认下一题。','action':'再次开始会先核对当前页面。'+record_note}
            if capture_pending or review_pending:
                return {'level':'error','title':'已停止，问题尚未解决','detail':self.reason or status,'action':record_note}
            return None
        if capture_pending or review_pending:
            return {'level':'error','title':'需要关注 · 录音核验未通过' if capture_pending else '需要关注 · 评价未完成',
                    'detail':self.reason or status,'action':record_note+' 请查看当前进度中的具体原因；不会把缺项当作完成。'}
        waiting='focus' if '焦点' in status else 'covered' if any(w in status for w in ('遮挡','最小化','远程画面恢复')) else None
        if waiting!=self.wait_key:self.wait_key=waiting;self.wait_at=now
        if waiting and now-self.wait_at>=15:
            seconds=int(now-self.wait_at)
            return {'level':'warning','title':f'等待远程窗口 · {seconds} 秒',
                    'detail':'鼠标操作正在等待焦点。' if waiting=='focus' else '远程画面不可用或被遮挡。',
                    'action':'返回并露出远程窗口后会继续核对。'+record_note}
        count=self.failures.get(self.reason,0)
        if count>=2:
            return {'level':'error' if count>=3 else 'warning','title':f'同一故障重复 {count} 次',
                    'detail':self.reason,'action':'正在按当前步骤恢复；若画面不动，请检查远程连接。'+record_note}
        idle=now-self.progress_at
        if self.awaiting_submit_outcome:analysis_started=None
        if analysis_started is not None:
            idle=now-analysis_started
        threshold=180 if analysis_started is not None or self.phase=='生成评价' else 30
        if idle>=threshold:
            return {'level':'error' if idle>=max(120,threshold) else 'warning',
                    'title':f'{"生成评价" if analysis_started is not None else self.phase} · {int(idle)} 秒未见进展',
                    'detail':('录音和模型评价已经保存。' if self.model_ready else '')+
                        (self.reason if self.phase=='核对题号' else '当前步骤未取得新进展，正在等待核对或恢复。'),
                    'action':'请查看远程画面是否更新；不要反复点开始。'+record_note}
        return None
