"""Keep the integrated task alive across recoverable per-attempt failures."""
from task_tracking import TaskTracking
from remote_ratings import SubmissionIncomplete,ReviewPending
from playback import PlaybackEvidenceError,CaptureRepairPending


def retry_delay(attempt):
    return min(8,2**min(attempt,3))


class ContinuousRunner:
    def __init__(self,event,send,factory,remote_ready,cancel_requested):
        self.event=event;self.send=send;self.factory=factory
        self.remote_ready=remote_ready;self.cancel_requested=cancel_requested

    def cancelled(self):
        if self.cancel_requested():self.event.set()
        return self.event.is_set()

    def wait(self,seconds):
        # Poll Esc even when focus/context checks are currently unavailable.
        for _ in range(max(1,round(seconds*5))):
            if self.cancelled() or self.event.wait(.2):return False
        return True

    def retire(self,flow):
        monitor=getattr(flow,'submit_monitor',None)
        if monitor is not None:monitor.stopped.set()
        flow.stop_recovery_observers()
        flow.finished.set()

    def run(self,initial_flow=None):
        flow=initial_flow;attempt=0;started=initial_flow is not None
        tracking=TaskTracking()
        if initial_flow is not None and getattr(initial_flow,'task_tracking',None):
            saved=initial_flow.task_tracking
            tracking.task_id=saved['task_id'];tracking.sequence=saved.get('task_sequence',1)
            tracking.started_at=saved.get('question_started_at',tracking.started_at)
            tracking.started_monotonic=getattr(initial_flow,'task_started_monotonic',tracking.started_monotonic)
            tracking.attempt=saved.get('attempt_number',1)
            tracking.previous_session=initial_flow.directory.name
        while not self.cancelled():
            if not self.remote_ready():
                self.send('status','等待远程画面恢复或返回所选窗口；任务保持运行，暂不操作。')
                if not self.wait(2):break
                continue
            try:
                if flow is None:
                    flow=self.factory();started=False
                    tracking.attach(flow)
                flow.finished.clear()
                if flow.remote_submitted is True:
                    # Submission is a one-way boundary. Never return to filling
                    # or click Submit again while its outcome is uncertain.
                    if attempt:flow.refresh_observation_context()
                    if not getattr(flow,'submission_observers_stopped',False):
                        flow.stop_recovery_observers()
                        stopper=getattr(flow,'stop_audio',None)
                        if callable(stopper):stopper()
                        flow.submission_observers_stopped=True
                    flow.wait_for_next_question()
                    self.retire(flow)
                    flow=None;attempt=0
                    tracking=TaskTracking(tracking.sequence+1)
                    continue
                if getattr(flow,'pending_review',False) is True:
                    if not flow.resume_pending_review():
                        if not self.wait(1):break
                        continue
                if getattr(flow,'capture_repair_pending',False) is True:
                    if not flow.recover_failed_capture():
                        self.retire(flow);flow=None;attempt=0
                        tracking=TaskTracking(tracking.sequence+1)
                        continue
                    flow.capture_repair_pending=False
                    started=True
                if not started:
                    started=True
                    flow.run()
                elif flow.round_captured:
                    if not flow.recover_current_question():
                        self.retire(flow);flow=None;attempt=0
                        tracking=TaskTracking(tracking.sequence+1)
                        continue
                else:
                    flow.stop_recovery_observers()
                    flow.stop_audio()
                    self.retire(flow);flow=None
                    continue
                if self.cancelled():break
                if flow.remote_submitted is True:
                    flow.save_completion_timing()
                    attempt=0
                    continue
                raise RuntimeError(getattr(flow,'remote_status',None) or '本题尚未完成，重新核对识别和评价。')
            except Exception as exc:
                if self.cancelled():break
                if flow is not None and flow.remote_submitted is True and not isinstance(exc,SubmissionIncomplete):
                    # Any old playback/rating exception after a possible click
                    # belongs to outcome verification, never capture repair.
                    self.send('status','已记录 Submit，继续核对提交结果和下一题。')
                    flow.refresh_observation_context()
                    if not self.wait(1):break
                    continue
                if isinstance(exc,CaptureRepairPending):
                    self.send('status',str(exc))
                    if not self.wait(8):break
                    continue
                if ((isinstance(exc,PlaybackEvidenceError) and not getattr(flow,'round_captured',False)) or getattr(flow,'capture_validation_failure',None)) and callable(getattr(flow,'recover_failed_capture',None)):
                    flow.capture_repair_pending=True
                    flow.log('capture_repair_pending',reason=getattr(flow,'capture_validation_failure',None) or str(exc),
                             recovery_error=str(exc),recording_preserved=True)
                    self.send('status','首轮录音校验失败，保留当前录音排查；不会重置次数反复重录整题。')
                    continue
                if isinstance(exc,ReviewPending) and flow is not None:
                    flow.pending_review=True
                    self.send('status',str(exc))
                    flow.log('review_pending',reason=str(exc),recording_preserved=True)
                    continue
                if isinstance(exc,SubmissionIncomplete) and flow is not None:
                    flow.remote_submitted=False
                    flow.submission_repair_pending=True
                    flow.submission_observers_stopped=False
                    progress=getattr(flow,'question_progress',None)
                    if progress is not None:progress.incomplete(exc.missing)
                    flow.log('submission_incomplete',missing=exc.missing)
                    flow.comparison_record.update(submission_observation={
                        **flow.comparison_record.data.get('submission_observation',{}),
                        'server_success_verified':False,'result':'incomplete_form','missing':exc.missing})
                    self.send('status','原题未提交成功，正在补选漏项并重新核对。')
                    attempt=0
                    continue
                attempt+=1
                delay=retry_delay(attempt)
                if flow is not None:
                    flow.finished.clear()
                    flow.remote_status=f'自动恢复第 {attempt} 次：{exc}；{delay} 秒后重新核对。'
                    try:
                        flow.log('continuous_recovery',attempt=attempt,reason=str(exc),retry_seconds=delay,
                                 submitted=flow.remote_submitted is True)
                    except OSError:pass
                self.send('status',f'自动恢复第 {attempt} 次：{exc}；{delay} 秒后重试，无需重新开始。')
                if not self.wait(delay):break
        if flow is not None:
            self.retire(flow)
