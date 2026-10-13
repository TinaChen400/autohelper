import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from continuous_runner import ContinuousRunner,retry_delay


class ContinuousRunnerTests(unittest.TestCase):
    def test_optional_replay_failure_does_not_invalidate_completed_capture(self):
        from playback import PlaybackEvidenceError
        flow=self.flow();flow.round_captured=True
        flow.run.side_effect=PlaybackEvidenceError('optional replay paused')
        flow.recover_failed_capture=Mock()
        runner=self.runner(flow)
        flow.recover_current_question.side_effect=lambda:runner.event.set() or True
        runner.run()
        flow.recover_failed_capture.assert_not_called()
        flow.recover_current_question.assert_called_once()
        self.assertFalse(getattr(flow,'capture_repair_pending',False))

    def test_exhausted_capture_retry_keeps_session_instead_of_resetting_budget(self):
        from playback import PlaybackEvidenceError,CaptureRepairPending
        flow=self.flow();flow.round_captured=False
        flow.run.side_effect=PlaybackEvidenceError('A boundary gap')
        flow.recover_failed_capture=Mock(side_effect=CaptureRepairPending('retained PCM'))
        runner=self.runner(flow)
        runner.wait.side_effect=lambda seconds:runner.event.set() or False
        runner.run()
        runner.factory.assert_called_once()
        flow.run.assert_called_once()
        flow.recover_failed_capture.assert_called_once()
        self.assertTrue(flow.capture_repair_pending)
        flow.recover_current_question.assert_not_called()

    def test_resume_keeps_existing_model_session_instead_of_recording_again(self):
        from pathlib import Path
        flow=self.flow();runner=self.runner(flow)
        flow.directory=Path('saved_session')
        flow.task_tracking={'task_id':'saved_task','task_sequence':3}
        def recover():runner.event.set();return True
        flow.recover_current_question.side_effect=recover
        runner.run(initial_flow=flow)
        runner.factory.assert_not_called();flow.run.assert_not_called()
        flow.recover_current_question.assert_called_once()
    def runner(self,flow,event=None):
        event=event or threading.Event()
        runner=ContinuousRunner(event,Mock(),Mock(return_value=flow),Mock(return_value=True),lambda:False)
        runner.wait=Mock(return_value=True)
        return runner

    def flow(self):
        return SimpleNamespace(finished=threading.Event(),remote_submitted=False,round_captured=True,
            run=Mock(),recover_current_question=Mock(),wait_for_next_question=Mock(),log=Mock(),
            stop_recovery_observers=Mock(),save_completion_timing=Mock(),stop_audio=Mock(),
            refresh_observation_context=Mock(),remote_status='待恢复')

    def test_rating_failure_recovers_same_question_and_submission_is_never_repeated(self):
        flow=self.flow();runner=self.runner(flow)
        flow.run.side_effect=RuntimeError('OCR temporarily unreadable')
        def recover():flow.remote_submitted=True;return True
        flow.recover_current_question.side_effect=recover
        calls=[0]
        def refresh():
            calls[0]+=1
            if calls[0]==1:raise RuntimeError('refresh pending')
            runner.event.set()
        flow.wait_for_next_question.side_effect=refresh
        runner.run()
        flow.run.assert_called_once()
        flow.recover_current_question.assert_called_once()
        self.assertEqual(flow.wait_for_next_question.call_count,2)
        flow.save_completion_timing.assert_called_once()
        flow.refresh_observation_context.assert_called_once()

    def test_switching_window_waits_without_clicks_or_creating_a_session(self):
        runner=self.runner(self.flow())
        runner.remote_ready.return_value=False
        runner.wait.side_effect=lambda seconds:(runner.event.set() or False)
        runner.run()
        runner.factory.assert_not_called()

    def test_esc_stops_in_recovery_wait(self):
        flow=self.flow();runner=self.runner(flow)
        runner.cancel_requested=lambda:True
        runner.run()
        self.assertTrue(runner.event.is_set())
        flow.run.assert_not_called()

    def test_failed_recording_gets_fresh_session_without_reusing_old_progress(self):
        first=self.flow();first.round_captured=False
        first.run.side_effect=RuntimeError('incomplete playback')
        second=self.flow();runner=self.runner(first)
        second.run.side_effect=lambda:runner.event.set()
        runner.factory.side_effect=[first,second]
        runner.run()
        first.stop_audio.assert_called_once()
        first.recover_current_question.assert_not_called()
        second.run.assert_called_once()
        self.assertEqual(first.task_tracking['task_id'],second.task_tracking['task_id'])
        self.assertEqual(second.task_tracking['attempt_number'],2)

    def test_backoff_does_not_leave_recoverable_tasks_idle_for_a_minute(self):
        self.assertEqual([retry_delay(n) for n in (1,2,3,20)],[2,4,8,8])
