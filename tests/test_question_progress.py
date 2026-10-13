import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
import numpy as np

from audio_lab.comparison_record import ComparisonRecord
from question_progress import QuestionProgress,resumable_session
from submit_monitor import ManualSubmitMonitor
from integrated_app import RecordedWorkflow
from continuous_runner import ContinuousRunner
from playback import PlaybackEvidenceError

ROOT=Path(__file__).resolve().parents[1]

class QuestionProgressTests(unittest.TestCase):
    def folder(self):return tempfile.TemporaryDirectory(dir=ROOT/'artifacts')

    def test_submit_boundary_survives_restart_and_late_model_completion(self):
        with self.folder() as folder:
            record=ComparisonRecord(folder);progress=QuestionProgress(record)
            self.assertEqual(record.data['question_progress']['phase'],'identifying')
            progress.observe('recording_ready')
            self.assertTrue(progress.submitted('user','visible_button_press'))
            resumed=QuestionProgress(ComparisonRecord.load(folder))
            resumed.observe('analysis_completed',state='draft_ready')
            self.assertEqual(resumed.data['phase'],'submit_pending')
            self.assertFalse(resumed.submitted('assistant','duplicate'))
            self.assertEqual(resumed.data['submit_attempt'],1)
            resumed.page_changed((('prompt','2'),),True)
            resumed.observe('model_rating_click')
            self.assertEqual(resumed.data['phase'],'page_changed')

    def test_missing_form_returns_to_fill_but_preserves_attempt_history(self):
        with self.folder() as folder:
            progress=QuestionProgress(ComparisonRecord(folder))
            progress.submitted('assistant','first')
            progress.incomplete(['pronunciation'])
            self.assertEqual(progress.data['phase'],'rating')
            self.assertTrue(progress.submitted('assistant','second'))
            self.assertEqual(progress.data['submit_attempt'],2)

    def test_manual_page_change_does_not_fabricate_submission(self):
        with self.folder() as folder:
            progress=QuestionProgress(ComparisonRecord(folder))
            progress.page_changed((('prompt','2'),),False)
            self.assertEqual(progress.data['submit_attempt'],0)
            self.assertEqual(progress.data['result'],'page_changed_without_observed_submit')

    def flow(self,folder):
        flow=object.__new__(RecordedWorkflow)
        flow.comparison_record=ComparisonRecord(folder)
        flow.question_progress=QuestionProgress(flow.comparison_record)
        flow.submission_guard=threading.RLock()
        flow.remote_submitted=False;flow.remote_question_changed=False
        flow.status=Mock();flow.log=Mock();flow.save_completion_timing=Mock()
        flow.capture_repair_pending=False;flow.pending_review=False
        return flow

    def test_manual_submit_monitor_works_without_rating_observer_and_is_idempotent(self):
        with self.folder() as folder:
            flow=self.flow(folder);flow.rating_observer=None
            monitor=ManualSubmitMonitor(flow)
            image=np.zeros((100,200,3),np.uint8)
            screen={'rows':[],'submit_boxes':[(50,50,80,20)],'submit_enabled':True}
            monitor.publish(screen,image,(10,20,200,100),now=10)
            self.assertTrue(monitor.click((80,60),(10,20,200,100),now=10.2))
            self.assertTrue(flow.remote_submitted)
            self.assertFalse(flow.comparison_record.data['submission_observation']['server_success_verified'])
            self.assertEqual(flow.question_progress.data['actor'],'user')
            self.assertFalse(monitor.click((80,60),(10,20,200,100),now=10.3))
            flow.save_completion_timing.assert_called_once()

    def test_stale_or_moved_form_is_not_treated_as_manual_submit(self):
        flow=SimpleNamespace(note_submission=Mock())
        monitor=ManualSubmitMonitor(flow);image=np.zeros((100,200,3),np.uint8)
        screen={'rows':[],'submit_boxes':[(50,50,80,20)],'submit_enabled':True}
        monitor.publish(screen,image,(0,0,200,100),now=10)
        self.assertFalse(monitor.click((80,60),(0,0,200,100),now=13))
        self.assertFalse(monitor.click((80,60),(10,0,200,100),now=10.2))
        flow.note_submission.assert_not_called()

    def test_all_answers_reset_invalidates_old_submit_until_identity_is_reverified(self):
        flow=SimpleNamespace(note_submission=Mock())
        monitor=ManualSubmitMonitor(flow);image=np.zeros((100,200,3),np.uint8)
        rows=[{'key':key,'selection_state':'selected'} for key in ('quality','speaker')]
        monitor.publish({'rows':rows,'submit_boxes':[(50,50,80,20)],'submit_enabled':True},image,(0,0,200,100),now=10)
        cleared=[{**row,'selection_state':'none'} for row in rows]
        with patch('submit_monitor.same_layout',return_value=True),patch('submit_monitor.refresh_selections',return_value=cleared):
            monitor.observe_frame(image,(0,0,200,100),now=11)
        self.assertTrue(monitor.invalidated)
        self.assertFalse(monitor.click((80,60),(0,0,200,100),now=11.1))
        flow.note_submission.assert_not_called()

    def test_pending_submit_exception_never_reenters_recording_or_rating(self):
        event=threading.Event()
        flow=SimpleNamespace(finished=threading.Event(),remote_submitted=False,round_captured=False,
            run=Mock(),log=Mock(),stop_recovery_observers=Mock(),stop_audio=Mock(),
            refresh_observation_context=Mock(),recover_failed_capture=Mock(),recover_current_question=Mock())
        def run():
            flow.remote_submitted=True
            raise PlaybackEvidenceError('late playback exception after manual submit')
        flow.run.side_effect=run
        flow.wait_for_next_question=Mock(side_effect=lambda:event.set())
        factory=Mock(return_value=flow)
        runner=ContinuousRunner(event,Mock(),factory,lambda:True,lambda:False)
        runner.wait=Mock(return_value=True)
        runner.run()
        factory.assert_called_once();flow.run.assert_called_once()
        flow.recover_failed_capture.assert_not_called();flow.recover_current_question.assert_not_called()
        flow.wait_for_next_question.assert_called_once()

    def test_start_selects_only_latest_resumable_question(self):
        with self.folder() as folder:
            path=Path(folder)/'integrated_session_20261013_200000';path.mkdir()
            for name in ('session.json','task_context.json'):(path/name).write_text('{}')
            (path/'comparison.json').write_text(json.dumps({'question_progress':{'phase':'submit_pending'},'model':{'state':'draft_ready'}}))
            self.assertEqual(resumable_session(folder),path)
            newer=Path(folder)/'integrated_session_20261013_200100';newer.mkdir()
            (newer/'comparison.json').write_text(json.dumps({'question_progress':{'phase':'page_changed'},'model':{'state':'draft_ready'}}))
            self.assertIsNone(resumable_session(folder))

    def test_saved_capture_can_resume_before_model_without_accepting_empty_audio(self):
        with self.folder() as folder:
            path=Path(folder)/'integrated_session_20261013_200000';path.mkdir()
            for name in ('session.json','task_context.json'):(path/name).write_text('{}')
            (path/'comparison.json').write_text(json.dumps({'question_progress':{'phase':'recording'}}))
            (path/'capture_status.json').write_text(json.dumps({'state':'saved','nonzero_samples_total':10}))
            (path/'timeline.jsonl').write_text('{"audio_seconds":1}\n')
            (path/'whole.wav').write_bytes(b'')
            self.assertIsNone(resumable_session(folder))
            (path/'whole.wav').write_bytes(b'RIFF'+b'\0'*100)
            self.assertEqual(resumable_session(folder),path)
            (path/'capture_status.json').write_text(json.dumps({'state':'saved','nonzero_samples_total':0}))
            self.assertIsNone(resumable_session(folder))

if __name__=='__main__':unittest.main()
