import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
from remote_ratings import SubmissionIncomplete,submission_missing_rows,rating_completion,choose_action
from integrated_app import RecordedWorkflow
from continuous_runner import ContinuousRunner


class SubmissionRepairTests(unittest.TestCase):
    def row(self,key,state='selected'):
        return {'key':key,'selection_state':state,'selected':'Both Good' if state=='selected' else None}

    def test_empty_and_ambiguous_pixels_override_stale_saved_answers(self):
        record=SimpleNamespace(data={'model':{'judgments':{'quality':{'judgment':'Both Good'}}},
                                    'remote_ratings':{'quality':{'judgment':'Both Good'},'factor':{'judgment':'speaker'}}})
        for state in ('none','ambiguous'):
            result=rating_completion(record,{'rows':[self.row('quality',state)],'submit_enabled':False})
            self.assertIn('quality',result['missing'])

    def test_partial_missing_form_differs_from_fully_reset_next_question(self):
        old=[self.row('quality'),self.row('pronunciation')]
        self.assertEqual(submission_missing_rows(old,[self.row('quality'),self.row('pronunciation','none')]),['pronunciation'])
        self.assertEqual(submission_missing_rows(old,[self.row('quality','none'),self.row('pronunciation','none')]),[])
        for choice in ('Both Good','Both Bad'):
            row={**self.row('quality','none'),'options':{choice:(10,10,20,10)}}
            self.assertEqual(choose_action(row,{'quality':{'judgment':choice}},set())[2],choice)

    def test_unchanged_partial_form_triggers_repair_without_another_submit_click(self):
        flow=object.__new__(RecordedWorkflow)
        image=np.zeros((50,100,3),np.uint8)
        flow.mouse=Mock();flow.status=Mock();flow.log=Mock();flow.hwnd=1
        flow.grab=Mock(return_value=(image,None));flow.question_marker=(('prompt','1'),)
        flow.detector=Mock();flow.detector.read_question_marker.return_value=()
        old=[self.row('quality'),self.row('pronunciation')]
        flow.submitted_screen={'rows':old,'submit_boxes':[(1,1,5,5)]};flow.submitted_image=image
        current=[self.row('quality'),self.row('pronunciation','none')]
        with patch('integrated_app.time.monotonic',side_effect=range(30)), \
                patch('rating_reader.refresh_selections',return_value=current), \
                patch('rating_reader.submit_enabled',return_value=False),patch('remote_ratings.same_layout',return_value=True):
            with self.assertRaises(SubmissionIncomplete):flow.wait_for_next_question()
        flow.mouse.click_box.assert_not_called();flow.mouse.scroll.assert_not_called()

    def test_runner_repairs_rejected_submission_in_same_session_without_replaying(self):
        event=threading.Event()
        flow=SimpleNamespace(finished=threading.Event(),remote_submitted=False,round_captured=True,
            run=Mock(),recover_current_question=Mock(),wait_for_next_question=Mock(),log=Mock(),
            stop_recovery_observers=Mock(),save_completion_timing=Mock(),refresh_observation_context=Mock(),
            comparison_record=Mock(data={}))
        flow.run.side_effect=lambda:setattr(flow,'remote_submitted',True)
        def recover():flow.remote_submitted=True;return True
        flow.recover_current_question.side_effect=recover
        outcomes=iter([False,True])
        def refresh():
            if not next(outcomes):raise SubmissionIncomplete(['quality'])
            event.set()
        flow.wait_for_next_question.side_effect=refresh
        ContinuousRunner(event,Mock(),lambda:flow,lambda:True,lambda:False).run()
        flow.run.assert_called_once();flow.recover_current_question.assert_called_once()
        self.assertTrue(flow.submission_repair_pending)
