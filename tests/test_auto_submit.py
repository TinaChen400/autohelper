import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from integrated_app import RecordedWorkflow
from remote_ratings import RatingController,short_ready_task


class AutoSubmitTests(unittest.TestCase):
    def test_short_complete_task_skips_optional_review_but_unresolved_task_does_not(self):
        flow=SimpleNamespace(known_readings={name:SimpleNamespace(total=6) for name in ('A','B')})
        judgments={key:{'judgment':'A'} for key in ('quality','speaker','pronunciation','naturalness','overall')}
        self.assertTrue(short_ready_task(flow,judgments))
        judgments['pronunciation']['judgment']='需复核'
        self.assertFalse(short_ready_task(flow,judgments))
        judgments['pronunciation']={'judgment':'A','uncertainty':'high_uncertainty'}
        self.assertFalse(short_ready_task(flow,judgments))
        judgments['pronunciation']={'judgment':'Both Good'}
        flow.known_readings['B'].total=20
        self.assertFalse(short_ready_task(flow,judgments))

    def test_auto_submit_uses_verified_options_without_requiring_new_ocr_labels(self):
        image=np.zeros((100,200,3),dtype=np.uint8)
        rows=[{'key':'quality','selected':'Both Good','selection_state':'selected','options':{'Both Good':(10,10,40,20)},'label_box':(1,10,5,10)},
              {'key':'factor','selected':'Naturalness','selection_state':'selected','options':{'Naturalness':(10,50,40,20)},'label_box':(1,50,5,10)}]
        record=Mock(data={'model':{'judgments':{'quality':{'judgment':'Both Good'}}},
                         'remote_ratings':{'quality':{'judgment':'Both Good','source':'model_auto'},
                                           'factor':{'judgment':'naturalness','source':'model_auto'}},'auto_actions':[]})
        flow=SimpleNamespace(comparison_record=record,mouse=Mock(),rng=Mock(),status=Mock(),log=Mock(),
            remote_submitted=False,remote_question_changed=False,snapshot=Mock(return_value=image))
        flow.rng.uniform.return_value=0
        observer=Mock(screen={'rows':rows,'submit_boxes':[(70,70,60,20)]},image=image)
        observer.thread.is_alive.return_value=False;observer.input_thread.is_alive.return_value=False
        flow.mouse.click_box.side_effect=lambda box,shape,before_press:before_press()
        controller=RatingController(flow);controller.evidence=Mock(return_value='evidence.png')
        with patch('remote_ratings.same_layout',return_value=True), \
                patch('remote_ratings.refresh_selections',return_value=rows), \
                patch('remote_ratings.submit_enabled',return_value=True), \
                patch('remote_ratings.read_rating_screen') as ocr:
            controller.submit_after_review(observer)
        self.assertTrue(flow.remote_submitted)
        flow.mouse.click_box.assert_called_once()
        ocr.assert_not_called()
        self.assertEqual(record.save_remote.call_args.kwargs,{'submit_clicked':True,'confirm_auto':False})
    def test_refresh_retained_scroll_navigates_only_after_form_reset(self):
        flow=object.__new__(RecordedWorkflow)
        flow.question_marker=(('prompt','1'),)
        flow.mouse=Mock();flow.status=Mock();flow.log=Mock();flow.hwnd=1
        flow.comparison_record=Mock(data={});flow.save_completion_timing=Mock()
        flow.submitted_screen={'rows':[{'key':k,'selection_state':'selected'} for k in ('quality','speaker')]}
        image=np.zeros((20,20,3),dtype=np.uint8)
        flow.grab=Mock(return_value=(image,None));flow.detector=Mock()
        markers=['','','2','2','2','2']
        def detect(image):
            value=markers.pop(0)
            flow.detector.last_task_lines=[(None,'Prompt '+value,1)] if value else []
            return {'A':Mock(),'B':Mock()},[],None
        flow.detector.detect.side_effect=detect
        cleared=[{'key':k,'selection_state':'none'} for k in ('quality','speaker')]
        with patch('integrated_app.time.monotonic',side_effect=range(40)), \
                patch('rating_reader.refresh_selections',return_value=cleared):
            flow.wait_for_next_question()
        flow.mouse.scroll.assert_called_once_with(image.shape,down=False,notches=8)
        flow.mouse.click_box.assert_not_called()
    def test_manual_submit_during_review_is_not_clicked_again(self):
        flow=SimpleNamespace(comparison_record=Mock(data={}),mouse=Mock(),rng=Mock(),
            status=Mock(),log=Mock(),remote_submitted=True,remote_question_changed=False)
        flow.rng.uniform.return_value=6
        observer=Mock()
        observer.thread.is_alive.return_value=False
        observer.input_thread.is_alive.return_value=False
        RatingController(flow).submit_after_review(observer)
        flow.rng.uniform.assert_called_once_with(3,5)
        flow.mouse.click_box.assert_not_called()

    def test_refresh_requires_changed_marker_and_stable_players(self):
        flow=object.__new__(RecordedWorkflow)
        flow.question_marker=(('prompt','1'),)
        flow.mouse=Mock();flow.status=Mock();flow.log=Mock();flow.hwnd=1
        flow.comparison_record=Mock(data={});flow.save_completion_timing=Mock()
        image=np.zeros((20,20,3),dtype=np.uint8)
        flow.grab=Mock(return_value=(image,None))
        flow.detector=Mock()
        markers=['1','1','2','2','2']
        def detect(image):
            value=markers.pop(0)
            flow.detector.last_task_lines=[(None,'Prompt '+value,1)]
            return {'A':Mock(),'B':Mock()},[],None
        flow.detector.detect.side_effect=detect
        with patch('integrated_app.time.monotonic',side_effect=range(30)):
            flow.wait_for_next_question()
        self.assertGreaterEqual(flow.detector.detect.call_count,4)
        flow.mouse.click_box.assert_not_called()
        flow.log.assert_called_once()
