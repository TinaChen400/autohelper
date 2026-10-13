import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock,patch
import numpy as np

from question_identity import relation
from continuous_runner import ContinuousRunner
from remote_ratings import ReviewPending,SelectionTracker,consistent_preference,preference_conflict
from audio_lab.decision_engine import pronunciation_candidate,reviewed_pronunciation,tokens
from text_follow import phrase_position,target_layout
from human_mouse import HumanMouse


class StabilityRepairsTests(unittest.TestCase):
    def test_corroborated_id_correction_is_used_for_fill_and_next_page_boundary(self):
        from integrated_app import RecordedWorkflow
        flow=object.__new__(RecordedWorkflow)
        original=(('prompt','678818'),('run','23998'))
        corrected=(('prompt','670818'),('run','23998'))
        data={'question_marker':original}
        flow.comparison_record=SimpleNamespace(data=data,update=lambda **values:data.update(values))
        flow.question_marker=original;flow.mouse=Mock(interference_count=0)
        flow.detector=SimpleNamespace(last_task_lines=[(None,'Prompt 670818 Run 23998',.99)])
        flow.identify=Mock();flow.response_identity=Mock(return_value='same');flow.log=Mock()
        flow.ensure_rating_identity()
        self.assertEqual(flow.question_marker,corrected)
        self.assertEqual(data['question_marker'],original)
        self.assertEqual(data['verified_question_marker'],corrected)
        image=np.zeros((50,100,3),np.uint8)
        flow.hwnd=1;flow.grab=Mock(return_value=(image,None));flow.status=Mock()
        flow.save_completion_timing=Mock()
        next_marker=(('prompt','670900'),('run','23999'))
        flow.detector.read_question_marker=Mock(side_effect=[corrected,corrected,corrected,next_marker,next_marker])
        flow.detector.detect=Mock(return_value=({'A':Mock(),'B':Mock()},[],0))
        with patch('integrated_app.time.monotonic',side_effect=range(30)):
            flow.wait_for_next_question()
        self.assertEqual(flow.detector.read_question_marker.call_count,5)
        self.assertEqual(flow.log.call_args.kwargs['marker'],next_marker)

    def test_numeric_transcript_is_valid_target_not_missing_english(self):
        from audio_lab.task_context import context_from_lines
        def line(text):return (np.array([[0,0],[800,0],[800,20],[0,20]]),text,.99)
        context=context_from_lines([line('Here is the transcript: 9 times12 equals 108.')])
        self.assertTrue(context['supported_english_text'])
        self.assertEqual(pronunciation_candidate(context['target_text'],
            {'A':'9 times 12 equals 108.','B':'No times Dotsa equals'})[0]['judgment'],'A')
        self.assertFalse(context_from_lines([line('Here is the transcript: ...')])['supported_english_text'])

    def test_failed_decisions_leave_pending_to_recover_without_new_recording(self):
        from integrated_app import RecordedWorkflow
        flow=SimpleNamespace(remote_question_changed=False,pending_review=True,
            rating_observer=SimpleNamespace(stopped=threading.Event()),
            comparison_record=SimpleNamespace(data={'model':{'pipeline':{'stage_errors':{'decisions':'missing text'}}}}))
        self.assertTrue(RecordedWorkflow.resume_pending_review(flow))
        self.assertFalse(flow.pending_review)

    def test_recovery_preserves_accepted_transcript_instead_of_replacing_with_worse_ocr(self):
        import tempfile
        from integrated_app import RecordedWorkflow
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as folder:
            flow=object.__new__(RecordedWorkflow)
            flow.audio_folder=Path(folder)
            (flow.audio_folder/'task_context.json').write_text(json.dumps(
                {'supported_english_text':True,'target_text':'9 times 12 equals 108.'}),encoding='utf-8')
            marker=(('prompt','10773749'),('run','24033'))
            flow.question_marker=marker
            flow.comparison_record=SimpleNamespace(data={'question_marker':marker,'model':
                {'state':'partial_ready','judgments':{'pronunciation':{'judgment':'需复核'}},
                 'pipeline':{'stage_errors':{'decisions':'missing text'}}}})
            flow.detector=SimpleNamespace(last_task_lines=[(None,'Prompt 10773749 Run 24033',1)])
            for name in ('stop_recovery_observers','refresh_observation_context','snapshot','identify',
                         'log','verify_recovery_identity','inspect_ratings','status'):
                setattr(flow,name,Mock())
            flow.recover_saved_decisions=Mock(return_value=True)
            with patch('audio_lab.task_context.capture_context') as reread:
                self.assertTrue(flow.recover_current_question())
            reread.assert_not_called()
            flow.recover_saved_decisions.assert_called_once()
            flow.inspect_ratings.assert_called_once()

    def test_prompt_digit_loss_does_not_prove_new_task(self):
        original={'prompt':'12455220','run':'24031'}
        shortened={'prompt':'1245522','run':'24031'}
        self.assertEqual(relation(original,shortened,'same'),'same')
        self.assertEqual(relation(shortened,original,'same'),'same')
        self.assertEqual(relation(original,shortened),'unknown')
        self.assertEqual(relation(original,shortened,'different'),'different')
        self.assertEqual(relation(original,{'prompt':'12455220'}),'same')

    def test_live_selected_speaker_with_tight_ocr_box_is_not_misread_as_empty(self):
        import cv2
        from rating_reader import selected_state
        image=cv2.imread(str(Path(__file__).parent/'fixtures/selected_speaker_tight_text.png'))
        options={'Response A':(1322,514,72,19),'Response B':(1417,514,71,19),
                 'N/A':(1271,514,29,18),'Both Good':(1508,515,62,17),'Both Bad':(1585,513,58,20)}
        selected,state,_=selected_state(image,options)
        self.assertEqual((selected,state),('Response A','selected'))

    def test_long_supported_paragraph_keeps_word_geometry_if_chunk_marker_loses_colon(self):
        import cv2,tempfile
        from audio_lab.task_context import capture_context
        root=Path(__file__).resolve().parents[1]
        image=cv2.imread(str(root/'tests/fixtures/long_paragraph_follow.png'))
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as folder:
            context=capture_context(SimpleNamespace(),image,(0,0,image.shape[1],image.shape[0]),folder)
            self.assertTrue(context['supported_english_text'])
            self.assertGreater(len(context['transcript_words']),25)
            self.assertTrue(all(len(word['box'])==4 for word in context['transcript_words']))

    def test_pending_review_keeps_same_flow_and_never_generates_or_records_again(self):
        event=threading.Event()
        flow=SimpleNamespace(finished=threading.Event(),round_captured=True,remote_submitted=False,
            pending_review=False,run=Mock(side_effect=ReviewPending('uncertain pronunciation')),
            log=Mock(),stop_recovery_observers=Mock(),recover_current_question=Mock(),
            resume_pending_review=Mock(side_effect=lambda:event.set() or False))
        factory=Mock(return_value=flow)
        ContinuousRunner(event,Mock(),factory,lambda:True,lambda:False).run()
        factory.assert_called_once();flow.run.assert_called_once();flow.recover_current_question.assert_not_called()

    def test_pronunciation_formatting_is_not_an_error(self):
        self.assertEqual(tokens('in 3O characters'),tokens('in thirty characters'))
        self.assertEqual(tokens('Why? Its just, ugh.'),tokens("Why? It's just, ugh."))
        self.assertEqual(tokens('its own source'),['its','own','source'])

    def test_two_decodes_can_compare_responses_without_blaming_common_asr_errors(self):
        target='No atenolol is not metabolized by CYP2D6.'
        first={'A':'No a tenelol is not metabolized by CYP 2D6.',
               'B':'No a tenelol is not metabolized by SIP.'}
        second={'A':'No a tenelol is not metabolized by CYP 2D6.',
                'B':'No a tenelol is metabolized by SIP.'}
        self.assertEqual(pronunciation_candidate(target,first)[0]['judgment'],'需复核')
        reviewed,_=reviewed_pronunciation(target,first,second)
        self.assertEqual(reviewed['judgment'],'A')
        self.assertEqual(reviewed['evidence']['review_method'],'two_decode_relative_alignment')
        tied={name:'No a tenelol is not metabolized by SIP.' for name in ('A','B')}
        self.assertEqual(reviewed_pronunciation(target,tied,tied)[0]['judgment'],'需复核')

    def test_preference_can_differ_from_naturalness_with_a_supported_factor(self):
        choices={'naturalness':'A','speaker':'B','overall':'B','factor':'speaker'}
        self.assertFalse(preference_conflict(choices))
        choices['factor']='naturalness'
        self.assertTrue(preference_conflict(choices))
        corrections=consistent_preference({'overall':{'dominant_factor':'naturalness'}},choices,
            {'overall':'model_auto','factor':'model_auto'})
        self.assertEqual(corrections,{'overall':'A'})

    def test_explicit_human_preference_is_preserved_and_auto_factor_is_corrected(self):
        choices={'naturalness':'B','speaker':'A','overall':'A','factor':'naturalness'}
        corrections=consistent_preference({'overall':{}},choices,{'overall':'human_remote','factor':'model_auto'})
        self.assertEqual(corrections,{'factor':'speaker'})

    def test_unknown_old_selections_are_not_independent_human_labels(self):
        tracker=SelectionTracker({},require_click_evidence=True)
        rows=[{'key':'naturalness','selected':'Response A','selection_state':'selected'}]
        tracker.observe(rows)
        self.assertEqual(tracker.observe(rows)['naturalness']['source'],'existing_remote')
        self.assertEqual(tracker.observe(rows,{'naturalness':'A'})['naturalness']['source'],'human_remote')

    def test_speech_phrase_matches_text_content_not_duration(self):
        target='The little cat was sleeping beside the window.'
        self.assertEqual(phrase_position(target,'cat was sleeping beside'),5)
        self.assertIsNone(phrase_position(target,'Completely unrelated speech.'))
        visual=[{'text':word,'box':[i*10,20,8,10]} for i,word in enumerate(('Instructions','The','little','cat','was','sleeping','beside','the','window'))]
        layout=target_layout(target,visual)
        self.assertEqual(layout[0]['target_index'],0)
        self.assertEqual(layout[0]['box'],[10,20,8,10])

    def test_observation_cursor_step_never_waits_or_steals_focus(self):
        mouse=object.__new__(HumanMouse)
        mouse.hwnd=1;mouse.expected=(0,0);mouse.rng=__import__('random').Random(7)
        mouse.geometry=(0,0,100,100);mouse.check_observation=Mock(return_value=False)
        mouse.wait=Mock();mouse.check=Mock()
        with patch('human_mouse.u.GetAncestor',return_value=1),patch('human_mouse.u.SetCursorPos',return_value=True):
            mouse.observation_glide((80,80),(100,100),now=10)
            mouse.observation_glide(None,(100,100),now=11)
        mouse.wait.assert_not_called();mouse.check.assert_not_called()
        self.assertGreater(mouse.expected[0],0)
        mouse.check_observation.return_value=True
        self.assertFalse(mouse.observation_glide((10,10),(100,100),now=12))


if __name__=='__main__':unittest.main()
