import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np

from integrated_app import RecordedWorkflow
from rating_reader import marker_changed,marker_matches
from audio_lab.task_context import capture_context


class RatingPhaseRecoveryTests(unittest.TestCase):
    marker=(('prompt','12449094'),('run','24020'))

    def test_compressed_timer_with_visible_dot_separator_is_read_without_inventing_digits(self):
        from playback import parse_timer,read_players
        from vision import Target,Detector
        self.assertEqual(parse_timer('0.03 / 0:03'),(3,3))
        self.assertIsNone(parse_timer('003 / 0:03'))
        image=cv2.imread(str(Path(__file__).parent/'fixtures/compressed_input_timer.png'))
        reading=read_players(Detector(),image,{'Input':Target('Input',(0,0,1,1),(7,6,30,30),'idle',1)})['Input']
        self.assertEqual((reading.current,reading.total),(3,3))

    def test_manual_scroll_after_submit_searches_header_instead_of_waiting_on_old_coordinates(self):
        flow=object.__new__(RecordedWorkflow)
        image=np.zeros((50,100,3),np.uint8)
        flow.mouse=Mock();flow.status=Mock();flow.log=Mock();flow.hwnd=1
        flow.question_marker=self.marker;flow.grab=Mock(return_value=(image,None))
        flow.detector=Mock()
        flow.detector.read_question_marker.side_effect=[(),(),(('prompt','12449175'),), (('prompt','12449175'),)]
        flow.detector.detect.return_value=({'A':Mock(),'B':Mock()},[],0)
        flow.comparison_record=Mock(data={});flow.save_completion_timing=Mock()
        with patch('integrated_app.time.monotonic',side_effect=range(30)):
            flow.wait_for_next_question()
        flow.mouse.scroll.assert_called_once_with(image.shape,down=False,notches=6)
        flow.mouse.click_box.assert_not_called()
        self.assertIn('refresh_header_search',[call.args[0] for call in flow.log.call_args_list])

    def test_observed_run_ocr_errors_do_not_restart_same_prompt(self):
        for run in ('824020','24820'):
            observed=(('prompt','12449094'),('run',run))
            self.assertFalse(marker_changed(self.marker,observed))
            self.assertTrue(marker_matches(self.marker,observed))
            self.assertTrue(marker_changed(self.marker,observed,form_reset=True))
        self.assertTrue(marker_changed(self.marker,(('prompt','12455340'),('run','24032'))))
        self.assertFalse(marker_changed(self.marker,(('run','824020'),)))
        self.assertFalse(marker_matches(self.marker,(('run','24020'),)))

    def test_rating_recovery_uses_existing_evaluation_despite_run_ocr_error(self):
        flow=object.__new__(RecordedWorkflow)
        for name in ('stop_recovery_observers','refresh_observation_context','snapshot','log',
                     'verify_recovery_identity','inspect_ratings'):
            setattr(flow,name,Mock())
        flow.question_marker=self.marker
        flow.comparison_record=SimpleNamespace(data={'question_marker':self.marker,
            'model':{'state':'draft_ready','judgments':{'quality':{'judgment':'Both Good'}}}})
        flow.detector=Mock(last_task_lines=[(None,'Prompt 12449094 Run 824020',1)])
        flow.response_identity=Mock(return_value='same')
        flow.identify=Mock();flow.mouse=Mock();flow.run=Mock();flow.start_audio=Mock()
        self.assertTrue(flow.recover_current_question())
        flow.inspect_ratings.assert_called_once()
        flow.run.assert_not_called();flow.start_audio.assert_not_called()
        flow.mouse.scroll.assert_not_called()

    def test_same_prompt_different_responses_is_a_new_task(self):
        flow=object.__new__(RecordedWorkflow)
        flow.log=Mock();flow.response_identity=Mock(return_value='different')
        self.assertTrue(flow.question_has_changed(self.marker,(('prompt','12449094'),('run','24021'))))
        flow.response_identity.return_value='unknown'
        self.assertFalse(flow.question_has_changed(self.marker,(('prompt','12449094'),('run','824020'))))

    def test_old_choices_cannot_satisfy_this_runs_unobserved_required_rows(self):
        from remote_ratings import rating_completion,hidden_earlier_row
        judgments={key:{'judgment':'A'} for key in ('quality','speaker','pronunciation','naturalness','overall')}
        record=SimpleNamespace(data={'model':{'judgments':judgments},'remote_ratings':{
            key:{'judgment':'A'} for key in (*judgments,'factor')}})
        screen={'rows':[{'key':'factor','selected':'Naturalness','selection_state':'selected'}],
                'observed_keys':{'factor'},'submit_enabled':False}
        self.assertFalse(rating_completion(record,screen)['all_required_observed'])
        self.assertEqual(hidden_earlier_row(screen['rows'],record,judgments,{'factor'}),'quality')

    def test_manual_input_invalidates_next_rating_press_until_identity_recheck(self):
        flow=object.__new__(RecordedWorkflow)
        flow.mouse=SimpleNamespace(interference_count=2);flow.rating_input_revision=1
        with self.assertRaisesRegex(RuntimeError,'重新核对题目身份'):flow.guard_rating_action()
        flow.rating_input_revision=2;flow.guard_rating_action()

    def test_real_long_transcript_is_read_completely(self):
        root=Path(__file__).resolve().parents[1]
        image=cv2.imread(str(root/'tests/fixtures/long_target_transcript.png'))
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as folder:
            result=capture_context(SimpleNamespace(),image,(0,0,image.shape[1],image.shape[0]),folder)
            self.assertEqual(result['target_text'].replace('one.Oh','one. Oh'),"I'm afraid to get eaten by one. Oh my goodness, do you have any animals that you're afraid of too? I feel alone here.")
            self.assertEqual(result['ocr_pass'],'word_bounded_transcript_chunks')
            self.assertTrue(result['supported_english_text'])

    def test_real_wrapped_transcript_keeps_all_chunks_and_both_rows(self):
        root=Path(__file__).resolve().parents[1]
        image=cv2.imread(str(root/'tests/fixtures/wrapped_target_transcript.png'))
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as folder:
            result=capture_context(SimpleNamespace(),image,(0,0,image.shape[1],image.shape[0]),folder)
            self.assertTrue(result['supported_english_text'])
            for phrase in ('running out of','have some','coffee in a book','set your alarm','sense of Like calm'):
                self.assertIn(phrase,result['target_text'])
            self.assertNotIn('Delivery:',result['target_text'])

    def test_unclear_text_is_saved_for_retry_without_aborting_audio_capture(self):
        root=Path(__file__).resolve().parents[1]
        image=np.full((150,1000,3),255,np.uint8)
        detector=SimpleNamespace(text_ocr=Mock(return_value=([],None)))
        with tempfile.TemporaryDirectory(dir=root/'artifacts') as folder,patch(
                'audio_lab.task_context.read_transcript_chunks',side_effect=ValueError('unclear word')):
            result=capture_context(detector,image,(0,0,1000,150),folder)
            self.assertFalse(result['supported_english_text'])
            self.assertEqual(json.loads((Path(folder)/'task_context.json').read_text())['ocr_error'],'unclear word')


if __name__=='__main__':unittest.main()
