import unittest
from unittest.mock import Mock,patch
import numpy as np
from integrated_app import RecordedWorkflow
from vision import Target
from workflow import LocationUnconfirmed
import dynamic_app
import cv2
from pathlib import Path


class BackgroundProgressTests(unittest.TestCase):
    def test_remote_cursor_outline_does_not_hide_the_play_triangle(self):
        from vision import glyph_state,player_buttons
        image=cv2.imread(str(Path(__file__).parent/'fixtures'/'hovered_input_cursor.png'))
        self.assertEqual(glyph_state(image,(0,0,image.shape[1],image.shape[0])),'idle')
        self.assertTrue(any(state=='idle' for box,state in player_buttons(image,threshold=170)))
    def test_visible_unfocused_remote_can_still_be_captured_without_input(self):
        image=np.zeros((50,100,3),np.uint8)
        with patch('dynamic_app.u.IsWindow',return_value=True),patch('dynamic_app.u.IsIconic',return_value=False), \
                patch('dynamic_app.foreground',return_value=False) as focused, \
                patch('dynamic_app.client',return_value=(0,0,100,50)), \
                patch('dynamic_app.visible_at',return_value=True),patch('dynamic_app.capture',return_value=None), \
                patch('dynamic_app.frame_bgr',return_value=image):
            self.assertIs(dynamic_app.grab(1)[0],image)
            focused.assert_not_called()

    def test_snapshot_is_read_only_even_between_playback_phases(self):
        flow=object.__new__(RecordedWorkflow)
        image=np.zeros((50,100,3),np.uint8)
        flow.mouse=Mock();flow.grab=Mock(return_value=(image,None));flow.hwnd=1
        self.assertIs(flow.snapshot(),image)
        flow.mouse.check_observation.assert_called_once()
        flow.mouse.check.assert_not_called()

    def test_recovery_uses_question_id_and_geometry_without_waveform_similarity(self):
        flow=object.__new__(RecordedWorkflow)
        flow.snapshot=Mock(return_value=np.zeros((50,100,3),np.uint8));flow.check_page=Mock();flow.log=Mock()
        target=Target('A',(1,1,5,5),(1,20,5,5),'idle',.99)
        flow.targets={'A':target};flow.recovery_identity={'marker':(('prompt','1'),('run','2'))}
        flow.detector=Mock(last_task_lines=[(None,'Prompt 1 Run 2',1)])
        flow.detector.detect.return_value=({'A':target},[],0)
        with patch('audio_lab.task_context.waveform_profiles',side_effect=AssertionError('waveform gate used')):
            flow.verify_recovery_identity()
        flow.detector.last_task_lines=[(None,'Prompt 3 Run 2',1)]
        with self.assertRaises(LocationUnconfirmed):flow.verify_recovery_identity()

    def test_refresh_reads_marker_crop_and_scans_players_only_once(self):
        flow=object.__new__(RecordedWorkflow)
        image=np.zeros((50,100,3),np.uint8)
        flow.question_marker=(('prompt','1'),);flow.mouse=Mock();flow.status=Mock();flow.log=Mock()
        flow.grab=Mock(return_value=(image,None));flow.hwnd=1
        flow.comparison_record=Mock(data={});flow.save_completion_timing=Mock()
        flow.detector=Mock()
        flow.detector.read_question_marker.side_effect=[(('prompt','1'),),(('prompt','2'),),(('prompt','2'),)]
        flow.detector.detect.return_value=({'A':Mock(),'B':Mock()},[],0)
        with patch('integrated_app.time.monotonic',side_effect=range(20)):
            flow.wait_for_next_question()
        flow.detector.detect.assert_called_once()
