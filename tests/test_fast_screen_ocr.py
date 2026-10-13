from types import SimpleNamespace
from unittest.mock import Mock
import unittest
import numpy as np
from screen_ocr import ui_text_lines
from vision import associate,classify_task
from audio_lab.task_context import context_from_lines
from playback import read_players
from vision import Target
from vision import locate_players
from unittest.mock import patch
from pathlib import Path
import cv2
from rating_reader import selected_state


def line(text,x,y,w=100,h=15):
    return (np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]),text,.99)


class FastScreenOCRTests(unittest.TestCase):
    def test_wide_text_box_keeps_gray_selection_and_rejects_empty_surface(self):
        options={'Both Good':(207,15,64,19),'Both Bad':(285,15,58,18),
                 'Response A':(23,16,69,17),'Response B':(118,16,68,17)}
        for name,expected in (('gray_wide_text_selected.png',('Both Good','selected')),
                              ('gray_wide_text_empty.png',(None,'none'))):
            im=cv2.imread(str(Path(__file__).parent/'fixtures'/name))
            self.assertEqual(selected_state(im,options)[:2],expected)

    def test_faded_remote_colon_is_recovered_from_real_pixels(self):
        from vision import Detector
        im=cv2.imread(str(Path(__file__).parent/'fixtures'/'timer_faded_colon.png'))
        result=read_players(Detector(),im,{'Input':Target('Input',(),(12,11,30,30),'active',1)})['Input']
        self.assertEqual((result.current,result.total),(1,10))

    def test_missing_colon_is_never_guessed_from_text(self):
        from playback import parse_timer
        self.assertIsNone(parse_timer('0:01/010'))
        self.assertIsNone(parse_timer('0:0070:06'))

    def test_clipped_players_at_form_bottom_skip_ocr_and_scroll_once(self):
        image=np.full((900,1600,3),255,np.uint8)
        targets={n:Target(n,(x,400,100,20),(x,600,30,30),'idle',1)
                 for n,x in (('Input',100),('A',300),('B',1000))}
        detector=SimpleNamespace(fast_search=True,last_task_lines=[],detect=Mock(return_value=(targets,[],.2)))
        probe=[((200,170,30,30),'idle'),((900,170,30,30),'idle')]
        mouse=Mock()
        with patch('vision.player_buttons',side_effect=[probe,[((200,600,30,30),'idle')]]):
            result=locate_players(detector,Mock(return_value=image),mouse,Mock(),required=None)
        self.assertEqual(set(result[1]),set(targets))
        detector.detect.assert_called_once()
        mouse.scroll.assert_called_once_with(image.shape,down=False,notches=12)

    def test_long_black_selected_factor_is_recognized_from_real_pixels(self):
        im=cv2.imread(str(Path(__file__).parent/'fixtures'/'speaker_factor_selected.png'))
        state=selected_state(im,{'Audio and Recording Quality':(142,562,111,8),
            'Persona Likeness (Speaker Similarity)':(268,562,132,8),
            'Pronunciation Faithfulness':(420,562,102,8),'Naturalness':(542,562,46,8)})
        self.assertEqual(state[:2],('Persona Likeness (Speaker Similarity)','selected'))

    def test_timer_wider_text_detection_retains_strict_validation(self):
        for value,score,expected in (('0:01/0:08',.99,(1,8)),('0:09/0:08',.99,(None,None)),
                                     ('0:01/0:08',.6,(None,None))):
            with self.subTest(value=value,score=score):
                engine=Mock(side_effect=[([],None)]*6+[([(np.zeros((4,2)),value,score)],None)])
                detector=SimpleNamespace(ocr=engine)
                target=Target('B',(),(20,20,30,30),'idle',1)
                result=read_players(detector,np.full((100,300,3),255,np.uint8),{'B':target})['B']
                self.assertEqual((result.current,result.total),expected)
                self.assertNotIn('use_det',engine.call_args.kwargs)

    def test_long_body_is_not_sent_to_ui_recognizer_and_coordinates_survive(self):
        short=line('Response A',10,10)[0]
        long=line('long body',10,40,w=900,h=10)[0]
        engine=Mock(return_value=([short,long],None))
        engine.get_crop_img_list.return_value=[np.ones((15,100,3),np.uint8)]
        engine.text_rec.return_value=([('Response A',.99)],.1)
        rows=ui_text_lines(engine,np.ones((100,1000,3),np.uint8))
        self.assertEqual(len(rows),1)
        np.testing.assert_array_equal(rows[0][0],short)
        self.assertEqual(len(engine.get_crop_img_list.call_args.args[1]),1)
        engine.assert_called_once()
        self.assertFalse(engine.call_args.kwargs['use_cls'])
        self.assertFalse(engine.call_args.kwargs['use_rec'])

    def test_inputs_only_requires_unique_actual_player_before_both_responses(self):
        lines=[line('Omni Tts Elo',10,10),line('Inputs',100,100),
               line('Response A',100,340),line('Response B',600,340)]
        responses=[((100,500,30,30),'idle'),((600,500,30,30),'idle')]
        for input_buttons,expected in (([],('A','B')),([((100,220,30,30),'idle')],('Input','A','B'))):
            with self.subTest(expected=expected):
                targets,_=associate(lines,responses+input_buttons,(700,1000,3))
                self.assertEqual(classify_task(targets,lines,(700,1000,3))[0],expected)

    def test_plain_transcript_requires_verified_reference_structure(self):
        lines=[line('A sentence below the reference player.',0,0),line('Translate to English',0,25)]
        self.assertFalse(context_from_lines(lines)['supported_english_text'])
        self.assertEqual(context_from_lines(lines,allow_plain=True)['target_text'],'A sentence below the reference player.')
        self.assertFalse(context_from_lines([line('You should adjust your speaking.',0,0)],allow_plain=True)['supported_english_text'])


if __name__=='__main__':unittest.main()
