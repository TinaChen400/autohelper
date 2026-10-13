import unittest
from unittest.mock import Mock,patch
import numpy as np
from vision import Target,classify_task,locate_players


class AutoModeTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((900,1600,3),dtype=np.uint8)
        self.targets={n:Target(n,(x,450,120,25),(x,620,30,30),'idle',.99)
                      for n,x in (('A',200),('B',900))}
        self.lines=[(np.zeros((4,2)),text,.99) for text in
                    ('Omni Tts Elo','Inputs','Here is the transcript: test')]

    def test_positive_page_evidence_recognizes_ab(self):
        self.assertEqual(classify_task(self.targets,self.lines,self.image.shape)[0],('A','B'))

    def test_missing_input_alone_does_not_select_ab(self):
        self.assertIsNone(classify_task(self.targets,[],self.image.shape)[0])

    def test_reference_hint_overrides_ab_page_clues(self):
        for text in ('Input audio','Generate speech using this voice','Play Input first'):
            with self.subTest(text=text):
                self.assertIsNone(classify_task(self.targets,self.lines+[(np.zeros((4,2)),text,.99)],self.image.shape)[0])

    def test_complete_reference_players_select_three_segment_mode(self):
        targets={**self.targets,'Input':Target('Input',(200,150,120,25),(200,320,30,30),'idle',.99)}
        self.assertEqual(classify_task(targets,self.lines,self.image.shape)[0],('Input','A','B'))

    def test_ab_requires_two_observations_without_scroll(self):
        detector=Mock(last_task_lines=self.lines)
        detector.detect.return_value=(self.targets,[],0)
        mouse=Mock()
        _,targets,_,_=locate_players(detector,lambda:self.image,mouse,Mock(),required=None)
        self.assertEqual(set(targets),{'A','B'})
        self.assertEqual(detector.detect.call_count,2)
        mouse.scroll.assert_not_called()

    def test_unchanged_second_frame_reuses_complete_mode_evidence(self):
        detector=Mock(last_task_lines=self.lines)
        detector.detect.return_value=(self.targets,[],0)
        snapshot=Mock(return_value=self.image)
        with patch('vision.glyph_state',return_value='idle'):
            result=locate_players(detector,snapshot,Mock(),Mock(),required=None)
        self.assertEqual(set(result[1]),{'A','B'})
        self.assertEqual(snapshot.call_count,2)
        self.assertEqual(detector.detect.call_count,1)

    def test_changed_second_frame_requires_new_ocr(self):
        detector=Mock(last_task_lines=self.lines)
        detector.detect.return_value=(self.targets,[],0)
        snapshot=Mock(side_effect=[self.image,self.image+20,self.image+20,self.image+20])
        with patch('vision.glyph_state',return_value='idle'):
            locate_players(detector,snapshot,Mock(),Mock(),required=None)
        self.assertEqual(detector.detect.call_count,2)

    def test_long_transcript_scrolls_down_from_input_to_complete_players(self):
        reference=Target('Input',(200,150,120,25),(200,320,30,30),'active',.99)
        detector=Mock(last_task_lines=self.lines)
        detector.detect.side_effect=[({'Input':reference},['A/B clipped'],0),
                                     ({**self.targets,'Input':reference},[],0)]
        mouse=Mock()
        result=locate_players(detector,Mock(side_effect=[self.image,self.image+1]),mouse,Mock(),required=None)
        self.assertEqual(set(result[1]),{'Input','A','B'})
        mouse.scroll.assert_called_once_with(self.image.shape,down=True,notches=1)

    def test_downward_search_never_combines_old_input_with_new_ab_coordinates(self):
        reference=Target('Input',(200,150,120,25),(200,320,30,30),'idle',.99)
        detector=Mock(last_task_lines=self.lines+[(np.zeros((4,2)),'Play Input first',.99)])
        detector.detect.side_effect=[({'Input':reference},[],0)]+[(self.targets,[],0)]*5
        mouse=Mock()
        result=locate_players(detector,Mock(return_value=self.image),mouse,Mock(),required=None)
        self.assertNotIn('Input',result[1])
        self.assertTrue(result[2])
        self.assertEqual(mouse.scroll.call_args_list[0].kwargs['down'],True)
        self.assertEqual(mouse.scroll.call_args_list[1].kwargs['down'],False)

    def test_stationary_downward_search_is_bounded(self):
        detector=Mock(last_task_lines=[])
        detector.detect.return_value=({'Input':Target('Input',(200,150,120,25),(200,320,30,30),'idle',.99)},[],0)
        mouse=Mock()
        result=locate_players(detector,Mock(return_value=self.image),mouse,Mock(),required=None)
        self.assertEqual(mouse.scroll.call_count,2)
        self.assertIn('下滚',result[2][-1])


if __name__=='__main__': unittest.main()
