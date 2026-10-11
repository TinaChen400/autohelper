import unittest
from unittest.mock import Mock, patch
import numpy as np
from vision import Target, locate_players
from workflow import Workflow


class HeaderRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 100, 3), np.uint8)
        self.targets = {n: Target(n, (1, 1, 10, 10), (1, 30, 15, 15), 'idle', .99) for n in ('Input', 'A', 'B')}

    def test_scroll_then_reidentify_before_returning(self):
        detector, mouse = Mock(), Mock()
        detector.detect.side_effect = [({n:t for n,t in self.targets.items() if n != 'Input'}, [], 0), (self.targets, [], 0)]
        with patch('vision.player_buttons', return_value=[((1, 30, 15, 15), 'idle')]):
            result = locate_players(detector, lambda:self.image, mouse, Mock())
        self.assertEqual(set(result[1]), {'Input', 'A', 'B'})
        mouse.scroll.assert_called_once_with(self.image.shape, down=False, notches=1)

    def test_initial_search_can_scroll_while_reference_is_playing(self):
        detector, mouse = Mock(), Mock()
        detector.detect.side_effect = [({n:t for n,t in self.targets.items() if n != 'Input'}, [], 0), (self.targets, [], 0)]
        with patch('vision.player_buttons', return_value=[((1, 30, 15, 15), 'active')]):
            result = locate_players(detector, lambda:self.image, mouse, Mock())
        mouse.scroll.assert_called_once()
        self.assertEqual(set(result[1]),set(self.targets))

    def test_starting_at_bottom_searches_before_identifying_players(self):
        detector, mouse = Mock(), Mock()
        detector.detect.side_effect = [({}, [], 0), ({}, [], 0), (self.targets, [], 0)]
        frames = iter([self.image, self.image+20, self.image+40])
        result = locate_players(detector, lambda:next(frames), mouse, Mock())
        self.assertEqual(set(result[1]),set(self.targets))
        self.assertEqual(mouse.scroll.call_count,2)
        mouse.scroll.assert_called_with(self.image.shape,down=False,notches=3)

    def test_input_alone_does_not_end_search_early(self):
        detector, mouse = Mock(), Mock()
        detector.detect.side_effect = [({'Input':self.targets['Input']}, [], 0), (self.targets, [], 0)]
        locate_players(detector, lambda:self.image, mouse, Mock())
        mouse.scroll.assert_called_once()

    def test_unchanged_page_bounds_search_and_explains_failure(self):
        detector, mouse = Mock(), Mock()
        detector.detect.return_value = ({}, ['A: 未确认'], 0)
        result = locate_players(detector, lambda:self.image, mouse, Mock())
        self.assertEqual(mouse.scroll.call_count,2)
        self.assertIn('滚轮未生效',result[2][-1])

    def test_preview_preserves_detected_states_without_timer_readings(self):
        workflow = object.__new__(Workflow)
        workflow.targets, workflow.send = self.targets, Mock()
        workflow.preview(self.image, {}, '识别')
        shown = workflow.send.call_args.args[1][1]
        self.assertTrue(all(t.state == 'idle' for t in shown.values()))
