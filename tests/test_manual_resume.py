import unittest
from unittest.mock import Mock, patch
from human_mouse import HumanMouse


class ManualResumeTests(unittest.TestCase):
    def test_pointer_movement_waits_then_adopts_current_position(self):
        mouse = object.__new__(HumanMouse)
        mouse.hwnd, mouse.geometry, mouse.expected = 123, (0, 0, 100, 100), (0, 0)
        mouse.event = Mock()
        mouse.event.is_set.return_value = False
        mouse.event.wait.return_value = False
        mouse.on_pause = Mock()
        with patch.object(HumanMouse, 'position', return_value=(20, 20)), \
                patch('human_mouse.client', return_value=mouse.geometry), \
                patch('human_mouse.u.GetAsyncKeyState', return_value=0), \
                patch('human_mouse.u.GetAncestor', return_value=123), \
                patch('human_mouse.time.monotonic', side_effect=[0, .6, 1.3]):
            self.assertTrue(mouse.check())
        self.assertEqual(mouse.expected, (20, 20))
        self.assertEqual(mouse.on_pause.call_count, 2)
