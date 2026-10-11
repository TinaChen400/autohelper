import unittest
from unittest.mock import patch
import dynamic_app


class PointerTests(unittest.TestCase):
    def test_park_inside_remote_label_not_host_border(self):
        with patch.object(dynamic_app, 'foreground', return_value=True), \
                patch.object(dynamic_app, 'client', return_value=(100, 200, 960, 540)), \
                patch.object(dynamic_app, 'visible_at', return_value=True), \
                patch.object(dynamic_app.u, 'SetCursorPos') as move:
            dynamic_app.park_pointer(123, (120, 220, 60, 20), (540, 960, 3))
            move.assert_called_once_with(250, 430)

    def test_do_not_move_when_target_not_foreground(self):
        with patch.object(dynamic_app, 'foreground', return_value=False), \
                patch.object(dynamic_app.u, 'SetCursorPos') as move:
            dynamic_app.park_pointer(123)
            move.assert_not_called()
