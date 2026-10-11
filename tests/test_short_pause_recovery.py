import random
import unittest
from unittest.mock import Mock, patch
import numpy as np
from workflow import Workflow, ControlChanged
from vision import Target


class ShortPauseRecoveryTests(unittest.TestCase):
    def exercise(self, failed_press):
        workflow = object.__new__(Workflow)
        image = np.zeros((100,200,3),np.uint8)
        workflow.targets = {'Input':Target('Input',(),(10,40,20,20),'idle',1)}
        workflow.snapshot = Mock(return_value=image)
        workflow.check_page = Mock()
        workflow.log = Mock()
        workflow.status = Mock()
        workflow.rng = random.Random(1)
        workflow.mouse = Mock()
        clock = [0.0]
        state = ['active' if failed_press else 'unknown']
        def move(*args,**kwargs): state[0] = 'idle'
        workflow.mouse.move.side_effect = move
        workflow.mouse.wait.side_effect = lambda seconds: clock.__setitem__(0,clock[0]+seconds)
        workflow.preview_control = Mock(side_effect=ControlChanged('timer at end'))
        with patch('workflow.time.monotonic',side_effect=lambda:clock[0]), \
                patch('workflow.glyph_state',side_effect=lambda *args:state[0]):
            workflow.quiet_players()
        self.assertEqual(state[0],'idle')
        workflow.mouse.move.assert_called_once()
        self.assertLess(clock[0],8)

    def test_unknown_icon_is_cleared_before_rechecking(self):
        self.exercise(False)

    def test_aborted_pause_press_parks_cursor_before_rechecking(self):
        self.exercise(True)
