from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

import numpy as np

from human_mouse import HumanMouse,Cancelled
from playback import Reading,PlaybackEvidenceError
from workflow import Workflow


class PlaybackObservationTests(unittest.TestCase):
    def mouse(self):
        mouse=object.__new__(HumanMouse)
        mouse.hwnd=123;mouse.geometry=(0,0,100,100)
        mouse.expected=(0,0);mouse.position=Mock(return_value=(80,80))
        mouse.event=Mock(is_set=Mock(return_value=False),wait=Mock(return_value=False))
        mouse.on_pause=Mock();mouse.on_interference=Mock()
        return mouse

    def test_manual_pointer_motion_does_not_block_full_round_observations(self):
        clock=[0.0];mouse=self.mouse()
        mouse.check_context=Mock()
        mouse.check=Mock(side_effect=AssertionError('blocking input guard used while sampling'))
        mouse.move=Mock(side_effect=AssertionError('input issued during playback observation'))
        mouse.event.wait.side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
        order=('Input','A','B');seen=[]
        def read(*args):
            index=int(clock[0]//2)
            rows={name:Reading('idle',2,2) for name in order}
            if index<3:rows[order[index]]=Reading('active',int(clock[0]%2),2)
            seen.append(rows)
            return rows
        with tempfile.TemporaryDirectory() as directory:
            flow=object.__new__(Workflow)
            flow.mouse=mouse;flow.hwnd=123;flow.directory=Path(directory)
            flow.grab=Mock(return_value=(np.zeros((60,90,3),np.uint8),None))
            flow.targets={};flow.known_readings={};flow.detector=Mock()
            flow.check_page=Mock();flow.log=Mock();flow.preview=Mock();flow.status=Mock()
            flow.identity=(10,10,40,20);flow.observation_only_playback=True
            with patch('human_mouse.u.GetAsyncKeyState',return_value=0),\
                 patch('human_mouse.time.monotonic',side_effect=lambda:clock[0]),\
                 patch('workflow.read_players',side_effect=read):
                flow.await_chain(order)
            self.assertTrue(all(any(rows[name].state=='active' for rows in seen) for name in order))
            self.assertGreater(len(seen),15)
            self.assertFalse(flow.observing_playback)
            mouse.check.assert_not_called();mouse.move.assert_not_called()
            mouse.on_interference.assert_called_once()
            self.assertTrue(mouse.on_interference.call_args.kwargs['observation_continues'])

    def test_observation_context_failure_restores_normal_guard(self):
        flow=object.__new__(Workflow)
        flow._await_chain=Mock(side_effect=Cancelled('focus lost'))
        with self.assertRaises(Cancelled):flow.await_chain(('B',))
        self.assertFalse(flow.observing_playback)

    def test_focus_loss_waits_for_return_without_discarding_the_round(self):
        mouse=self.mouse();mouse.check_context=Mock()
        with patch('human_mouse.u.GetAsyncKeyState',return_value=0),\
             patch('human_mouse.u.GetForegroundWindow',side_effect=[999,123]),\
             patch('human_mouse.u.GetAncestor',side_effect=lambda hwnd,level:hwnd):
            self.assertTrue(mouse.check())
        self.assertEqual(mouse.event.wait.call_count,1)

    def test_read_only_sampling_ignores_focus_but_checks_cancel_and_geometry(self):
        mouse=self.mouse()
        with patch('human_mouse.u.GetAsyncKeyState',return_value=0),\
             patch('human_mouse.u.GetAncestor',side_effect=lambda hwnd,level:hwnd),\
             patch('human_mouse.u.GetForegroundWindow',return_value=999),\
             patch('human_mouse.client',return_value=mouse.geometry):
            mouse.check_observation()
        mouse.event.is_set.return_value=True
        with self.assertRaises(Cancelled):mouse.check_observation()

    def test_unclear_progress_gets_fresh_frames_but_never_waits_forever(self):
        clock=[0.]
        flow=object.__new__(Workflow)
        flow.targets={};flow.known_readings={'A':Reading('idle',0,20)};flow.detector=Mock()
        flow.snapshot=Mock(return_value=np.zeros((40,40,3),np.uint8));flow.check_page=Mock()
        flow.mouse=Mock();flow.log=Mock();flow.status=Mock();flow.preview=Mock();flow.identity=None
        flow.mouse.wait_observation.side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
        def read(*args):return {'A':Reading('active',0,20) if clock[0]==0 else Reading('unknown')}
        with patch('workflow.time.monotonic',side_effect=lambda:clock[0]),patch('workflow.read_players',side_effect=read):
            with self.assertRaisesRegex(PlaybackEvidenceError,'连续 12 秒'):flow.await_chain(('A',))
        self.assertGreater(flow.snapshot.call_count,30)
        self.assertLess(clock[0],13)
        flow.mouse.click_box.assert_not_called()

    def test_temporary_timer_failure_recovers_and_completes_from_actual_new_frames(self):
        clock=[0.]
        with tempfile.TemporaryDirectory() as directory:
            flow=object.__new__(Workflow)
            flow.directory=Path(directory);flow.targets={};flow.known_readings={'A':Reading('idle',0,5)}
            flow.detector=Mock();flow.snapshot=Mock(return_value=np.zeros((40,40,3),np.uint8));flow.check_page=Mock()
            flow.mouse=Mock();flow.log=Mock();flow.status=Mock();flow.preview=Mock();flow.identity=None
            flow.mouse.wait_observation.side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
            def read(*args):
                if 1<=clock[0]<2:return {'A':Reading('active')}
                return {'A':Reading('active' if clock[0]<5 else 'idle',min(5,int(clock[0])),5)}
            with patch('workflow.time.monotonic',side_effect=lambda:clock[0]),patch('workflow.read_players',side_effect=read):
                flow.await_chain(('A',))
            events=[call.args[0] for call in flow.log.call_args_list]
            self.assertIn('playback_reading_unclear',events)
            self.assertIn('playback_reading_recovered',events)
            self.assertIn('chain_completed',events)
            flow.mouse.click_box.assert_not_called()


if __name__=='__main__':unittest.main()
