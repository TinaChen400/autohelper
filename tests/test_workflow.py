import threading
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
from pathlib import Path
import cv2

from playback import parse_timer, Reading, Segment, RoundTracker, read_players
from human_mouse import HumanMouse, Cancelled, smooth_path, safe_click_point
from workflow import Settings, Workflow
from rating_reader import dimensions_from_lines
from vision import Detector, Target


class TimerTests(unittest.TestCase):
    def test_first_round_requires_readable_idle_origin_before_playing(self):
        readings={'A':Reading('idle')}
        flow=SimpleNamespace(settings=SimpleNamespace(player_order=('A',)),
            targets={'A':object()},snapshot=Mock(),parked=Mock(),
            observe_players=Mock(return_value=(None,dict(readings))))
        with self.assertRaisesRegex(RuntimeError,'开始前未读清'):
            Workflow.prepare_first_round(flow,readings)
        self.assertEqual(flow.observe_players.call_count,2)

    def test_timer_parser_rejects_incomplete_or_impossible_times(self):
        self.assertEqual(parse_timer('0:04 / 0:07'), (4, 7))
        self.assertEqual(parse_timer('O：04/O：07'), (4, 7))
        for text in ('04 07', '0:20/0:2', '0:08/0:06', '0:99/1:00', '0:00/0:00'):
            self.assertIsNone(parse_timer(text))

    def test_initial_idle_at_end_is_not_proof_of_playback(self):
        segment = Segment()
        segment.observe(Reading('idle', 7, 7), 0)
        segment.observe(Reading('idle', 7, 7), 1)
        self.assertFalse(segment.done)

    def test_end_requires_observed_playback_and_two_end_samples(self):
        segment = Segment()
        segment.observe(Reading('active', 0, 7), 0)
        segment.observe(Reading('active', 6, 7), 6)
        segment.observe(Reading('idle', 7, 7), 7)
        self.assertFalse(segment.done)
        segment.observe(Reading('idle', 7, 7), 7.4)
        self.assertTrue(segment.done)

    def test_reset_after_finish_preserves_near_end_evidence(self):
        segment = Segment()
        segment.observe(Reading('active', 6, 7), 6)
        segment.observe(Reading('idle', 0, 7), 6.6)
        self.assertFalse(segment.done)
        segment.observe(Reading('idle', 0, 7), 7.1)
        segment.observe(Reading('idle', 0, 7), 7.5)
        self.assertTrue(segment.done)

    def test_early_pause_does_not_finish_and_eventually_stops(self):
        segment = Segment()
        segment.observe(Reading('active', 2, 7), 0)
        segment.observe(Reading('idle', 2, 7), 1)
        with self.assertRaisesRegex(RuntimeError, '提前停止'):
            segment.observe(Reading('idle', 2, 7), 5)
        self.assertFalse(segment.done)

    def test_full_autoplay_and_skipped_segment(self):
        tracker = RoundTracker()
        tracker.observe({'Input': Reading('active', 1, 2), 'A': Reading('idle', 0, 2), 'B': Reading('idle', 0, 2)}, 0)
        tracker.observe({'Input': Reading('idle', 2, 2), 'A': Reading('active', 0, 2), 'B': Reading('idle', 0, 2)}, 1)
        tracker.observe({'Input': Reading('idle', 2, 2), 'A': Reading('active', 1, 2), 'B': Reading('idle', 0, 2)}, 2)
        tracker.observe({'Input': Reading('idle', 2, 2), 'A': Reading('idle', 2, 2), 'B': Reading('active', 0, 2)}, 3)
        tracker.observe({'Input': Reading('idle', 2, 2), 'A': Reading('idle', 2, 2), 'B': Reading('idle', 2, 2)}, 5)
        self.assertTrue(tracker.observe({'Input': Reading('idle', 2, 2), 'A': Reading('idle', 2, 2), 'B': Reading('idle', 2, 2)}, 5.4))
        with self.assertRaisesRegex(RuntimeError, '跳过'):
            RoundTracker().observe({'B': Reading('active', 0, 2)}, 0)

    def test_next_autoplay_item_can_reset_last_round_end_time(self):
        tracker = RoundTracker()
        tracker.observe({'Input': Reading('active', 0, 6), 'A': Reading('idle', 6, 6), 'B': Reading('idle', 5, 5)}, 0)
        tracker.observe({'Input': Reading('active', 5, 6), 'A': Reading('idle', 6, 6), 'B': Reading('idle', 5, 5)}, 5)
        tracker.observe({'Input': Reading('idle', 6, 6), 'A': Reading('active', 0, 6), 'B': Reading('idle', 5, 5)}, 6)
        self.assertTrue(tracker.segments['Input'].done)
        self.assertTrue(tracker.segments['A'].started)
        tracker.observe({'Input': Reading('idle', 6, 6), 'A': Reading('active', 5, 6), 'B': Reading('idle', 5, 5)}, 11)
        tracker.observe({'Input': Reading('idle', 6, 6), 'A': Reading('idle', 6, 6), 'B': Reading('active', 0, 5)}, 12)
        tracker.observe({'Input': Reading('idle', 6, 6), 'A': Reading('idle', 6, 6), 'B': Reading('idle', 5, 5)}, 17)
        self.assertTrue(tracker.observe({'Input': Reading('idle', 6, 6), 'A': Reading('idle', 6, 6), 'B': Reading('idle', 5, 5)}, 17.4))

    def test_replay_can_start_from_previous_completed_time(self):
        tracker = RoundTracker(('A',))
        tracker.observe({'A': Reading('idle', 6, 6)}, 0)
        tracker.observe({'A': Reading('active', 0, 6)}, .5)
        tracker.observe({'A': Reading('active', 5, 6)}, 5.5)
        tracker.observe({'A': Reading('idle', 6, 6)}, 6.5)
        self.assertTrue(tracker.observe({'A': Reading('idle', 6, 6)}, 7))

    def test_actual_backward_progress_is_confirmed_then_followed(self):
        segment = Segment()
        segment.observe(Reading('active', 4, 6), 0)
        segment.observe(Reading('active', 1, 6), 1)
        self.assertEqual(segment.previous.current, 4)
        segment.observe(Reading('active', 2, 6), 2)
        self.assertEqual(segment.previous.current, 2)
        self.assertEqual(segment.expected_end, 6)

    def test_stall_and_duration_change_stop(self):
        for reading, moment in ((Reading('active', 2, 8), 1), (Reading('active', 2, 7), 16)):
            segment = Segment(); segment.observe(Reading('active', 2, 7), 0)
            with self.assertRaises(RuntimeError): segment.observe(reading, moment)

    def test_timer_ocr_on_native_sample(self):
        image = cv2.imread(str(Path(__file__).parent.parent/'artifacts'/'frame_13.png'))
        if image is None: self.skipTest('Native sample not available')
        detector = Detector()
        targets = {'Input': Target('Input', (), (223,274,32,32), 'idle', 1),
                   'A': Target('A', (), (220,634,32,32), 'active', 1),
                   'B': Target('B', (), (1016,634,32,32), 'idle', 1)}
        readings = read_players(detector, image, targets)
        self.assertEqual((readings['Input'].current, readings['Input'].total), (20,20))
        self.assertEqual((readings['A'].current, readings['A'].total), (4,7))
        self.assertEqual((readings['B'].current, readings['B'].total), (6,6))


class InputTests(unittest.TestCase):
    def test_click_variation_covers_safe_interior_of_circle_and_rectangle(self):
        import random,math
        for width,height in ((30,30),(100,30)):
            rng=random.Random(123)
            points=[safe_click_point((10,20,width,height),rng) for _ in range(200)]
            self.assertGreater(max(p[0] for p in points)-min(p[0] for p in points),width*.45)
            self.assertGreater(max(p[1] for p in points)-min(p[1] for p in points),height*.45)
            for x,y in points:
                self.assertLess(math.hypot((x-10-width/2)/width,(y-20-height/2)/height),.5)

    def test_smooth_path_ends_at_target_and_eases(self):
        points = smooth_path((0,0), (100,0), 20, 5)
        self.assertEqual(points[-1], (100,0))
        self.assertLess(points[0][0], points[10][0]-points[9][0])

    def test_motion_profiles_vary_without_overshooting_or_reversing(self):
        paths=[smooth_path((0,0),(200,0),30,8,easing=e) for e in ('smooth','gentle','cosine')]
        self.assertEqual(len({tuple(p) for p in paths}),3)
        for path in paths:
            self.assertEqual(path[-1],(200,0))
            self.assertTrue(all(0<=x<=200 for x,y in path))
            self.assertTrue(all(a[0]<=b[0] for a,b in zip(path,path[1:])))

    def test_cancel_stops_before_any_input(self):
        event = threading.Event(); event.set()
        with patch('human_mouse.client', return_value=(0,0,100,100)), patch.object(HumanMouse, 'position', return_value=(0,0)):
            mouse = HumanMouse(123,event)
            with patch('human_mouse.u.SetCursorPos') as move:
                with self.assertRaises(Cancelled): mouse.move((10,10),(100,100,3))
                move.assert_not_called()

    def test_settings_reject_reversed_or_extreme_ranges(self):
        Settings().validate()
        for settings in (Settings(4,2), Settings(0,99), Settings(pause_min=5,pause_max=1)):
            with self.assertRaises(ValueError): settings.validate()

    def test_rating_rows_are_separate_from_player_and_quick_replay_labels(self):
        def line(text,x,y): return ([[x,y],[x+100,y],[x+100,y+20],[x,y+20]], text,.99)
        lines = [line('Response A',50,100),line('Response B',500,100),
                 line('Audio Quality *',50,400),line('Response A',500,400),line('Response B',620,400),
                 line('Both Good',740,400),line('Both Bad',860,400)]
        rows = dimensions_from_lines(lines,(800,1000,3))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['name'],'Audio Quality')
        self.assertEqual(set(rows[0]['options']), {'Response A','Response B','Both Good','Both Bad'})


if __name__ == '__main__': unittest.main()
