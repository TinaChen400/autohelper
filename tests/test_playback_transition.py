import unittest
from playback import Reading, RoundTracker


class PlaybackTransitionTests(unittest.TestCase):
    def test_autoplay_active_icon_precedes_timer_reset(self):
        tracker = RoundTracker(('B',), {'B': Reading('idle', 25, 25)})
        tracker.observe({'B': Reading('active', 25, 25)}, 0)
        tracker.observe({'B': Reading('active', 0, 25)}, .44)
        tracker.observe({'B': Reading('active', 24, 25)}, 24.44)
        self.assertFalse(tracker.observe({'B': Reading('idle', 25, 25)}, 25.44))
        self.assertTrue(tracker.observe({'B': Reading('idle', 25, 25)}, 25.88))

    def test_single_backward_ocr_error_does_not_change_timeline(self):
        from playback import Segment
        segment = Segment()
        segment.observe(Reading('active', 10, 25), 0)
        segment.observe(Reading('active', 1, 25), 5)
        segment.observe(Reading('active', 16, 25), 6)
        self.assertEqual(segment.previous.current, 16)
        self.assertIsNone(segment.backward_candidate)

    def test_known_durations_survive_missing_timers_until_round_end(self):
        known = {n: Reading('idle', total, total) for n, total in zip(('Input', 'A', 'B'), (19, 44, 40))}
        tracker = RoundTracker(known=known)
        for now in range(105):
            active = 'Input' if now < 19 else 'A' if now < 63 else 'B' if now < 103 else None
            # All timer text temporarily disappears; icons and the previously
            # read lengths remain usable, including the final B completion.
            rows = {n: Reading('active' if n == active else 'idle') for n in known}
            finished = tracker.observe(rows, now)
        self.assertTrue(finished)

    def test_elapsed_time_alone_cannot_finish_an_active_player(self):
        tracker = RoundTracker(('B',), {'B': Reading('idle', 0, 5)})
        for now in range(8):
            self.assertFalse(tracker.observe({'B': Reading('active')}, now))
        self.assertFalse(tracker.observe({'B': Reading('idle')}, 8))
        self.assertTrue(tracker.observe({'B': Reading('idle')}, 9))

    def test_idle_time_does_not_count_as_stalled_new_play(self):
        tracker = RoundTracker()
        for second in range(19):
            tracker.observe({'Input': Reading('active', second, 19),
                             'A': Reading('idle', 44, 44),
                             'B': Reading('idle', 30, 40)}, second)
        tracker.observe({'Input': Reading('idle', 19, 19),
                         'A': Reading('active'),
                         'B': Reading('idle', 30, 40)}, 19)
        for second in range(44):
            tracker.observe({'Input': Reading('idle', 19, 19),
                             'A': Reading('active', second, 44),
                             'B': Reading('idle', 30, 40)}, 20+second)
        for second in range(40):
            tracker.observe({'Input': Reading('idle', 19, 19),
                             'A': Reading('idle', 44, 44),
                             'B': Reading('active', second, 40)}, 64+second)
        idle = {'Input': Reading('idle', 19, 19), 'A': Reading('idle', 44, 44), 'B': Reading('idle', 40, 40)}
        self.assertFalse(tracker.observe(idle, 104))
        self.assertTrue(tracker.observe(idle, 105))

    def test_real_stalled_progress_still_detected(self):
        tracker = RoundTracker(('A',))
        tracker.observe({'A': Reading('idle', 44, 44)}, 0)
        tracker.observe({'A': Reading('active')}, 30)
        with self.assertRaisesRegex(RuntimeError, '15 秒'):
            tracker.observe({'A': Reading('active')}, 90)
