from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

from workflow import Workflow
from playback import Reading
from audio_lab.segment_session import progress_interval,first_progress_run


class CaptureTimingTests(unittest.TestCase):
    def test_new_frame_never_reuses_callers_initial_readings(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            flow=object.__new__(Workflow)
            flow.known_readings={};flow.targets={};flow.directory=Path(directory);flow.detector=Mock()
            flow.snapshot=Mock(return_value=np.zeros((30,30,3),np.uint8))
            flow.check_page=Mock();flow.log=Mock();flow.preview=Mock();flow.status=Mock()
            tracker=Mock(segments={})
            tracker.observe.return_value=True
            tracker.summary.return_value='done'
            flow.tracker_type=Mock(return_value=tracker)
            old={'B':Reading('active',0,5)}
            fresh={'B':Reading('active',4,5)}
            with patch('workflow.read_players',return_value=fresh):
                flow.await_chain(('B',),old)
            self.assertIs(tracker.observe.call_args.args[0],fresh)
            self.assertIs(flow.preview.call_args.args[1],fresh)

    def test_one_stale_zero_is_recovered_from_real_trace(self):
        run=[{'audio_seconds':second,'readings':{'B':{'current':current,'total':5}}}
             for second,current in ((15.506,0),(15.928,0),(17.897,0),(20.491,4),(20.913,5))]
        low,high,excluded=progress_interval(run,'B')
        self.assertAlmostEqual(low,15.491)
        self.assertAlmostEqual(high,15.506)
        self.assertEqual(excluded[0]['audio_seconds'],17.897)

    def test_multiple_conflicts_are_not_silently_accepted(self):
        run=[{'audio_seconds':second,'readings':{'B':{'current':current,'total':5}}}
             for second,current in ((15.506,0),(15.928,0),(17.897,0),(19,0),(20.491,4),(20.913,5))]
        with self.assertRaisesRegex(ValueError,'B:'):
            progress_interval(run,'B')

    def test_single_early_digit_requires_long_complete_consensus(self):
        run=[{'audio_seconds':6.9+i*.5,'readings':{'Input':{'current':i//2,'total':19}}} for i in range(40)]
        # Use the real failure pattern: the first of four 16 readings was
        # actually 15, one second before the surrounding consistent clock.
        run[30]['readings']['Input']['current']=16
        run[31]['readings']['Input']['current']=16
        low,high,excluded=progress_interval(run,'Input')
        self.assertEqual([r['audio_seconds'] for r in excluded],[21.9])
        self.assertLessEqual(low,high+.25)
        with self.assertRaises(ValueError):progress_interval(run[20:],'Input')

    def test_short_repeated_nonzero_delay_recovers_only_with_strong_consensus(self):
        values=((55.318,0),(55.756,0),(56.193,1),(56.631,1),(57.053,2),(57.49,2),
                (57.943,3),(60.803,5),(61.225,5),(61.662,5),(62.1,7),(62.521,7),
                (63.912,9),(64.35,9),(64.787,9),(65.209,10))
        run=[{'audio_seconds':second,'readings':{'B':{'current':current,'total':10}}} for second,current in values]
        low,high,excluded=progress_interval(run,'B')
        self.assertAlmostEqual(low,54.803)
        self.assertAlmostEqual(high,54.912)
        self.assertEqual([row['audio_seconds'] for row in excluded],[61.225,61.662])
        # Removing the corroborating early values must not turn weak evidence
        # into an accepted timing correction.
        with self.assertRaises(ValueError): progress_interval(run[7:],'B')

    def test_two_adjacent_early_frames_require_95_percent_complete_consensus(self):
        rows=json.loads((Path(__file__).parent/'fixtures'/'two_early_input_frames.json').read_text(encoding='utf-8'))
        run=[row for row in rows if row['label']=='Input']
        low,high,excluded=progress_interval(run,'Input')
        self.assertEqual([round(row['audio_seconds'],3) for row in excluded],[17.834,18.318])
        self.assertLessEqual(low,high+.25)
        with self.assertRaises(ValueError):progress_interval(run[:-5],'Input')
        # A third conflicting frame exceeds the bounded correction.
        import copy
        damaged=copy.deepcopy(run)
        damaged[28]['readings']['Input']['current']=15
        with self.assertRaises(ValueError):progress_interval(damaged,'Input')

    def test_false_midplay_zeros_require_full_clock_and_nearby_recovery(self):
        import copy
        all_rows=json.loads((Path(__file__).parent/'fixtures'/'midplay_false_zeros.json').read_text(encoding='utf-8'))
        for name in ('Input','B'):
            rows=[r for r in all_rows if r['label']==name]
            run,excluded=first_progress_run(rows,name)
            self.assertEqual(len(excluded),2)
            self.assertGreaterEqual(run[-1]['readings'][name]['current'],20)
            progress_interval(run,name)
            # A true restart keeps its new clock rather than immediately
            # recovering the old progress; sustained zeros also stay rejected.
            first=next(i for i in range(1,len(rows)) if rows[i]['readings'][name]['current']<rows[i-1]['readings'][name]['current'])
            for mode in ('restart','three_zeros','missing_end','low_consensus'):
                damaged=copy.deepcopy(rows)
                if mode=='restart':
                    for r in damaged[first:]:r['readings'][name]['current']=max(0,int(r['audio_seconds']-damaged[first]['audio_seconds']))
                elif mode=='three_zeros':damaged[first+2]['readings'][name]['current']=0
                elif mode=='missing_end':damaged=damaged[:-12]
                else:damaged=damaged[:first+4]
                short,excluded=first_progress_run(damaged,name)
                with self.subTest(name=name,mode=mode):
                    self.assertFalse(excluded)
                    self.assertEqual(len(short),first)

    def test_short_complete_recording_recovers_bounded_false_zeros(self):
        rows=json.loads((Path(__file__).parent/'fixtures'/'short_midplay_false_zeros.json').read_text(encoding='utf-8'))
        rows=[r for r in rows if r['label']=='Input']
        self.assertLess(len(rows),40)
        run,excluded=first_progress_run(rows,'Input')
        self.assertEqual(len(excluded),2)
        self.assertEqual(run[-1]['readings']['Input']['current'],14)
        progress_interval(run,'Input')

    def test_early_terminal_digit_needs_independent_end_and_later_clock(self):
        import copy
        rows=json.loads((Path(__file__).parent/'fixtures'/'early_terminal_input_frames.json').read_text(encoding='utf-8'))
        run=[row for row in rows if row['label']=='Input' and row['readings']['Input']['current'] is not None]
        with self.assertRaises(ValueError):progress_interval(run,'Input')
        low,high,excluded=progress_interval(run,'Input',end_confirmed=True)
        self.assertEqual([round(row['audio_seconds'],3) for row in excluded],[29.381,29.647])
        self.assertAlmostEqual(low,3.975)
        self.assertAlmostEqual(high,3.99)
        # No later corroborating samples, a short trace, sustained bad clock,
        # and a third conflicting terminal sample must remain unaccepted.
        variants=[run[:-3],run[-15:]]
        for index in (45,46):
            damaged=copy.deepcopy(run)
            damaged[index]['readings']['Input']['current']=26
            variants.append(damaged)
        for damaged in variants:
            with self.subTest(length=len(damaged)):
                with self.assertRaises(ValueError):
                    progress_interval(damaged,'Input',end_confirmed=True)


if __name__=='__main__': unittest.main()
