import json
from pathlib import Path
import unittest
from playback import Reading, RoundTracker
from audio_lab.resilient_tracker import ResilientTracker

class TrackerTests(unittest.TestCase):
    def test_recorded_ocr_regression_at_handoff(self):
        path=Path(__file__).resolve().parent.parent/'artifacts'/'session_20261011_205604_769963'/'events.jsonl'
        if not path.exists(): self.skipTest('Local regression trace unavailable')
        samples=[r for r in map(json.loads,path.read_text(encoding='utf-8').splitlines()) if r['event']=='playback_sample']
        tracker=ResilientTracker()
        for row in samples:
            tracker.observe({n:Reading(**r) for n,r in row['readings'].items()},row['elapsed'])
        self.assertTrue(tracker.segments['A'].done)
        self.assertTrue(tracker.segments['B'].started)
        self.assertFalse(tracker.segments['B'].done)
        self.assertTrue(tracker.corrections)

    def test_genuine_early_successor_not_accepted(self):
        tracker=ResilientTracker()
        tracker.observe({'Input':Reading('active',1,19)},0)
        with self.assertRaises(RuntimeError):
            tracker.observe({'Input':Reading('idle',2,19),'A':Reading('active',0,31)},1)

if __name__=='__main__': unittest.main()
