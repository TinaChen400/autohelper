import json
from pathlib import Path
import shutil
import tempfile
import unittest
from segment_session import segment


class SegmentTests(unittest.TestCase):
    def test_real_recording_and_missing_reference(self):
        source=Path(__file__).resolve().parent/'sessions'/'paired_20261011_203027_238752'
        if not source.exists(): self.skipTest('Local recording fixture unavailable')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for name in ('whole.wav','timeline.jsonl'): shutil.copy2(source/name,p/name)
            result=segment(p)
            self.assertAlmostEqual(result['bounds_seconds']['A'][0],19.33,places=2)
            self.assertAlmostEqual(result['bounds_seconds']['B'][0],24.15,places=2)
            self.assertFalse(result['complete_confirmed'])
            rows=[json.loads(s) for s in (p/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
            (p/'timeline.jsonl').write_text('\n'.join(json.dumps(r) for r in rows if r['label']!='Input'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Input'): segment(p)

if __name__=='__main__': unittest.main()
