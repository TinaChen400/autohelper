from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from audio_lab.performance import write_performance


class PerformanceTests(unittest.TestCase):
    def test_phase_times_keep_parallel_overlap_separate_and_preserve_resume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            rows=[{'time':f'2026-10-12T11:00:{second:02d}','event':event} for second,event in
                  ((0,'session_start'),(10,'play_click'),(24,'chain_completed'),(26,'replay_plan'),(45,'rating_scroll_start'))]
            (folder/'events.jsonl').write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
            (folder/'pipeline_status.json').write_text(json.dumps({'elapsed_seconds':18,
                'completed_at':'2026-10-12T11:00:44+01:00','stage_timings':{'asr':12,'naturalness':18}}))
            report=write_performance(folder,folder,datetime(2026,10,12,11,0,50))
            self.assertEqual(report['total_until_rating_check_seconds'],50)
            self.assertEqual(report['startup_location_context_seconds'],10)
            self.assertEqual(report['first_complete_playback_seconds'],14)
            self.assertEqual(report['model_ready_to_rating_check_seconds'],6)
            self.assertEqual(report['rating_location_fill_check_seconds'],5)
            self.assertFalse(report['submitted_by_assistant'])
            (folder/'events.jsonl').write_text(json.dumps(rows[0]))
            self.assertEqual(write_performance(folder,folder),report)
