import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from integrated_app import RecordedWorkflow
from workflow import Workflow, Settings, ReplayFinished
from audio_lab.comparison_record import ComparisonRecord, DIMENSIONS


class IntegratedComparisonTests(unittest.TestCase):
    def flow(self):
        flow = object.__new__(RecordedWorkflow)
        flow.settings = Settings(1, 2, pause_min=0, pause_max=0, after_b_pause=0)
        flow.rng = Mock()
        flow.rng.randint.return_value = 2
        flow.rng.choice.return_value = 'B'
        flow.rng.uniform.return_value = 0
        flow.analysis_done = threading.Event()
        flow.log = Mock()
        flow.status = Mock()
        flow.mouse = Mock()
        flow.click = Mock()
        flow.settle_and_follow = Mock()
        return flow

    def test_ready_before_replay_causes_no_clicks(self):
        flow = self.flow()
        flow.analysis_done.set()
        flow.replay_rounds()
        flow.click.assert_not_called()

    def test_ready_during_replay_finishes_current_but_cancels_next(self):
        flow = self.flow()
        flow.settle_and_follow.side_effect = lambda name: flow.analysis_done.set()
        flow.replay_rounds()
        self.assertEqual(flow.click.call_count, 1)
        self.assertEqual(flow.settle_and_follow.call_count, 1)
        self.assertFalse(flow.supplemental)

    def test_slow_model_never_exceeds_two_supplements(self):
        flow = self.flow()
        flow.replay_rounds()
        self.assertEqual(flow.click.call_count, 2)

    def test_ready_during_wait_skips_press(self):
        flow = self.flow()
        flow.settings.after_b_pause = 1
        flow.mouse.wait.side_effect = lambda seconds: flow.analysis_done.set()
        flow.replay_rounds()
        flow.click.assert_not_called()

    def test_prepress_guard_does_not_interrupt_first_round(self):
        flow = self.flow()
        flow.analysis_done.set()
        flow.supplemental = True
        with self.assertRaises(ReplayFinished):
            Workflow._click_once(flow, 'Input')

    def test_human_and_model_are_kept_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'model_comparison.json').write_text(json.dumps({'speaker_similarity': {'A': .8, 'B': .4}}))
            (folder/'naturalness.json').write_text(json.dumps({'scores': {'A': {'mos': 2}, 'B': {'mos': 3}}}))
            record = ComparisonRecord(folder)
            judgments = {key: {'judgment': 'A', 'reason': '人工听感依据'} for key in DIMENSIONS}
            record.save_human(judgments, record.started, False)
            record.save_model({'state': 'draft_ready'}, record.started, record.started)
            data = json.loads((folder/'comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(data['human']['judgments']['naturalness']['judgment'], 'A')
            self.assertEqual(data['model']['judgments']['naturalness']['judgment'], 'B')
            self.assertEqual(data['model']['judgments']['quality']['judgment'], '需复核')
            self.assertFalse(data['human']['model_visible_before_save'])
            self.assertGreaterEqual(data['human']['elapsed_seconds'], 0)
            self.assertIn('first_round_to_model_seconds', data['timings'])

    def test_optional_reasons_confirmation_corrections_and_history(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            (folder/'model_comparison.json').write_text(json.dumps({'speaker_similarity': {'A': .8, 'B': .4}}))
            record=ComparisonRecord(folder,folder/'feedback_library.jsonl')
            record.save_model({'state':'draft_ready'},record.started,record.started)
            answers={key:{'judgment':'','reason':''} for key in DIMENSIONS}
            record.save_human(answers,record.started,True)
            self.assertEqual(record.data['human']['judgments']['speaker']['source'],'model_confirmation')
            self.assertFalse(record.data['final_judgments']['quality']['usable_for_learning'])
            answers['speaker']={'judgment':'B','reason':'B 更接近参考的低沉音色'}
            record.save_human(answers,record.started,True)
            self.assertEqual(record.data['final_judgments']['speaker']['judgment'],'B')
            self.assertEqual(len(record.data['feedback_history']),2)
            self.assertEqual(record.data['feedback_history'][0]['judgments']['speaker']['judgment'],'A')
            self.assertEqual(record.data['model']['judgments']['speaker']['judgment'],'A')
            self.assertEqual(record.data['rule_candidates'][0]['dimension'],'speaker')
            self.assertEqual(record.data['rule_candidates'][0]['status'],'待跨题验证')
            library=[json.loads(line) for line in (folder/'feedback_library.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(library),2)
            self.assertEqual(library[-1]['final_judgments']['speaker']['judgment'],'B')

    def test_correction_does_not_require_reason(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            record=ComparisonRecord(directory)
            answers={key:{'judgment':'B','reason':''} for key in DIMENSIONS}
            record.save_human(answers,record.started,True)
            self.assertEqual(record.data['final_judgments']['speaker']['judgment'],'B')

    def test_failed_analysis_is_not_confirmed_as_a_label(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            (folder/'model_comparison.json').write_text(json.dumps({'speaker_similarity':{'A':.8,'B':.4}}))
            record=ComparisonRecord(folder)
            record.save_model({'state':'needs_review','reason':'自然度分析失败'},record.started,record.started)
            record.save_human({key:{'judgment':'','reason':''} for key in DIMENSIONS},record.started,True)
            self.assertEqual(record.data['model']['judgments']['speaker']['judgment'],'需复核')
            self.assertFalse(record.data['final_judgments']['speaker']['usable_for_learning'])


if __name__ == '__main__':
    unittest.main()
