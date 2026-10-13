import json
from pathlib import Path
import sys
import threading
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audio_lab import pipeline
from audio_lab import segment_session,decision_engine
from audio_lab.comparison_record import ComparisonRecord, DIMENSIONS


class PipelineIsolationTests(unittest.TestCase):
    def test_heavy_stages_overlap_and_owned_fields_merge_without_stale_votes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            (folder/'model_comparison.json').write_text(json.dumps({'transcripts':{'Input':'old task'}}))
            barrier=threading.Barrier(2)
            def run(command,**kwargs):
                stage=command[command.index('--stage')+1] if '--stage' in command else 'naturalness'
                if stage in ('asr','naturalness'):barrier.wait(timeout=2)
                if stage!='naturalness':
                    path=Path(command[command.index('--output')+1])
                    part={'transcripts':{'A':'current text'} if stage=='asr' else {'Input':'must not leak'},
                          'quality':{'A':{'MOS_OVRL':4}} if stage=='quality' else {},
                          'timings':{stage:1},'models':{stage:stage}}
                    path.write_text(json.dumps(part))
                return SimpleNamespace(returncode=0)
            manifest={'task_type':'ab_only','bounds_seconds':{}}
            with patch.object(sys,'argv',['pipeline.py',directory]), \
                    patch.object(segment_session,'segment',return_value=manifest), \
                    patch.object(pipeline.subprocess,'run',side_effect=run), \
                    patch.object(decision_engine,'build_decisions'):
                pipeline.main()
            status=json.loads((folder/'pipeline_status.json').read_text(encoding='utf-8'))
            merged=json.loads((folder/'model_comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(status['state'],'draft_ready')
            self.assertEqual(merged['transcripts'],{'A':'current text'})
            self.assertEqual(merged['quality'],{'A':{'MOS_OVRL':4}})
            self.assertFalse(merged['speaker_applicable'])
            self.assertEqual(set(merged['timings']),{'asr','quality'})

    def test_failed_asr_does_not_prevent_other_three_stages(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            calls=[]
            def run(command,**kwargs):
                calls.append(command)
                failed='asr' in command
                if failed: kwargs['stdout'].write('ImportError: 应用程序控制策略已阻止此文件。\n')
                return SimpleNamespace(returncode=int(failed))
            with patch.object(sys,'argv',['pipeline.py',directory]), \
                    patch.object(segment_session,'segment',return_value={}), \
                    patch.object(pipeline.subprocess,'run',side_effect=run):
                pipeline.main()
            status=json.loads((Path(directory)/'pipeline_status.json').read_text(encoding='utf-8'))
            self.assertEqual(len(calls),4)
            self.assertEqual(status['state'],'partial_ready')
            self.assertIn('Windows 应用控制',status['stage_errors']['asr'])
            self.assertIn('completed_at',status)

    def test_partial_results_preserve_successful_model_judgment(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            (folder/'model_comparison.json').write_text(json.dumps({'speaker_similarity':{'A':.8,'B':.4}}))
            record=ComparisonRecord(folder)
            record.save_model({'state':'partial_ready','stage_errors':{'asr':'转写依赖失败'}},record.started,record.started)
            record.save_human({key:{'judgment':'','reason':''} for key in DIMENSIONS},record.started,True)
            self.assertEqual(record.data['final_judgments']['speaker']['judgment'],'A')
            self.assertTrue(record.data['final_judgments']['speaker']['usable_for_learning'])
            self.assertEqual(record.data['model']['judgments']['pronunciation']['judgment'],'需复核')


if __name__=='__main__': unittest.main()
