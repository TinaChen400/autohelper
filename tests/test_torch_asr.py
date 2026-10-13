import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import wave
import numpy as np

from audio_lab.torch_asr import TorchTranscriber,read_wav,resample
from audio_lab.transcript_report import transcript_report


class TorchAsrTests(unittest.TestCase):
    def test_pcm_stereo_mix_preserves_scale_and_sample_rate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            path=Path(directory)/'A.wav'
            with wave.open(str(path),'wb') as out:
                out.setnchannels(2);out.setsampwidth(2);out.setframerate(48000)
                out.writeframes(np.array([[16384,16384],[-32768,32767],[0,0]],dtype='<i2').tobytes())
            audio,rate=read_wav(path)
            self.assertEqual(rate,48000)
            np.testing.assert_allclose(audio,[.5,-1/65536,0])
            self.assertEqual(audio.dtype,np.float32)

    def test_empty_audio_rejected_before_inference(self):
        with self.assertRaisesRegex(ValueError,'录音为空'):
            resample(np.array([],dtype=np.float32),48000)

    def test_cpu_checkpoint_uses_float32_instead_of_half_precision(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            path=Path(directory);(path/'model.safetensors').touch()
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False),float32='float32',
                                  float16='float16',set_num_threads=Mock(),inference_mode=Mock())
            model_class=Mock();processor_class=Mock()
            transformers=SimpleNamespace(WhisperProcessor=processor_class,WhisperForConditionalGeneration=model_class)
            with patch.dict('sys.modules',torch=torch,transformers=transformers):
                engine=TorchTranscriber(path)
            model_class.from_pretrained.return_value.to.assert_called_once_with(device='cpu',dtype='float32')
            self.assertEqual(engine.dtype,'float32')
            self.assertTrue(model_class.from_pretrained.call_args.kwargs['local_files_only'])

    def test_report_for_two_player_question_has_no_invented_input(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            report={'manifest':{'bounds_seconds':{'A':[0,5],'B':[5,36]}},
                    'transcription_backend':{'model':'distil-whisper/distil-small.en','device':'cpu'},
                    'transcripts':{'A':'First recording.','B':'Long recording tail.'},
                    'transcription_details':{'B':{'source_seconds':31.23}},'timings':{'asr_B':3.4}}
            (folder/'model_comparison.json').write_text(json.dumps(report),encoding='utf-8')
            text=transcript_report(folder)
            self.assertIn('Long recording tail.',text)
            self.assertIn('31.23',text)
            self.assertIn('3.40',text)
            self.assertNotIn('Input',text)

    def test_report_shows_asr_failure_with_other_evidence_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            (folder/'model_comparison.json').write_text('{}',encoding='utf-8')
            (folder/'pipeline_status.json').write_text(json.dumps({'stage_errors':{'asr':'模型未准备'}}),encoding='utf-8')
            self.assertIn('转写异常：模型未准备',transcript_report(folder))


if __name__=='__main__': unittest.main()
