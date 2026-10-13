import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave
from types import SimpleNamespace
import numpy as np

from audio_lab.durable_io import write_json
from health_monitor import interrupted_capture

ROOT=Path(__file__).resolve().parents[1]


class DurableCaptureTests(unittest.TestCase):
    def test_live_recorder_publishes_checkpoint_before_saved_status(self):
        from audio_lab import live_capture
        from audio_lab.durable_io import checkpoint
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            root=Path(directory);folder=root/'sessions'/'capture';folder.mkdir(parents=True)
            class Source:
                calls=0
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def record(self,**kwargs):
                    self.calls+=1
                    if self.calls==15:(folder/'STOP').touch()
                    return np.full((4800,2),0.1)
            source=Source();speaker=SimpleNamespace(id='test-output',name='Test output')
            with patch.object(live_capture,'ROOT',root), \
                 patch.object(sys,'argv',['live_capture','--folder',str(folder)]), \
                 patch.dict(sys.modules,{'soundcard':SimpleNamespace(default_speaker=lambda:speaker,
                     get_microphone=lambda **kw:SimpleNamespace(recorder=lambda **kw:source))}), \
                 patch.object(live_capture.time,'monotonic',side_effect=[n*.11 for n in range(100)]), \
                 patch.object(live_capture,'checkpoint',wraps=checkpoint) as sync:
                live_capture.main()
            status=json.loads((folder/'capture_status.json').read_text())
            self.assertEqual(status['state'],'saved')
            self.assertGreater(status['checkpoint_seconds'],0)
            self.assertEqual(status['seconds'],1.5)
            self.assertGreaterEqual(sync.call_count,2)
            with wave.open(str(folder/'whole.wav'),'rb') as audio:
                self.assertEqual(audio.getnframes(),72000)

    def test_failed_replace_keeps_last_complete_json(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            path=Path(directory)/'state.json'
            write_json(path,{'state':'old'})
            with patch('audio_lab.durable_io.os.replace',side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):write_json(path,{'state':'new'})
            self.assertEqual(json.loads(path.read_text()),{'state':'old'})
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_abrupt_process_exit_keeps_checkpointed_wav_readable(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            path=Path(directory)/'whole.wav'
            code="""import os, sys, wave
from audio_lab.durable_io import checkpoint
raw=open(sys.argv[1],'w+b',buffering=0)
out=wave.open(raw,'wb')
out.setparams((2,2,48000,0,'NONE','not compressed'))
out.writeframes(b'\\x01\\x00' * 96000)
checkpoint(raw)
os._exit(17)
"""
            result=subprocess.run([sys.executable,'-c',code,str(path)],cwd=ROOT)
            self.assertEqual(result.returncode,17)
            with wave.open(str(path),'rb') as audio:
                self.assertEqual(audio.getnframes(),48000)
                self.assertEqual(len(audio.readframes(48000)),192000)

    def test_startup_distinguishes_empty_partial_and_saved_audio(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as directory:
            folder=Path(directory)/'integrated_session_20261013_175133'
            folder.mkdir();path=folder/'whole.wav';path.touch()
            (folder/'capture_status.json').write_bytes(b'\0'*20)
            self.assertIn('为空或损坏',interrupted_capture(directory)['detail'])
            with wave.open(str(path),'wb') as audio:
                audio.setparams((1,2,8000,0,'NONE','not compressed'))
                audio.writeframes(b'\x01\x00'*8000)
            self.assertIn('部分录音',interrupted_capture(directory)['detail'])
            write_json(folder/'capture_status.json',{'state':'saved'})
            self.assertIsNone(interrupted_capture(directory))

if __name__=='__main__':unittest.main()
