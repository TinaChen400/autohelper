import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import wave
import numpy as np

from audio_lab.segment_session import segment
from integrated_app import RecordedWorkflow
from playback import Reading
from remote_ratings import RatingController


class AudioCaptureTests(unittest.TestCase):
    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts')

    def test_all_zero_capture_reports_audio_failure_before_gap_segmentation(self):
        for value in (0,1):
            with self.subTest(value=value),self.folder() as directory:
                folder=Path(directory)
                (folder/'timeline.jsonl').write_text('',encoding='utf-8')
                (folder/'capture_status.json').write_text(json.dumps({'device':'test loopback'}),encoding='utf-8')
                with wave.open(str(folder/'whole.wav'),'wb') as out:
                    out.setnchannels(2);out.setsampwidth(2);out.setframerate(48000)
                    out.writeframes(np.full((48000,2),value,dtype='<i2').tobytes())
                expected='未采集到声音：test loopback' if value==0 else 'Input: 有效播放进度不足'
                with self.assertRaisesRegex(ValueError,expected):segment(folder)
                self.assertFalse((folder/'Input.wav').exists())

    def test_completed_startup_timer_requires_two_actual_zero_frames(self):
        for zeros in (True,False):
            with self.subTest(zeros=zeros),self.folder() as directory:
                folder=Path(directory);rate=16000
                (folder/'session.json').write_text(json.dumps({'player_order':['A','B']}),encoding='utf-8')
                samples=np.zeros(rate*9,dtype='<i2')
                for start,stop in ((1,3.9),(5,7.9)):
                    x=np.arange(round((stop-start)*rate))
                    samples[round(start*rate):round(start*rate)+len(x)]=(np.sin(x*.1)*10000).astype('<i2')
                with wave.open(str(folder/'whole.wav'),'wb') as out:
                    out.setnchannels(1);out.setsampwidth(2);out.setframerate(rate);out.writeframes(samples.tobytes())
                rows=[{'label':'A','audio_seconds':1.05,'readings':{'A':{'current':3,'total':3}}}]
                for name,start in (('A',1),('B',5)):
                    for current,offset in ((0,.1),(0,.3),(1,1.1),(2,2.1),(3,3.05)):
                        value=current if zeros or name=='B' or current else 1
                        rows.append({'label':name,'audio_seconds':start+offset,'readings':{name:{'current':value,'total':3}}})
                (folder/'timeline.jsonl').write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
                if zeros:
                    manifest=segment(folder)
                    self.assertEqual(manifest['estimates']['A']['excluded_stale_samples'][0]['current'],3)
                else:
                    with self.assertRaisesRegex(ValueError,'缺少起始计时证据'):segment(folder)

    def test_signal_warning_does_not_abort_quiet_opening_and_clears_on_audio(self):
        with self.folder() as directory:
            folder=Path(directory);path=folder/'capture_status.json'
            path.write_text(json.dumps({'device':'test','seconds':4,'nonzero_samples_total':0}),encoding='utf-8')
            flow=object.__new__(RecordedWorkflow)
            flow.audio_folder=folder;flow.status=Mock();flow.log=Mock()
            flow.check_capture_signal({'A':Reading('active',1,10)})
            flow.status.assert_not_called()
            flow.check_capture_signal({'A':Reading('active',3,10)})
            self.assertTrue(flow.capture_signal_warning)
            flow.status.assert_called_once()
            path.write_text(json.dumps({'nonzero_samples_total':50}),encoding='utf-8')
            flow.check_capture_signal({'A':Reading('active',4,10)})
            self.assertFalse(flow.capture_signal_warning)

    def test_short_autoplay_gap_requires_real_silence_and_complete_visual_clock(self):
        for gap_seconds,complete in ((.06,True),(.03,True),(0,True),(.06,False),(.05,'idle_end'),(.03,'idle_end')):
            with self.subTest(gap=gap_seconds,complete=complete),self.folder() as directory:
                folder=Path(directory);rate=16000
                (folder/'session.json').write_text(json.dumps({'player_order':['A','B']}),encoding='utf-8')
                samples=np.zeros(rate*8,dtype='<i2')
                for start,stop in ((1,4-gap_seconds),(4,6.9)):
                    x=np.arange(round((stop-start)*rate))
                    samples[round(start*rate):round(start*rate)+len(x)]=(np.sin(x*.1)*10000).astype('<i2')
                with wave.open(str(folder/'whole.wav'),'wb') as out:
                    out.setnchannels(1);out.setsampwidth(2);out.setframerate(rate);out.writeframes(samples.tobytes())
                rows=[]
                for name,start in (('A',1),('B',4)):
                    for current,offset in ((0,.1),(0,.6),(1,1.1),(2,2.1),(3,3.0)):
                        if name=='B' and not complete and current==3:continue
                        ended=complete=='idle_end' and name=='B' and current==3
                        rows.append({'label':'idle' if ended else name,'audio_seconds':start+offset,
                            'readings':{name:{'current':current,'total':3,'state':'idle' if ended else 'active'}}})
                (folder/'timeline.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
                if gap_seconds>=.04 and complete:
                    manifest=segment(folder)
                    self.assertEqual(manifest['estimates']['B']['boundary_minimum_ms'],40)
                    self.assertAlmostEqual(manifest['estimates']['B']['start'],4-gap_seconds/2,places=2)
                else:
                    with self.assertRaisesRegex(ValueError,'声音间隙'):segment(folder)

    def test_failed_analysis_does_not_scroll_or_click_empty_form(self):
        done=threading.Event();done.set()
        record=SimpleNamespace(data={'model':{'judgments':{'quality':{'judgment':'需复核'}},
                    'reason':'未采集到声音'},'auto_actions':[]})
        flow=SimpleNamespace(comparison_record=record,analysis_done=done,mouse=Mock(),log=Mock(),status=Mock())
        RatingController(flow).run()
        flow.mouse.scroll.assert_not_called()
        flow.mouse.click_box.assert_not_called()
        self.assertIn('未采集到声音',flow.remote_status)


if __name__=='__main__':unittest.main()
