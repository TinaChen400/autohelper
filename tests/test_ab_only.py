import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave
import numpy as np

from workflow import Workflow,Settings
from playback import Reading
from audio_lab.segment_session import segment
from audio_lab.comparison_record import ComparisonRecord,DIMENSIONS


class ABOnlyTests(unittest.TestCase):
    def test_startup_playback_is_paused_and_reset_before_full_fresh_round(self):
        for order in (('Input','A','B'),('A','B')):
            for active_name in order:
                for current in (0,3):
                    with self.subTest(order=order,active=active_name,current=current):
                        flow=object.__new__(Workflow)
                        flow.settings=Settings(player_order=order)
                        events=[]
                        flow.status=Mock();flow.log=Mock()
                        flow.quiet_players=Mock(side_effect=lambda:events.append('pause'))
                        flow.prepare_first_round=Mock(side_effect=lambda rows:events.append('reset'))
                        flow.click=Mock(side_effect=lambda name,**kw:events.append('click '+name))
                        flow.await_chain=Mock(side_effect=lambda names,*args:events.append('await '+','.join(names)))
                        idle={n:Reading('idle',0,5) for n in order}
                        flow.observe_players=Mock(return_value=(None,idle))
                        old={n:Reading('active' if n==active_name else 'idle',current if n==active_name else 5,5) for n in order}
                        flow.play_first_round(old)
                        self.assertEqual(events[:3],['pause','reset','click '+order[0]])
                        self.assertEqual(events[3],'await '+(','.join(order) if len(order)==3 else 'A'))
                        flow.prepare_first_round.assert_called_once_with(idle)

    def test_unconfirmed_startup_pause_never_clicks_a_new_round(self):
        flow=object.__new__(Workflow)
        flow.settings=Settings();flow.status=Mock();flow.log=Mock()
        flow.quiet_players=Mock();flow.click=Mock()
        active={n:Reading('active' if n=='A' else 'idle',0,5) for n in flow.settings.player_order}
        flow.observe_players=Mock(return_value=(None,active))
        with self.assertRaisesRegex(RuntimeError,'未确认原有播放已暂停'):
            flow.play_first_round(active)
        flow.click.assert_not_called()

    def test_partial_paused_b_resets_and_confirms_zero_before_full_round(self):
        for order in (('A','B'),('Input','A','B')):
            for paused in order:
                with self.subTest(order=order,paused=paused):
                    flow=object.__new__(Workflow)
                    flow.settings=Settings(player_order=order)
                    flow.targets={name:object() for name in order};flow.snapshot=Mock(return_value=np.zeros((100,100,3),np.uint8))
                    flow.check_page=Mock();flow.preview_control=Mock();flow.parked=Mock();flow.mouse=Mock();flow.log=Mock()
                    fresh={name:Reading('idle',0 if name==paused else 12,12) for name in order}
                    flow.observe_players=Mock(return_value=(None,fresh))
                    readings={name:Reading('idle',8 if name==paused else 12,12) for name in order}
                    with patch('short_listen.reset_control',return_value=(20,20,10,10)):
                        flow.prepare_first_round(readings)
                    flow.preview_control.assert_called_once_with(paused,(20,20,10,10),'idle',reset=True)
                    self.assertEqual(flow.observe_players.call_count,2)
                    self.assertEqual(readings[paused].current,0)

    def test_ab_first_round_clicks_b_if_not_autoplaying(self):
        flow=object.__new__(Workflow)
        flow.settings=Settings(player_order=('A','B'))
        flow.click=Mock(); flow.await_chain=Mock()
        idle={'A':Reading('idle',2,2),'B':Reading('idle',0,2)}
        flow.observe_players=Mock(return_value=(None,idle))
        flow.play_first_round(idle)
        self.assertEqual([c.args[0] for c in flow.click.call_args_list],['A','B'])
        self.assertEqual([c.args[0] for c in flow.await_chain.call_args_list],[('A',),('B',)])

    def test_autoplay_b_does_not_get_clicked_again(self):
        flow=object.__new__(Workflow)
        flow.settings=Settings(player_order=('A','B'))
        flow.click=Mock(); flow.await_chain=Mock()
        idle={'A':Reading('idle',0,2),'B':Reading('idle',0,2)}
        flow.observe_players=Mock(return_value=(None,{'A':Reading('idle',2,2),'B':Reading('active',0,2)}))
        flow.play_first_round(idle)
        self.assertEqual([c.args[0] for c in flow.click.call_args_list],['A'])
        self.assertEqual(flow.await_chain.call_args_list[-1].args[0],('B',))

    def test_two_clip_segmentation_and_no_reference_label(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts') as directory:
            folder=Path(directory)
            (folder/'session.json').write_text(json.dumps({'player_order':['A','B']}))
            rate=16000
            audio=np.zeros(rate*7,dtype=np.int16)
            for begin,end in ((1,2.9),(4,5.9)):
                x=np.arange(round((end-begin)*rate))
                audio[round(begin*rate):round(begin*rate)+len(x)]=(np.sin(x*.1)*10000).astype(np.int16)
            with wave.open(str(folder/'whole.wav'),'wb') as out:
                out.setnchannels(1);out.setsampwidth(2);out.setframerate(rate);out.writeframes(audio.tobytes())
            rows=[]
            for name,start in (('A',1),('B',4)):
                for current,offset in ((0,.1),(1,1.1),(2,2.05)):
                    rows.append({'label':name,'audio_seconds':start+offset,
                                 'readings':{name:{'current':current,'total':2}}})
            (folder/'timeline.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            manifest=segment(folder)
            self.assertEqual(set(manifest['bounds_seconds']),{'A','B'})
            self.assertEqual(manifest['task_type'],'ab_only')
            self.assertFalse((folder/'Input.wav').exists())
            self.assertTrue((folder/'B.wav').exists())
            record=ComparisonRecord(folder)
            record.save_model({'state':'partial_ready','stage_errors':{'asr':'转写失败'}},record.started,record.started)
            record.save_human({key:{'judgment':'A','reason':''} for key in DIMENSIONS},record.started,True)
            self.assertEqual(record.data['model']['judgments']['speaker']['judgment'],'不适用')
            self.assertEqual(record.data['final_judgments']['speaker']['judgment'],'不适用')
            self.assertFalse(record.data['final_judgments']['speaker']['usable_for_learning'])


if __name__=='__main__': unittest.main()
