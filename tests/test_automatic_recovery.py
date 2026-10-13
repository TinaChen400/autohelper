"""Bounded recovery with actual polling and no desktop input."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np

from human_mouse import Cancelled
from playback import Reading,PlaybackEvidenceError
from workflow import Workflow,Settings,LocationUnconfirmed,PlaybackDidNotStart
from integrated_app import RecordedWorkflow
from audio_lab.segment_session import CaptureUnavailable


class AutomaticRecoveryTests(unittest.TestCase):
    def flow(self):
        flow=object.__new__(Workflow)
        flow.automatic_recovery=True
        flow.settings=Settings()
        for name in ('log','status','prepare_first_round','play_first_round','validate_first_round',
                     'verify_recovery_identity','quiet_players','on_first_round_retry'):
            setattr(flow,name,Mock())
        flow.observe_players=Mock(return_value=(None,{}))
        flow.mouse=Mock()
        return flow

    def test_location_recovers_and_has_three_attempt_limit(self):
        flow=self.flow();flow.identify=Mock(side_effect=[LocationUnconfirmed('B'),(None,{})])
        flow.locate_first_round()
        self.assertEqual(flow.identify.call_count,2)
        flow.identify=Mock(side_effect=LocationUnconfirmed('B'))
        with self.assertRaises(LocationUnconfirmed):flow.locate_first_round()
        self.assertEqual(flow.identify.call_count,3)

    def test_location_does_not_retry_cancel_or_unrelated_error(self):
        for error in (Cancelled('Esc'),RuntimeError('OCR model missing')):
            flow=self.flow();flow.identify=Mock(side_effect=error)
            with self.assertRaises(type(error)):flow.locate_first_round()
            self.assertEqual(flow.identify.call_count,1)

    def test_partial_round_restarts_once_only_after_same_question_and_pause(self):
        flow=self.flow();events=[]
        flow.play_first_round.side_effect=[PlaybackEvidenceError('early pause'),None]
        flow.verify_recovery_identity.side_effect=lambda:events.append('verify')
        flow.quiet_players.side_effect=lambda:events.append('pause')
        flow.on_first_round_retry.side_effect=lambda *args:events.append('new capture')
        flow.confirmed_first_round({})
        self.assertEqual(events,['verify','pause','verify','new capture'])
        flow.prepare_first_round.assert_any_call({},force=True)
        self.assertEqual(flow.validate_first_round.call_count,1)

    def test_second_failed_round_never_loops(self):
        flow=self.flow();flow.play_first_round.side_effect=PlaybackEvidenceError('restart')
        with self.assertRaises(PlaybackEvidenceError):flow.confirmed_first_round({})
        self.assertEqual(flow.play_first_round.call_count,2)
        self.assertEqual(flow.on_first_round_retry.call_count,1)

    def test_changed_question_or_cancel_never_resets_or_restarts_recording(self):
        for error in (RuntimeError('new question'),Cancelled('Esc')):
            flow=self.flow();flow.play_first_round.side_effect=PlaybackEvidenceError('pause')
            flow.verify_recovery_identity.side_effect=error
            with self.assertRaises(type(error)):flow.confirmed_first_round({})
            flow.quiet_players.assert_not_called()
            flow.on_first_round_retry.assert_not_called()
        flow=self.flow();flow.play_first_round.side_effect=Cancelled('Esc')
        with self.assertRaises(Cancelled):flow.confirmed_first_round({})
        flow.verify_recovery_identity.assert_not_called()

    def test_failed_press_is_retried_but_delayed_active_play_is_never_toggled(self):
        for state,clicks in (('idle',2),('active',1)):
            flow=self.flow();flow.click=Mock()
            flow.await_chain=Mock(side_effect=[PlaybackDidNotStart('not started'),None])
            readings={n:Reading(state if n=='Input' else 'idle',0,5) for n in flow.settings.player_order}
            flow.observe_players.return_value=(None,readings)
            flow.start_and_wait('Input',flow.settings.player_order)
            self.assertEqual(flow.click.call_count,clicks)
            self.assertIsNone(flow.pending_play_start)

    def test_missing_start_has_three_press_limit_and_unknown_is_not_clicked(self):
        flow=self.flow();flow.click=Mock()
        flow.await_chain=Mock(side_effect=PlaybackDidNotStart('not started'))
        flow.observe_players.return_value=(None,{n:Reading('idle',0,5) for n in flow.settings.player_order})
        with self.assertRaises(PlaybackDidNotStart):flow.start_and_wait('Input',flow.settings.player_order)
        self.assertEqual(flow.click.call_count,3)
        flow.click.reset_mock()
        flow.observe_players.return_value=(None,{n:Reading('unknown') for n in flow.settings.player_order})
        with self.assertRaises(PlaybackEvidenceError):flow.start_and_wait('Input',flow.settings.player_order)
        self.assertEqual(flow.click.call_count,1)

    def test_real_polling_times_out_a_missed_press_without_false_completion(self):
        flow=self.flow();clock=[0.]
        image=np.zeros((40,40,3),np.uint8)
        flow.snapshot=Mock(return_value=image);flow.check_page=Mock();flow.preview=Mock()
        flow.targets={};flow.known_readings={'A':Reading('idle',0,2)}
        flow.detector=Mock()
        flow.pending_play_start='A'
        flow.mouse.wait_observation.side_effect=lambda seconds:clock.__setitem__(0,clock[0]+seconds)
        with patch('workflow.time.monotonic',side_effect=lambda:clock[0]),patch('workflow.read_players',return_value={'A':Reading('idle',0,2)}):
            with self.assertRaises(PlaybackDidNotStart):flow.await_chain(('A',))
        self.assertLess(clock[0],4.5)
        self.assertFalse(any(call.args[0]=='chain_completed' for call in flow.log.call_args_list))

    def test_waveform_change_with_same_text_is_rejected(self):
        flow=self.flow();del flow.verify_recovery_identity
        image=np.zeros((40,40,3),np.uint8)
        # stable_target uses dataclass.replace, as real targets do.
        from vision import Target
        target=Target('A',(1,1,5,5),(1,20,5,5),'idle',.99)
        flow.targets={'A':target};flow.settings.player_order=('A','B')
        flow.snapshot=Mock(return_value=image);flow.check_page=Mock()
        flow.detector=Mock();flow.detector.detect.return_value=({'A':target},[],0)
        flow.detector.last_task_lines=[]
        profile=np.linspace(0,1,256)
        flow.recovery_identity={'marker':(),'waveforms':{'A':profile}}
        with patch('audio_lab.task_context.waveform_profiles',return_value={'A':profile[::-1]}):
            with self.assertRaisesRegex(RuntimeError,'波形'):flow.verify_recovery_identity()
        with patch('audio_lab.task_context.waveform_profiles',return_value={'A':profile}):
            flow.verify_recovery_identity()

    def test_recording_integrity_is_checked_before_model_and_retry_uses_separate_pcm(self):
        with tempfile.TemporaryDirectory() as directory:
            flow=object.__new__(RecordedWorkflow)
            flow.audio_folder=Path(directory);flow.mouse=Mock();flow.stop_audio=Mock();flow.log=Mock()
            with patch('audio_lab.segment_session.segment',side_effect=ValueError('B tail')):
                with self.assertRaises(PlaybackEvidenceError):flow.validate_first_round()
            with patch('audio_lab.segment_session.segment',side_effect=CaptureUnavailable('silent')):
                with self.assertRaises(CaptureUnavailable):flow.validate_first_round()
            flow.comparison_record=Mock();flow.start_audio=Mock()
            originals={'whole.wav':b'original pcm','timeline.jsonl':b'old clock','STOP':b''}
            for name,data in originals.items():(flow.audio_folder/name).write_bytes(data)
            flow.on_first_round_retry(1,'B tail')
            archive=flow.audio_folder/'recovery_attempts'/'failed_1'
            for name,data in originals.items():
                self.assertEqual((archive/name).read_bytes(),data)
                self.assertFalse((flow.audio_folder/name).exists())
            self.assertEqual(json.loads((archive/'failure.json').read_text())['reason'],'B tail')
            flow.start_audio.assert_called_once()


if __name__=='__main__':unittest.main()
