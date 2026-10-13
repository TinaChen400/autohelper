import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
from integrated_app import IntegratedApp
from dynamic_app import DynamicApp
from integrated_app import RecordedWorkflow
from playback import Reading
import numpy as np


class IntegratedStartTests(unittest.TestCase):
    def setUp(self):
        # Startup unit tests must not load a real unfinished operator session.
        checkpoint=patch('question_progress.resumable_session',return_value=None)
        checkpoint.start();self.addCleanup(checkpoint.stop)

    def startup_flow(self,events):
        flow=object.__new__(RecordedWorkflow)
        flow.settings=SimpleNamespace(player_order=('Input','A','B'),short_enabled=False,scroll_after=False)
        flow.targets={};flow.known_readings={};flow.locked_start=False
        flow.finished=threading.Event();flow.round_captured=False;flow.audio_folder=None
        flow.last_image=None;flow.last_readings={}
        flow.prepare_audio_session=Mock();flow.stop_audio=Mock();flow.log=Mock();flow.status=Mock()
        image=np.zeros((40,40,3),np.uint8)
        flow.snapshot=Mock(return_value=image);flow.parked=Mock()
        readings={n:Reading('idle',0,1) for n in flow.settings.player_order}
        flow.identify=Mock(side_effect=lambda:(events.append('identify') or (image,readings)))
        flow.start_audio=Mock(side_effect=lambda:events.append('record'))
        flow.observe_players=Mock(side_effect=lambda:(events.append('refresh') or (image,readings)))
        flow.prepare_first_round=Mock()
        flow.play_first_round=Mock(side_effect=lambda rows:events.append('play'))
        flow.on_first_round_complete=Mock(side_effect=lambda:setattr(flow,'round_captured',True))
        flow.replay_rounds=Mock()
        return flow

    def test_failed_location_does_not_start_audio_or_playback(self):
        flow=self.startup_flow([])
        flow.identify.side_effect=RuntimeError('B not identified')
        with self.assertRaisesRegex(RuntimeError,'B not identified'):flow.run()
        flow.start_audio.assert_not_called()
        flow.play_first_round.assert_not_called()
        self.assertTrue(flow.finished.is_set())

    def test_capture_starts_after_location_and_before_first_play(self):
        events=[];flow=self.startup_flow(events)
        flow.run()
        self.assertEqual(events,['identify','record','refresh','play'])
        self.assertTrue(flow.finished.is_set())

    def test_recorder_setup_transition_uses_current_player_not_old_input(self):
        for old_name,new_name in (('Input','A'),('A','B')):
            with self.subTest(old=old_name,new=new_name):
                flow=self.startup_flow([])
                image=flow.snapshot.return_value
                old={n:Reading('active' if n==old_name else 'idle',3 if n==old_name else 5,7 if n=='Input' else 5)
                     for n in flow.settings.player_order}
                fresh={n:Reading('active' if n==new_name else 'idle',0 if n==new_name else 5,7 if n=='Input' else 5)
                       for n in flow.settings.player_order}
                flow.identify=Mock(return_value=(image,old))
                flow.observe_players=Mock(return_value=(image,fresh))
                flow.run()
                self.assertIs(flow.prepare_first_round.call_args.args[0],fresh)
                self.assertIs(flow.play_first_round.call_args.args[0],fresh)

    def test_playback_finishing_during_recorder_setup_uses_new_idle_state(self):
        flow=self.startup_flow([])
        image=flow.snapshot.return_value
        old={n:Reading('active' if n=='B' else 'idle',4,5) for n in flow.settings.player_order}
        fresh={n:Reading('idle',5,5) for n in flow.settings.player_order}
        flow.identify=Mock(return_value=(image,old))
        flow.observe_players=Mock(return_value=(image,fresh))
        flow.run()
        self.assertIs(flow.play_first_round.call_args.args[0],fresh)

    def test_window_refresh_excludes_helpers_and_keeps_selected_remote(self):
        app=object.__new__(DynamicApp)
        app.items=[]
        app.combo=Mock()
        app.combo.current.return_value=-1
        app.combo.__setitem__=Mock()
        windows=[(1,'向日葵助手 · 旧版'),(2,'向日葵远程控制'),(3,'KVMQ1'),(4,'KVMQ2')]
        with patch('dynamic_app.windows',return_value=windows):
            app.refresh()
            self.assertNotIn(1,[hwnd for hwnd,_ in app.items])
            app.combo.current.assert_called_with(1)
            app.combo.current.return_value=2
            app.refresh()
            app.combo.current.assert_called_with(2)

    def app(self):
        app=object.__new__(IntegratedApp)
        app.audio_workflow=None
        for name,value in [('pause_min','1'),('pause_max','3'),('scroll_max','12'),('after_b_pause','1')]:
            setattr(app,name,Mock(get=Mock(return_value=value)))
        app.scroll_after=Mock(get=Mock(return_value=True))
        app.short_enabled=Mock(get=Mock(return_value=True))
        for name,value in [('short_count','6'),('short_min','0.4'),('short_max','1.2')]:
            setattr(app,name,Mock(get=Mock(return_value=value)))
        for name in ('audio_status','human_status','status','result_context','task_type','detector'):
            setattr(app,name,Mock())
        app.rating_table=Mock(get_children=Mock(return_value=[]))
        app.human_fields={}
        app.select_window=Mock(return_value=123)
        app.activate=Mock();app.refresh_task_labels=Mock();app.start_human_timer=Mock();app.start_task=Mock()
        return app

    def test_unconfirmed_previous_feedback_does_not_block_new_question(self):
        app=self.app()
        done=threading.Event();done.set()
        record=Mock(data={'human':None})
        app.audio_workflow=SimpleNamespace(finished=done,round_captured=False,comparison_record=record)
        app.run_integrated()
        record.update.assert_called_once_with(review_state='not_confirmed')
        app.start_task.assert_called_once()
        self.assertIsNone(app.audio_workflow)

    def test_worker_reactivates_after_model_loading_before_start(self):
        app=self.app();app.run_integrated()
        event=Mock(wait=Mock(return_value=False))
        with patch('integrated_app.foreground',return_value=True),patch('integrated_app.RecordedWorkflow') as flow, \
                patch('continuous_runner.ContinuousRunner') as runner:
            app.start_task.call_args.args[0](event,Mock())
            runner.return_value.run.assert_called_once()
            runner.call_args.args[2]()
            flow.assert_called_once()
        self.assertEqual(app.activate.call_count,2)

    def test_failed_activation_never_starts_recorder(self):
        app=self.app();app.run_integrated()
        event=Mock(wait=Mock(return_value=False))
        with patch('integrated_app.foreground',return_value=False),patch('integrated_app.RecordedWorkflow') as flow, \
                patch('integrated_app.grab',return_value=(None,None)), \
                patch('continuous_runner.ContinuousRunner') as runner:
            app.start_task.call_args.args[0](event,Mock())
            self.assertTrue(runner.call_args.args[3]())
            flow.assert_not_called()


if __name__=='__main__': unittest.main()
