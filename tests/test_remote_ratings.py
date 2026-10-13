import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import cv2
import numpy as np

from audio_lab.comparison_record import ComparisonRecord
from rating_reader import dimensions_from_lines,selected_state,read_rating_screen,submit_enabled
from remote_ratings import SelectionTracker,choose_action,RatingController,RemoteRatingObserver,same_layout,retry_candidates,rating_pause,RatingRecheck
from workflow import Workflow,Settings,ControlChanged,ReplayFinished
from playback import Reading


def rating_row(key='naturalness',selected=None,state=None):
    return {'key':key,'name':key,'label_box':(10,50,60,15),
            'options':{'Response A':(150,50,70,15),'Response B':(250,50,70,15)},
            'selected':selected,'selection_state':state or ('selected' if selected else 'none')}


class RemoteRatingTests(unittest.TestCase):
    def test_long_audio_allows_two_replays_but_result_still_cancels_them(self):
        from integrated_app import RecordedWorkflow
        flow=object.__new__(RecordedWorkflow)
        flow.settings=Settings(replay_min=1,replay_max=2)
        flow.known_readings={'A':Reading('idle',20,20),'B':Reading('idle',18,18)}
        flow.analysis_done=threading.Event()
        self.assertEqual(flow.replay_count(),2)
        self.assertFalse(flow.stop_replays())
        flow.analysis_done.set()
        self.assertTrue(flow.stop_replays())

    def test_paced_dispatch_preserves_human_selection_and_batched_verification(self):
        with self.folder() as folder:
            record=self.record(folder)
            for key,value in (('quality','A'),('pronunciation','B'),('naturalness','A')):
                record.data['model']['judgments'][key]['judgment']=value
            image=np.full((300,400,3),255,np.uint8)
            rows=[rating_row('quality'),rating_row('pronunciation'),rating_row('naturalness','Response B')]
            for index,row in enumerate(rows):
                row['label_box']=(10,50+index*60,60,15)
                row['options']={name:(box[0],50+index*60,box[2],box[3]) for name,box in row['options'].items()}
            clicked=[];order=[]
            done=threading.Event();done.set()
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),directory=Path(folder),
                analysis_done=done,event=threading.Event(),settings=SimpleNamespace(max_scrolls=1),
                status=Mock(),mouse=Mock(),snapshot=Mock(return_value=image),detector=object(),send=Mock(),log=Mock())
            def refresh(im,current):
                result=[]
                for row in current:
                    option='Response B' if row['key']=='naturalness' else dict(clicked).get(row['key'])
                    result.append({**row,'selected':option,'selection_state':'selected' if option else 'none'})
                return result
            def click(box,shape,before_press=None,quick=False):
                before_press()
                key=next(row['key'] for row in rows if box[1]==row['label_box'][1])
                name=next(name for row in rows if row['key']==key for name,b in row['options'].items() if b==box)
                order.append(key);clicked.append((key,name))
            flow.mouse.click_box.side_effect=click
            flow.mouse.wait.side_effect=lambda seconds:order.append('wait')
            with patch('remote_ratings.read_rating_screen',return_value={'rows':rows,'submit_boxes':[(90,250,80,20)]}), \
                    patch('remote_ratings.refresh_selections',side_effect=refresh),patch('remote_ratings.RemoteRatingObserver'):
                RatingController(flow).run()
            self.assertLess(order.index('quality'),order.index('pronunciation'))
            self.assertIn('wait',order[order.index('quality')+1:order.index('pronunciation')])
            self.assertTrue(all(not c.kwargs['quick'] for c in flow.mouse.click_box.call_args_list))
            self.assertEqual(flow.mouse.click_box.call_count,2)
            self.assertEqual(record.data['remote_ratings']['quality']['source'],'model_auto')
            self.assertEqual(record.data['remote_ratings']['pronunciation']['source'],'model_auto')
            self.assertEqual(record.data['remote_ratings']['naturalness']['source'],'human_remote')
            self.assertTrue(all(a['state']=='verified' for a in record.data['auto_actions'] if a['state']!='planned'))

    def test_retry_needs_two_empty_frames_and_preserves_ambiguous_or_manual(self):
        row=rating_row();judgments={'naturalness':{'judgment':'A'}}
        self.assertEqual(retry_candidates([row],[row],judgments,{'naturalness':1},set()),[row])
        for first,second,counts,manual in ((row,rating_row(state='ambiguous'),{},set()),
                (rating_row(selected='Response B'),row,{},set()),
                (row,row,{'naturalness':3},set()),(row,row,{}, {'naturalness'})):
            self.assertEqual(retry_candidates([first],[second],judgments,counts,manual),[])

    def test_reading_pauses_vary_and_stay_bounded_for_close_scores(self):
        import random
        rng=random.Random(42)
        simple=[rating_pause(rating_row(),{},rng) for _ in range(50)]
        close=[rating_pause(rating_row(),{'naturalness':{'evidence':{'A':3.1,'B':3.11}}},rng) for _ in range(50)]
        self.assertTrue(all(.2<=s<=.5 for s in simple))
        self.assertTrue(all(.45<=s<=.85 for s in close))
        self.assertGreater(len(set(simple)),40)

    def test_failed_click_is_retried_and_verified_without_becoming_human_label(self):
        for succeeds,interference in ((True,False),(False,False),(False,True)):
            with self.subTest(succeeds=succeeds,interference=interference),self.folder() as folder:
                record=self.record(folder)
                record.data['model']['judgments']={'naturalness':{'judgment':'A'}}
                record.save_auto_action('naturalness','A','planned','first.png')
                im=np.full((200,400,3),255,np.uint8);clicked=[]
                mouse=SimpleNamespace(interference_count=0,move=Mock(),wait=Mock(),click_box=Mock())
                flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),mouse=mouse,
                    snapshot=Mock(return_value=im),log=Mock(),detector=object())
                controller=RatingController(flow);controller.attempt_counts['naturalness']=1
                if interference:mouse.interference_count=1
                def refresh(image,rows):
                    return [{**r,'selected':'Response A' if clicked and succeeds else None,
                             'selection_state':'selected' if clicked and succeeds else 'none'} for r in rows]
                def click(box,shape,before_press=None,quick=False):
                    before_press();clicked.append(True)
                mouse.click_box.side_effect=click
                with patch('remote_ratings.refresh_selections',side_effect=refresh):
                    controller.recover_rows(im,{'rows':[rating_row()],'submit_boxes':[]},record.data['model']['judgments'])
                # Pointer movement alone is not a human answer. After two
                # stable empty frames the missing click can still be retried.
                self.assertEqual(len(clicked),1 if succeeds else 2)
                self.assertIsNone(record.data.get('human'))
                if succeeds:
                    self.assertEqual(record.data['remote_ratings']['naturalness']['source'],'model_auto')
                    self.assertEqual(record.data['auto_actions'][-1]['state'],'verified')

    def test_human_selection_before_batch_press_cancels_without_overwriting(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.data['model']['judgments']['naturalness']['judgment']='A'
            im=np.full((200,400,3),255,np.uint8)
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),mouse=Mock(),
                                 snapshot=Mock(return_value=im),log=Mock())
            controller=RatingController(flow)
            flow.mouse.click_box.side_effect=lambda box,shape,before_press=None,quick=False:before_press()
            with patch('remote_ratings.refresh_selections',return_value=[rating_row(selected='Response B')]):
                with self.assertRaisesRegex(RatingRecheck,'取消本项代选'):
                    controller.apply_row(im,rating_row(),record.data['model']['judgments'],defer_verification=True)
            self.assertFalse(any(a['state']=='verified' for a in record.data['auto_actions']))
            self.assertFalse(controller.expected)

    def test_human_choice_during_first_press_does_not_abort_remaining_rows(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.data['model']['judgments']['quality']['judgment']='A'
            record.data['model']['judgments']['naturalness']['judgment']='A'
            image=np.full((300,400,3),255,np.uint8)
            rows=[rating_row('quality'),rating_row('naturalness')]
            rows[1]={**rows[1],'label_box':(10,110,60,15),
                     'options':{'Response A':(150,110,70,15),'Response B':(250,110,70,15)}}
            chosen={};presses=[];done=threading.Event();done.set()
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),directory=Path(folder),
                analysis_done=done,event=threading.Event(),settings=SimpleNamespace(max_scrolls=1),
                status=Mock(),mouse=Mock(),snapshot=Mock(return_value=image),detector=object(),send=Mock(),log=Mock())
            def refresh(im,current):
                return [{**r,'selected':chosen.get(r['key']),
                         'selection_state':'selected' if r['key'] in chosen else 'none'} for r in current]
            def click(box,shape,before_press=None,quick=False):
                key='quality' if box[1]==50 else 'naturalness'
                if key=='quality':chosen[key]='Response B'
                before_press()
                presses.append(key);chosen[key]='Response A'
            flow.mouse.click_box.side_effect=click
            with patch('remote_ratings.read_rating_screen',return_value={'rows':rows,'submit_boxes':[(90,250,80,20)]}),\
                 patch('remote_ratings.refresh_selections',side_effect=refresh),patch('remote_ratings.RemoteRatingObserver'):
                RatingController(flow).run()
            self.assertEqual(presses,['naturalness'])
            self.assertEqual(record.data['remote_ratings']['quality']['judgment'],'B')
            self.assertEqual(record.data['remote_ratings']['quality']['source'],'human_remote')
            self.assertEqual(record.data['remote_ratings']['naturalness']['source'],'model_auto')

    def test_manual_clear_remains_protected_after_stable_recheck(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.data['model']['judgments']['naturalness']['judgment']='A'
            record.save_auto_action('naturalness','A','verified','old.png')
            record.save_remote({'naturalness':{'judgment':'A','source':'model_auto'}},'selected.png')
            image=np.full((200,400,3),255,np.uint8)
            mouse=SimpleNamespace(interference_count=0,move=Mock(),wait=Mock(),click_box=Mock())
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),mouse=mouse,
                snapshot=Mock(return_value=image),log=Mock(),detector=object())
            controller=RatingController(flow);mouse.interference_count=1
            controller.store(image,[rating_row()]);controller.store(image,[rating_row()])
            with patch('remote_ratings.refresh_selections',side_effect=lambda im,rows:rows):
                controller.recover_rows(image,{'rows':[rating_row()],'submit_boxes':[]},record.data['model']['judgments'])
            mouse.click_box.assert_not_called()
            self.assertEqual(record.data['remote_ratings']['naturalness']['source'],'human_remote')
            self.assertEqual(record.data['remote_ratings']['naturalness']['judgment'],'需复核')

    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'artifacts')

    def record(self,folder):
        folder=Path(folder)
        (folder/'naturalness.json').write_text(json.dumps({'scores':{'A':{'mos':3},'B':{'mos':2}}}))
        record=ComparisonRecord(folder)
        record.save_model({'state':'draft_ready'},record.started,record.started)
        return record

    def test_two_option_rows_are_detected_and_quick_replay_is_excluded(self):
        def line(text,x,y,width=100):
            return ([[x,y],[x+width,y],[x+width,y+15],[x,y+15]],text,.99)
        rows=dimensions_from_lines([line('Response A',100,10),line('Response B',230,10),
            line('Naturalness *',10,80),line('Response A',200,80),line('Response B',320,80),
            line('Overall Preference *',10,130),line('Response A',200,130),line('Response B',320,130)],(200,500,3))
        self.assertEqual([row['key'] for row in rows],['naturalness','overall'])

    def test_ab_short_previews_never_request_missing_input(self):
        flow=object.__new__(Workflow)
        flow.settings=Settings(player_order=('A','B'),short_count=3)
        flow.supplemental=False
        flow.log=Mock();flow.quiet_players=Mock();flow.status=Mock();flow.mouse=Mock()
        flow.snapshot=Mock(return_value=np.zeros((200,400,3),np.uint8));flow.check_page=Mock()
        flow.targets={key:SimpleNamespace(button=(1,1,20,20)) for key in ('A','B')}
        flow.known_readings={key:Reading('idle',5,5) for key in ('A','B')}
        flow.rng=Mock();flow.rng.choice.side_effect=lambda choices:choices[0]
        flow.rng.uniform.return_value=.5
        flow.preview_control=Mock(side_effect=ControlChanged('test recheck'))
        with patch('short_listen.reset_control',return_value=(1,1,10,10)):
            flow.opening_previews()
        self.assertEqual([call.args[0] for call in flow.preview_control.call_args_list],['A','B','A'])
        self.assertFalse(flow.supplemental)

    def test_result_ready_prevents_new_short_reset_or_play(self):
        flow=object.__new__(Workflow)
        flow.stop_replays=Mock(return_value=True);flow.supplemental=True
        flow._opening_previews=Mock()
        flow.opening_previews()
        flow._opening_previews.assert_not_called()
        with self.assertRaises(ReplayFinished): flow.preview_control('A',(1,1,10,10),'idle',reset=True)

    def test_result_ready_pauses_only_supplemental_audio(self):
        flow=object.__new__(Workflow)
        flow.known_readings={'A':Reading('active',1,8)}
        flow.stop_replays=Mock(return_value=True)
        flow.quiet_players=Mock();flow.status=Mock();flow.log=Mock();flow.snapshot=Mock()
        flow.supplemental=True
        flow.await_chain(('A',))
        flow.quiet_players.assert_called_once()
        flow.snapshot.assert_not_called()
        flow.supplemental=False
        flow.quiet_players.reset_mock()
        flow.snapshot.side_effect=RuntimeError('first round continues')
        with self.assertRaisesRegex(RuntimeError,'first round continues'): flow.await_chain(('A',))
        flow.quiet_players.assert_not_called()

    def test_pause_reads_clear_timer_once_before_moving_to_button(self):
        image=np.full((200,400,3),255,np.uint8)
        flow=SimpleNamespace(snapshot=Mock(return_value=image),check_page=Mock(),
            targets={'A':SimpleNamespace(button=(20,50,30,30))},detector=object(),
            mouse=Mock(),supplemental=True,stop_replays=Mock(return_value=True))
        order=[]
        def reading(*args):
            order.append('read_clear_timer');return {'A':Reading('active',1,8)}
        def click(box,shape,before_press=None,quick=False):
            order.append('move_to_button');before_press();order.append('press')
        flow.mouse.click_box.side_effect=click
        with patch('workflow.glyph_state',return_value='active'),patch('workflow.read_players',side_effect=reading) as read:
            Workflow.preview_control(flow,'A',(20,50,30,30),'active')
        self.assertEqual(order,['read_clear_timer','move_to_button','press'])
        self.assertEqual(read.call_count,1)

    def test_result_ready_does_not_toggle_clip_about_to_end(self):
        image=np.full((200,400,3),255,np.uint8)
        flow=SimpleNamespace(snapshot=Mock(return_value=image),check_page=Mock(),
            targets={'A':SimpleNamespace(button=(20,50,30,30))},detector=object(),
            mouse=Mock(),supplemental=True,stop_replays=Mock(return_value=True))
        with patch('workflow.glyph_state',return_value='active'),patch('workflow.read_players',return_value={'A':Reading('active',7,8)}):
            with self.assertRaises(ControlChanged):
                Workflow.preview_control(flow,'A',(20,50,30,30),'active')
        flow.mouse.click_box.assert_not_called()

    def test_blue_orange_text_and_outlines_are_not_selections(self):
        image=np.full((130,380,3),255,np.uint8)
        options={'Response A':(30,48,85,17),'Response B':(210,48,85,17)}
        for name,color in [('Response A',(180,80,0)),('Response B',(0,130,180))]:
            x,y,w,h=options[name]
            cv2.rectangle(image,(x-7,y-7),(x+w+7,y+h+7),color,1)
            cv2.putText(image,name,(x,y+13),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        self.assertEqual(selected_state(image,options)[:2],(None,'none'))
        cv2.rectangle(image,(23,41),(122,72),(180,80,0),-1)
        cv2.putText(image,'Response A',(30,61),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
        self.assertEqual(selected_state(image,options)[:2],('Response A','selected'))
        cv2.rectangle(image,(23,41),(122,72),(255,246,239),-1)
        self.assertEqual(selected_state(image,options)[1],'ambiguous')

    def test_ambiguous_or_preselected_row_is_never_overwritten(self):
        judgments={'naturalness':{'judgment':'B'}}
        self.assertIsNone(choose_action(rating_row(selected='Response A'),judgments,set()))
        self.assertIsNone(choose_action(rating_row(state='ambiguous'),judgments,set()))
        self.assertIsNone(choose_action(rating_row(),{'naturalness':{'judgment':'需复核'}},set()))
        self.assertEqual(choose_action(rating_row(),judgments,set()),('naturalness','B','Response B'))

    def test_actual_remote_blue_and_orange_selected_styles(self):
        # Coordinates describe saved screenshots only; production locates rows by OCR.
        options={
            'speaker':{'N/A':(849,350,15,10),'Response A':(884,350,44,10),
                'Response B':(947,350,44,10),'Both Good':(1006,350,40,10),'Both Bad':(1060,350,35,10)},
            'naturalness':{'Response A':(986,445,44,10),'Response B':(1049,445,44,10)}}
        tracker=SelectionTracker({'speaker':'B','naturalness':'A'})
        for index,name in enumerate(('live_rating_selected.png','live_rating_selected_second.png')):
            image=cv2.imread(str(Path(__file__).parent/'fixtures'/name))
            self.assertIsNotNone(image)
            rows=[]
            for key,buttons in options.items():
                selected,state,_=selected_state(image,buttons)
                rows.append({'key':key,'selected':selected,'selection_state':state})
            observed=tracker.observe(rows)
            if index==0: self.assertEqual(observed,{})
            else: self.assertEqual(observed,{
                'speaker':{'judgment':'B','source':'model_auto'},
                'naturalness':{'judgment':'A','source':'model_auto'}})

    def test_actual_remote_neutral_gray_good_and_bad_selections(self):
        image=cv2.imread(str(Path(__file__).parent/'fixtures'/'grey_selected_validation.png'))
        self.assertEqual(selected_state(image,{'Both Good':(1009,578,36,9),'Both Bad':(1060,578,36,9)})[:2],('Both Good','selected'))
        self.assertEqual(selected_state(image,{'Both Good':(1009,625,36,9),'Both Bad':(1060,625,36,9)})[:2],('Both Bad','selected'))

    def test_neighboring_selected_blue_button_does_not_contaminate_clear_b(self):
        image=cv2.imread(str(Path(__file__).parent/'fixtures'/'neighbor_selected_validation.png'))
        self.assertEqual(selected_state(image,{'Response A':(1475,817,71,19),'Response B':(1569,817,71,19)})[:2],('Response A','selected'))
        self.assertEqual(selected_state(image,{'Response A':(1476,888,70,19),'Response B':(1567,886,74,22)})[:2],('Response A','selected'))

    def test_recheck_keeps_prior_machine_click_provenance(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.save_auto_action('naturalness','A','unverified','old.png')
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder))
            controller=RatingController(flow)
            image=np.full((200,400,3),255,np.uint8)
            controller.store(image,[rating_row(selected='Response A')])
            controller.store(image,[rating_row(selected='Response A')])
            self.assertIsNone(record.data['human'])
            self.assertEqual(record.data['remote_ratings']['naturalness']['source'],'model_auto')
            self.assertEqual(record.data['auto_actions'][-1]['state'],'verified')

    def test_stable_observation_distinguishes_auto_manual_and_cleared(self):
        tracker=SelectionTracker({'naturalness':'A'})
        a=[rating_row(selected='Response A')]
        self.assertEqual(tracker.observe(a),{})
        self.assertEqual(tracker.observe(a)['naturalness']['source'],'model_auto')
        self.assertEqual(tracker.observe(a,{'naturalness':'A'})['naturalness']['source'],'human_remote')
        b=[rating_row(selected='Response B')]
        self.assertEqual(tracker.observe(b),{})
        self.assertEqual(tracker.observe(b)['naturalness']['judgment'],'B')
        self.assertEqual(tracker.observe([rating_row(state='ambiguous')]),{})
        self.assertEqual(tracker.observe([rating_row()]),{})
        self.assertEqual(tracker.observe([rating_row()])['naturalness']['judgment'],'需复核')

    def test_machine_labels_wait_for_user_confirmation_and_blank_reasons_are_kept(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.save_remote({'naturalness':{'judgment':'A','source':'model_auto'}},'auto.png')
            self.assertIsNone(record.data['human'])
            record.save_remote({'naturalness':{'judgment':'B','source':'human_remote'}},'user.png')
            self.assertEqual(record.data['model']['judgments']['naturalness']['judgment'],'A')
            self.assertEqual(record.data['human']['judgments']['naturalness']['reason'],'')
            self.assertEqual(record.data['disagreement_cases'][0]['human_judgment'],'B')
            self.assertEqual(record.data['rule_candidates'],[])
            record.save_remote({'naturalness':{'judgment':'需复核','source':'human_remote'}},'cleared.png')
            self.assertFalse(record.data['final_judgments']['naturalness']['usable_for_learning'])

    def test_submit_observation_confirms_auto_without_claiming_server_success(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.save_remote({'naturalness':{'judgment':'A','source':'model_auto'}},'auto.png')
            flow=SimpleNamespace(audio_folder=Path(folder),comparison_record=record,event=threading.Event(),grab=Mock())
            observer=RemoteRatingObserver(flow,{'rows':[],'submit_boxes':[(100,100,80,20)],'submit_enabled':True},np.zeros((200,400,3),np.uint8),{})
            observer.clicks.put(((130,110),0,observer.screen))
            observer.run()
            flow.grab.assert_not_called()
            self.assertTrue(flow.remote_submitted)
            self.assertEqual(record.data['human']['judgments']['naturalness']['source'],'user_submit_confirmation')
            self.assertFalse(record.data['submission_observation']['server_success_verified'])

    def test_disabled_submit_is_not_treated_as_user_confirmation(self):
        image=np.full((200,400,3),255,np.uint8)
        box=(100,100,80,20)
        cv2.rectangle(image,(80,90),(200,130),(145,145,145),-1)
        self.assertFalse(submit_enabled(image,[box]))
        cv2.rectangle(image,(80,90),(200,130),(130,60,35),-1)
        self.assertTrue(submit_enabled(image,[box]))

    def test_auto_fill_verifies_only_rating_click_and_does_not_create_human_label(self):
        with self.folder() as folder:
            record=self.record(folder)
            image=np.full((200,400,3),255,np.uint8)
            clicked=[False]
            done=threading.Event();done.set()
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),directory=Path(folder),
                analysis_done=done,event=threading.Event(),settings=SimpleNamespace(max_scrolls=1),
                status=Mock(),mouse=Mock(),snapshot=Mock(return_value=image),detector=object(),send=Mock(),log=Mock())
            screen={'rows':[rating_row()],'submit_boxes':[(90,150,80,20)]}
            def refresh(im,rows):
                return [{**row,'selected':'Response A' if clicked[0] else None,
                         'selection_state':'selected' if clicked[0] else 'none'} for row in rows]
            def click(box,shape,before_press=None,quick=False):
                before_press();clicked[0]=True
            flow.mouse.click_box.side_effect=click
            with patch('remote_ratings.read_rating_screen',return_value=screen), \
                    patch('remote_ratings.refresh_selections',side_effect=refresh), \
                    patch('remote_ratings.RemoteRatingObserver'):
                RatingController(flow).run()
            flow.mouse.click_box.assert_called_once()
            self.assertEqual(flow.mouse.click_box.call_args.args[0],(150,50,70,15))
            self.assertIsNone(record.data['human'])
            self.assertEqual(record.data['auto_actions'][-1]['state'],'verified')

    def test_bottom_scan_recovers_hidden_upper_required_rows_without_submit_click(self):
        with self.folder() as folder:
            record=self.record(folder)
            for key in record.data['model']['judgments']:
                record.data['model']['judgments'][key]['judgment']='不适用' if key=='speaker' else 'A'
            record.save_remote({'naturalness':{'judgment':'A','source':'model_auto'},
                                'pronunciation':{'judgment':'A','source':'model_auto'},
                                'overall':{'judgment':'A','source':'model_auto'},
                                'factor':{'judgment':'naturalness','source':'model_auto'}},'before.png')
            image=np.full((200,400,3),255,np.uint8)
            lower=[rating_row('naturalness','Response A')]
            upper=[rating_row('quality','Response A'),*lower]
            screens=[{'rows':lower,'submit_boxes':[(90,150,80,20)]},
                     {'rows':upper,'submit_boxes':[(90,150,80,20)]}]
            done=threading.Event();done.set()
            flow=SimpleNamespace(comparison_record=record,audio_folder=Path(folder),directory=Path(folder),
                analysis_done=done,event=threading.Event(),settings=SimpleNamespace(max_scrolls=3),
                status=Mock(),mouse=Mock(),snapshot=Mock(return_value=image),detector=object(),send=Mock(),log=Mock(),rng=Mock())
            with patch('remote_ratings.read_rating_screen',side_effect=screens), \
                    patch('remote_ratings.refresh_selections',side_effect=lambda image,rows:rows), \
                    patch('remote_ratings.RemoteRatingObserver'):
                RatingController(flow).run()
            flow.mouse.scroll.assert_called_once_with(image.shape,down=False,notches=3)
            flow.mouse.click_box.assert_not_called()
            self.assertTrue(record.data['rating_completion']['all_required_observed'])

    def test_page_reset_ends_observation_without_clearing_old_human_record(self):
        with self.folder() as folder:
            record=self.record(folder)
            record.save_remote({key:{'judgment':'A','source':'human_remote'} for key in ('naturalness','overall')},'old.png')
            image=np.full((200,400,3),255,np.uint8)
            rows=[rating_row('naturalness'),rating_row('overall')]
            flow=SimpleNamespace(audio_folder=Path(folder),comparison_record=record,event=threading.Event(),
                                 hwnd=1,grab=Mock(return_value=(image,None)))
            observer=RemoteRatingObserver(flow,{'rows':rows,'submit_boxes':[]},image,{})
            with patch('remote_ratings.refresh_selections',return_value=rows): observer.run()
            self.assertTrue(flow.remote_question_changed)
            self.assertEqual(record.data['human']['judgments']['naturalness']['judgment'],'A')

    def test_cached_row_labels_do_not_hide_a_changed_question_marker(self):
        before=np.full((200,400,3),255,np.uint8)
        after=before.copy();after[10:15,100:105]=0
        self.assertTrue(same_layout(before,after,[rating_row()]))
        self.assertFalse(same_layout(before,after,[rating_row()],[(50,5,150,20)]))


if __name__=='__main__': unittest.main()
