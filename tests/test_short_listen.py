import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from playback import Reading
from short_listen import choose_duration, reset_control
from vision import Target
from workflow import Settings, Workflow


class ShortListenTests(unittest.TestCase):
    def test_short_duration_floor_randomness_and_clip_headroom(self):
        rng = random.Random(23)
        for total in (1, 2, 5, 30):
            values = [choose_duration(total, .4, 1.2, rng) for _ in range(20)]
            self.assertTrue(all(.4 <= v <= min(1.2,total*.5,total-.35) for v in values))
            self.assertGreater(len(set(values)), 1)
        self.assertIsNone(choose_duration(.5, .4, 1.2, rng))

    def test_reset_template_on_actual_three_players_and_blank(self):
        image = cv2.imread(str(Path(__file__).parent/'fixtures'/'timer_resampling_retry.png'))
        for x, y in ((212,309),(209,651),(965,651)):
            target = Target('Input', (), (x,y,30,29), 'idle',1)
            box = reset_control(image,target)
            self.assertIsNotNone(box)
            self.assertTrue(x+30 < box[0] < x+60)
            self.assertIsNone(reset_control(np.full_like(image,255),target))

    def test_real_short_workflow_resets_plays_pauses_and_rotates(self):
        image = np.zeros((200,500,3),np.uint8)
        targets = {n:Target(n,(x,10,20,10),(x,40,30,30),'idle',1) for n,x in zip(('Input','A','B'),(10,160,310))}
        totals = dict(Input=2,A=5,B=9)
        clock = [0.0]
        state = {n:dict(active=False, current=totals[n], start=0) for n in targets}
        actions = []
        def reading(name):
            row = state[name]
            current = min(totals[name],int(clock[0]-row['start'])) if row['active'] else row['current']
            return Reading('active' if row['active'] else 'idle',current,totals[name])
        def glyph(im,box):
            name = next(n for n,t in targets.items() if t.button == box)
            return reading(name).state
        def reset(im,target):
            x,y,w,h = target.button
            return (x+40,y,20,20)
        class Mouse:
            def __init__(self,*args): pass
            def check(self): pass
            def wait(self,seconds): clock[0] += seconds
            def move(self,*args,**kwargs): clock[0] += .1
            def click_box(self,box,shape,before_press=None,**kwargs):
                if before_press: before_press()
                for name,target in targets.items():
                    row = state[name]
                    if box == reset(image,target):
                        row.update(active=False,current=0)
                        actions.append(('reset',name))
                        break
                    if box == target.button:
                        if row['active']:
                            row.update(active=False,current=reading(name).current)
                            actions.append(('pause',name))
                        else:
                            self_test.assertEqual(row['current'],0)
                            row.update(active=True,start=clock[0])
                            actions.append(('play',name))
                        break
        self_test = self
        with tempfile.TemporaryDirectory() as folder, patch('workflow.HumanMouse',Mouse), \
                patch('workflow.glyph_state',side_effect=glyph), \
                patch('workflow.read_players',side_effect=lambda d,im,ts:{n:reading(n) for n in ts}), \
                patch('workflow.time.monotonic',side_effect=lambda:clock[0]), \
                patch('short_listen.reset_control',side_effect=reset):
            workflow = Workflow(123,object(),threading.Event(),lambda *args:None,lambda hwnd:(image,None),
                                Settings(short_enabled=True),folder)
            workflow.targets = targets
            workflow.known_readings = {n:reading(n) for n in targets}
            workflow.check_page = lambda im:None
            workflow.opening_previews()
            plays = [n for action,n in actions if action == 'play']
            self.assertEqual(plays[:3],['Input','A','B'])
            self.assertEqual(len(plays),6)
            self.assertEqual(len([a for a in actions if a[0]=='pause']),6)
            self.assertFalse(any(row['active'] for row in state.values()))
            events = [json.loads(s) for s in (workflow.directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sum(e['event']=='short_listen_completed' for e in events),6)
