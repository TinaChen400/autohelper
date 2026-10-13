"""Exercise the real playback loop through replays and scrolling, without UI input."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from playback import Reading
from workflow import Workflow, Settings


class ContinuousFlowTests(unittest.TestCase):
    def test_actual_lengths_and_long_autoplay_gap_are_not_fixed_deadlines(self):
        for lengths in ((1,5,2), (4,2,9)):
            with self.subTest(lengths=lengths):
                order = ('Input','A','B')
                rows = []
                now = 0.0
                for index,name in enumerate(order):
                    for tick in range(lengths[index]*2):
                        readings = {n: Reading('idle',duration,duration) for n,duration in zip(order,lengths)}
                        readings[name] = Reading('active',tick//2,lengths[index])
                        rows.append((now,readings)); now += .5
                    idle = {n: Reading('idle',duration,duration) for n,duration in zip(order,lengths)}
                    rows.extend(((now,idle),(now+.5,idle)))
                    now += 1
                    if index == 0:
                        rows.append((now+60,idle)); now += 61
                clock = [0.0]; iterator = iter(rows)
                image = np.zeros((80,120,3),np.uint8)
                class Mouse:
                    def __init__(self,*args): pass
                    def check(self): pass
                    def wait(self,*args): pass
                def read(*args):
                    clock[0],readings = next(iterator)
                    return readings
                with tempfile.TemporaryDirectory() as directory, patch('workflow.HumanMouse',Mouse), \
                        patch('workflow.read_players',side_effect=read), \
                        patch('workflow.time.monotonic',side_effect=lambda:clock[0]):
                    workflow = Workflow(123,object(),threading.Event(),lambda *args:None,
                                        lambda hwnd:(image,None),Settings(),directory)
                    workflow.check_page = lambda value:None
                    workflow.await_chain(order)
                    self.assertIn('chain_completed',(workflow.directory/'events.jsonl').read_text(encoding='utf-8'))

    def test_recorded_23_10_10_second_round_reaches_completion(self):
        rows = json.loads((Path(__file__).parent/'fixtures'/'autoplay_trace.json').read_text(encoding='utf-8'))
        clock = [0.0]
        iterator = iter(rows)
        image = np.zeros((80,120,3), np.uint8)
        class Mouse:
            def __init__(self,*args): pass
            def check(self): pass
            def wait(self,*args): pass
        def read(*args):
            row = next(iterator)
            clock[0] = row['elapsed']
            return {name: Reading(**reading) for name,reading in row['readings'].items()}
        with tempfile.TemporaryDirectory() as directory, patch('workflow.HumanMouse',Mouse), \
                patch('workflow.read_players',side_effect=read), \
                patch('workflow.time.monotonic',side_effect=lambda: clock[0]):
            workflow = Workflow(123,object(),threading.Event(),lambda *args: None,
                                lambda hwnd:(image,None),Settings(),directory)
            workflow.check_page = lambda value: None
            workflow.await_chain(('Input','A','B'))
            events = [json.loads(line)['event'] for line in (workflow.directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(events[-1],'chain_completed')

    def test_completed_autoplay_reaches_two_replays_then_scroll(self):
        self.exercise_replays(False)

    def test_missing_timer_text_still_reaches_input_replay_and_scroll(self):
        self.exercise_replays(True)

    def exercise_replays(self, missing_timers):
        clock = [0.0]
        order = ('Input', 'A', 'B')
        playback = {'start': None, 'selected': 'Input'}
        clicks, scrolls = [], []
        image = np.zeros((80, 120, 3), np.uint8)

        class Mouse:
            def __init__(self, *args): pass
            def check(self): pass
            def move(self, *args, **kwargs): pass
            def wait(self, seconds): clock[0] += seconds
            def scroll(self, *args, **kwargs): scrolls.append(kwargs); clock[0] += 1

        def read(*args):
            # Controls retain the previous play's final time until their turn.
            result = {name: Reading('idle', 2, 2) for name in order}
            if playback['start'] is not None:
                elapsed = clock[0]-playback['start']
                index = order.index(playback['selected'])+int(elapsed//2)
                if index < len(order): result[order[index]] = Reading('active', int(elapsed%2), 2)
            if missing_timers and playback['start'] is not None:
                result = {name: Reading(value.state) for name, value in result.items()}
            return result

        def click(name, **kwargs):
            clicks.append(name)
            playback.update(start=clock[0], selected=name)

        dimensions = [{'name': 'Audio Quality', 'options': {}}]
        with tempfile.TemporaryDirectory() as directory, \
                patch('workflow.HumanMouse', Mouse), \
                patch('workflow.read_players', side_effect=read), \
                patch('workflow.time.monotonic', side_effect=lambda: clock[0]), \
                patch('workflow.read_rating_page', side_effect=lambda *args: (dimensions, bool(scrolls))):
            workflow = Workflow(123, object(), threading.Event(), lambda *args: None,
                                lambda hwnd: (image, None), Settings(2,2,0,0,3), directory)
            workflow.identify = lambda: (image, read())
            workflow.known_readings = {name: Reading('idle', 2, 2) for name in order}
            workflow.check_page = lambda value: None
            workflow.click = click
            workflow.run()
            events = [json.loads(line)['event'] for line in (workflow.directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(clicks), 3)
            self.assertEqual(clicks[0], 'Input')
            self.assertEqual(clicks[1], 'Input')
            self.assertIn(clicks[2], ('A', 'B'))
            self.assertTrue(scrolls)
            self.assertEqual(events.count('replay_completed'), 2)
            self.assertIn('rating_scroll_start', events)
            self.assertEqual(events[-1], 'session_end')
            self.assertTrue((workflow.directory/'dimensions.json').exists())
            waits = [json.loads(line) for line in (workflow.directory/'events.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['event'] == 'replay_wait']
            self.assertEqual(waits[0]['seconds'],1.0)


if __name__ == '__main__': unittest.main()
