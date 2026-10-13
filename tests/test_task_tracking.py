from pathlib import Path
from types import SimpleNamespace
import unittest
from task_tracking import TaskTracking,record_relation


class TaskTrackingTests(unittest.TestCase):
    def test_restarted_recording_keeps_question_and_links_attempts(self):
        tracking=TaskTracking()
        first=SimpleNamespace(directory=Path('first'))
        second=SimpleNamespace(directory=Path('second'))
        tracking.attach(first);tracking.attach(second)
        self.assertEqual(first.task_tracking['task_id'],second.task_tracking['task_id'])
        self.assertEqual(second.task_tracking['attempt_number'],2)
        self.assertEqual(second.task_tracking['retry_of_session'],'first')
        self.assertEqual(first.task_started_monotonic,second.task_started_monotonic)
        next_question=TaskTracking(2);third=SimpleNamespace(directory=Path('third'))
        next_question.attach(third)
        self.assertNotEqual(first.task_tracking['task_id'],third.task_tracking['task_id'])

    def test_explicit_question_ids_override_time_proximity(self):
        a={'task_tracking':{'task_id':'a'}};b={'task_tracking':{'task_id':'b'}}
        self.assertEqual(record_relation(a,a),'same_question_retry')
        self.assertEqual(record_relation(a,b),'separate_question_occurrence')

    def test_adjacent_times_alone_do_not_prove_same_question(self):
        a={'started_at':'2026-10-12T10:00:00+01:00'}
        b={'started_at':'2026-10-12T10:01:00+01:00'}
        self.assertEqual(record_relation(a,b),'unknown')
        a['question_marker']=[('prompt','1'),('run','2')]
        b['question_marker']=[('prompt','1'),('run','2')]
        self.assertEqual(record_relation(a,b),'likely_same_question_retry')
        b['question_marker']=[('prompt','3'),('run','2')]
        self.assertEqual(record_relation(a,b),'different_question_marker')
