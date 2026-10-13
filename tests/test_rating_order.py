import unittest
from unittest.mock import Mock,patch
from integrated_app import RecordedWorkflow
from remote_ratings import ordered_rows,choose_action,hidden_earlier_row


class RatingOrderTests(unittest.TestCase):
    def test_hidden_quality_is_located_before_lower_rows_are_clicked(self):
        from types import SimpleNamespace
        rows=[{'key':'speaker','label_box':(0,100,10,10)}]
        record=SimpleNamespace(data={'remote_ratings':{}})
        self.assertEqual(hidden_earlier_row(rows,record,{'quality':{'judgment':'Both Good'}}),'quality')
        record.data['remote_ratings']['quality']={'judgment':'Both Good'}
        self.assertIsNone(hidden_earlier_row(rows,record,{'quality':{'judgment':'Both Good'}}))
    def test_rows_follow_screen_order_even_if_ocr_returns_reversed_rows(self):
        rows=[{'key':key,'label_box':(10,y,40,15)} for key,y in [('overall',200),('quality',20),('naturalness',150)]]
        self.assertEqual([r['key'] for r in ordered_rows(rows)],['quality','naturalness','overall'])

    def test_both_good_is_a_clickable_action(self):
        row={'key':'quality','selection_state':'none','options':{'Response A':(1,1,5,5),'Both Good':(20,1,5,5)}}
        self.assertEqual(choose_action(row,{'quality':{'judgment':'Both Good'}},set()),('quality','Both Good','Both Good'))

    def test_review_restores_settings_on_failure_and_never_uses_old_form_coordinates(self):
        from workflow import Settings
        flow=object.__new__(RecordedWorkflow)
        flow.settings=Settings();flow.recovery_identity={'marker':(('prompt','1'),)}
        flow.detector=Mock(last_task_lines=[(None,'Prompt 1',1)])
        flow.status=Mock();flow.log=Mock();flow.mouse=Mock();flow.rng=Mock()
        flow.rng.randint.return_value=3
        flow.snapshot=Mock(return_value=Mock(shape=(100,100,3)))
        flow.identify=Mock();flow.verify_recovery_identity=Mock()
        flow._opening_previews=Mock(side_effect=RuntimeError('pause failed'))
        with self.assertRaisesRegex(RuntimeError,'pause failed'):flow.mid_rating_review()
        self.assertEqual(flow.settings.short_count,6)
        self.assertFalse(flow.review_in_progress)
        flow.verify_recovery_identity.assert_called_once()
