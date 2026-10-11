import unittest
from unittest.mock import Mock
from workflow import Workflow, ClickRecheck


class ClickRecheckTests(unittest.TestCase):
    def workflow(self):
        value = object.__new__(Workflow)
        value.targets = {'A': object()}
        value.snapshot = Mock(return_value=object())
        value.check_page = Mock()
        value.parked = Mock()
        value.mouse = Mock()
        value.log = Mock()
        value.status = Mock()
        return value

    def test_only_prepress_failure_is_retried_after_parking(self):
        value = self.workflow()
        value._click_once = Mock(side_effect=[ClickRecheck('pointer obscured'), None])
        value.click('A', fast=True)
        self.assertEqual(value._click_once.call_count, 2)
        value.parked.assert_called_once()

    def test_retries_are_bounded(self):
        value = self.workflow()
        value._click_once = Mock(side_effect=ClickRecheck('uncertain'))
        with self.assertRaisesRegex(RuntimeError, '连续三次'):
            value.click('A')
        self.assertEqual(value._click_once.call_count, 3)

    def test_postclick_or_page_failure_is_never_retried(self):
        value = self.workflow()
        value._click_once = Mock(side_effect=RuntimeError('page changed'))
        with self.assertRaisesRegex(RuntimeError, 'page changed'):
            value.click('A')
        self.assertEqual(value._click_once.call_count, 1)
