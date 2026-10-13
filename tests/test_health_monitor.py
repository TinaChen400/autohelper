import unittest
from health_monitor import HealthMonitor

class HealthMonitorTests(unittest.TestCase):
    def monitor(self):
        m=HealthMonitor();m.begin('same-question',0);return m
    def test_frozen_active_timer_is_not_progress(self):
        m=self.monitor();data={'task_id':'same-question','readings':{'B':{'state':'active','current':3,'total':12}}}
        m.observe('playback_sample',data,1)
        m.observe('playback_sample',data,40)
        alert=m.snapshot(41,True)
        self.assertIn('40 秒',alert['title'])
        self.assertEqual(alert['level'],'warning')
        self.assertEqual(m.snapshot(122,True)['level'],'error')
        data['readings']['B']['current']=4;m.observe('playback_sample',data,123)
        self.assertIsNone(m.snapshot(124,True))
    def test_retries_survive_new_attempt_and_duplicate_propagation_is_not_counted(self):
        m=self.monitor();data={'task_id':'same-question','reason':'A boundary gap'}
        m.observe('first_round_retry',data,1);m.observe('continuous_recovery',data,1.1)
        self.assertEqual(m.failures['A boundary gap'],1)
        m.begin('same-question',20)
        m.observe('first_round_retry',data,21)
        self.assertEqual(m.snapshot(22,True)['level'],'warning')
        m.observe('first_round_retry',data,31)
        self.assertEqual(m.snapshot(32,True)['level'],'error')
        m.observe('capture_validation_passed',{'task_id':'same-question'},33)
        self.assertIsNone(m.snapshot(34,True))
    def test_new_question_clears_previous_fault(self):
        m=self.monitor()
        for stamp in (1,10,20):m.observe('continuous_recovery',{'task_id':'same-question','reason':'failed'},stamp)
        m.begin('new-question',21)
        self.assertIsNone(m.snapshot(22,True))
    def test_model_budget_and_focus_wait_are_distinct(self):
        m=self.monitor();m.observe('analysis_started',{'task_id':'same-question'},1)
        self.assertIsNone(m.snapshot(121,True))
        self.assertIn('生成评价',m.snapshot(181,True)['title'])
        self.assertIsNone(m.snapshot(5,True,'远程窗口失去焦点'))
        self.assertIn('等待远程窗口',m.snapshot(21,True,'远程窗口失去焦点')['title'])
    def test_pending_issue_stays_visible_after_stop(self):
        m=self.monitor()
        m.observe('capture_repair_pending',{'task_id':'same-question','reason':'录音无法切分'},1)
        self.assertEqual(m.snapshot(2,False,capture_pending=True)['level'],'error')
        self.assertIsNone(m.snapshot(3,False))

    def test_replay_does_not_override_active_model_budget(self):
        m=self.monitor();m.observe('analysis_started',{'task_id':'same-question'},1)
        m.observe('playback_sample',{'task_id':'same-question','readings':{'A':3}},10)
        self.assertIsNone(m.snapshot(60,True,analysis_started=1))
        self.assertIn('生成评价',m.snapshot(182,True,analysis_started=1)['title'])

    def test_completed_model_and_unknown_id_report_actual_blocker(self):
        m=self.monitor()
        m.observe('analysis_completed',{'task_id':'same-question','state':'draft_ready'},20)
        m.observe('playback_sample',{'task_id':'same-question','readings':{'B':{'current':7}}},21)
        for stamp in (25,40,55):
            m.observe('question_identity_check',{'task_id':'same-question','result':'unknown'},stamp)
        alert=m.snapshot(56,True)
        self.assertIn('核对题号',alert['title'])
        self.assertIn('模型评价已经保存',alert['detail'])
        self.assertNotIn('播放观察',alert['title'])
        self.assertEqual(m.progress_at,21)

    def test_changed_rating_observation_counts_but_identical_rescans_do_not(self):
        m=self.monitor()
        data={'task_id':'same-question','dimensions':[{'key':'quality','selected':'Both Good','selection_state':'selected'}]}
        m.observe('rating_page',data,25)
        self.assertIsNone(m.snapshot(31,True))
        m.observe('rating_page',data,50)
        self.assertIsNotNone(m.snapshot(56,True))

    def test_late_background_events_cannot_replace_pending_submit_phase(self):
        m=self.monitor()
        for stamp in (1,5,10):m.observe('continuous_recovery',{'task_id':'same-question','reason':'old fault'},stamp)
        m.observe('submission_pending',{'task_id':'same-question'},20)
        m.observe('analysis_completed',{'task_id':'same-question','state':'draft_ready'},21)
        self.assertEqual(m.phase,'等待提交结果')
        self.assertIsNone(m.snapshot(22,True,analysis_started=1))
        self.assertIn('提交结果尚未确认',m.snapshot(23,False)['title'])
        self.assertIn('等待提交结果',m.snapshot(51,True,analysis_started=1)['title'])

if __name__=='__main__':unittest.main()
