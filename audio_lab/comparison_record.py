"""Persist independent model evidence and timed human labels for one trial."""
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from audio_lab.durable_io import write_json

DIMENSIONS = ('quality', 'speaker', 'pronunciation', 'naturalness', 'overall')


def has_reference(folder):
    folder=Path(folder)
    for filename in ('session.json','clip_manifest.json'):
        try:
            data=json.loads((folder/filename).read_text(encoding='utf-8'))
            if 'player_order' in data: return 'Input' in data['player_order']
        except (OSError,ValueError): pass
    return True


def model_judgments(folder):
    folder = Path(folder)
    result = {key: {'judgment': '需复核', 'reason': '当前模型未独立判定此维度'} for key in DIMENSIONS}
    try:
        comparison = json.loads((folder/'model_comparison.json').read_text(encoding='utf-8'))
        for key, scores in [('speaker', comparison.get('speaker_similarity', {}))]:
            if all(n in scores for n in ('A', 'B')):
                result[key] = {'judgment': 'A' if scores['A'] > scores['B'] else 'B' if scores['B'] > scores['A'] else '无差异',
                               'reason': f"ECAPA 相似度 A={scores['A']}, B={scores['B']}；未校准的候选倾向"}
        result['quality']['reason'] = 'SigMOS 仅提供辅助分数，不能确认明显可听缺陷或 Both Good'
        result['pronunciation']['reason'] = '目标正文尚未核对，转写不能直接决定忠实度'
        result['quality']['scores'] = comparison.get('quality', {})
        result['pronunciation']['transcripts'] = comparison.get('transcripts', {})
    except (OSError, ValueError, KeyError): pass
    try:
        scores = json.loads((folder/'naturalness.json').read_text(encoding='utf-8'))['scores']
        a, b = scores['A']['mos'], scores['B']['mos']
        result['naturalness'] = {'judgment': 'A' if a > b else 'B' if b > a else '无差异',
                                'reason': f'UTMOSv2 A={a}, B={b}；未校准的候选倾向'}
    except (OSError, ValueError, KeyError): pass
    if not has_reference(folder):
        result['speaker']={'judgment':'不适用','reason':'本题仅 A/B，没有参考音频，不评价与 Input 的音色相似度'}
    try:
        decisions=json.loads((folder/'evaluation_decisions.json').read_text(encoding='utf-8'))
        if decisions.get('state')=='complete_candidates':
            for key in DIMENSIONS:
                candidate=decisions['judgments'][key]
                if candidate['judgment'] not in ('A','B','Both Good','Both Bad','不适用','需复核'):
                    raise ValueError('无效自动候选')
            result={key:decisions['judgments'][key] for key in DIMENSIONS}
    except (OSError,ValueError,KeyError): pass
    return result


class ComparisonRecord:
    @classmethod
    def load(cls, folder, library_path=None):
        record=cls.__new__(cls)
        record.folder=Path(folder)
        record.library_path=Path(library_path) if library_path else None
        record.lock=threading.RLock()
        record.data=json.loads((record.folder/'comparison.json').read_text(encoding='utf-8'))
        record.data.setdefault('feedback_history',[])
        record.data.setdefault('remote_ratings',{})
        record.data.setdefault('auto_actions',[])
        started=datetime.fromisoformat(record.data['started_at'])
        record.started=time.monotonic()-(datetime.now().astimezone()-started).total_seconds()
        return record

    def __init__(self, folder, library_path=None):
        self.folder = Path(folder)
        self.library_path = Path(library_path) if library_path else None
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.data = {'schema_version': 2, 'started_at': datetime.now().astimezone().isoformat(),
                     'human': None, 'model': None, 'timings': {}, 'feedback_history': [],
                     'remote_ratings':{},'auto_actions':[]}
        self.update()

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            human, model = self.data['human'], self.data['model']
            if human and model:
                self.data['differences'] = {
                    key: {'human': human['judgments'][key]['judgment'],
                          'model': model['judgments'][key]['judgment'],
                          'same': human['judgments'][key]['judgment'] == model['judgments'][key]['judgment'],
                          'requires_review': model['state'] not in ('draft_ready','partial_ready') or
                              model['judgments'][key]['judgment'] == '需复核'}
                    for key in DIMENSIONS}
            if human:
                self.data['final_judgments'] = {
                    key: {**human['judgments'][key], 'authority': 'user' if human['judgments'][key]['judgment'] != '需复核' else 'unresolved',
                          'usable_for_learning': human['judgments'][key]['judgment'] not in ('需复核','不适用')}
                    for key in DIMENSIONS}
                self.data['rule_candidates'] = [
                    {'dimension': key, 'human_judgment': human['judgments'][key]['judgment'],
                     'model_judgment': model['judgments'][key]['judgment'],
                     'reason': human['judgments'][key]['reason'],
                     'status': '待跨题验证', 'source_session': self.folder.name}
                    for key in DIMENSIONS if model and
                    human['judgments'][key]['judgment'] != '需复核' and
                    human['judgments'][key]['judgment'] != model['judgments'][key]['judgment'] and
                    human['judgments'][key]['reason'].strip()]
                self.data['disagreement_cases']=[
                    {'dimension':key,'human_judgment':human['judgments'][key]['judgment'],
                     'model_judgment':model['judgments'][key]['judgment'],
                     'reason':human['judgments'][key].get('reason',''),'status':'待后续总结',
                     'source_session':self.folder.name}
                    for key in DIMENSIONS if model and
                    human['judgments'][key]['judgment'] not in ('需复核','不适用') and
                    model['judgments'][key]['judgment'] not in ('需复核','不适用') and
                    human['judgments'][key]['judgment']!=model['judgments'][key]['judgment']]
            write_json(self.folder/'comparison.json', self.data)
            if self.library_path and human and ('human' in values or 'model' in values):
                with self.library_path.open('a',encoding='utf-8') as out:
                    out.write(json.dumps({'session':self.folder.name,
                        'recorded_at':datetime.now().astimezone().isoformat(),
                        'human':human,'model':model,
                        'final_judgments':self.data['final_judgments'],
                        'rule_candidates':self.data['rule_candidates'],
                        'disagreement_cases':self.data.get('disagreement_cases',[])},ensure_ascii=False)+'\n')

    def save_remote(self, observations, evidence, submit_clicked=False, confirm_auto=True):
        with self.lock:
            self._save_remote(observations,evidence,submit_clicked,confirm_auto)

    def _save_remote(self, observations, evidence, submit_clicked, confirm_auto):
        """Store visible choices; machine clicks are not independent human labels."""
        remote={**self.data.get('remote_ratings',{})}
        now=time.monotonic()
        stamp=datetime.now().astimezone().isoformat()
        changed=False
        for key,item in observations.items():
            allowed=('quality','speaker','pronunciation','naturalness','需复核') if key=='factor' else ('A','B','Both Good','Both Bad','不适用','需复核')
            if key not in (*DIMENSIONS,'factor') or item.get('judgment') not in allowed:
                continue
            previous=remote.get(key,{})
            if previous.get('judgment')==item['judgment'] and previous.get('source')==item['source']:
                continue
            remote[key]={**item,'reason':'','observed_at':stamp,'evidence':str(evidence)}
            changed=True
        if submit_clicked and confirm_auto:
            for key,item in remote.items():
                if item['source']=='model_auto':
                    remote[key]={**item,'source':'user_submit_confirmation','confirmed_at':stamp}
                    changed=True
        if not changed and not submit_clicked: return
        values={'remote_ratings':remote}
        if submit_clicked:
            values['submission_observation']={'clicked_at':stamp,'server_success_verified':False,
                'evidence':str(evidence),'note':'仅观察到用户点击 Submit，未代为提交，也未验证服务器处理结果'}
        manual={key:item for key,item in remote.items() if item['source'] in ('human_remote','user_submit_confirmation')}
        if manual:
            old=self.data.get('human') or {}
            judgments={key:dict(old.get('judgments',{}).get(key,{'judgment':'需复核','reason':'','source':'unresolved'})) for key in DIMENSIONS}
            for key,item in manual.items():
                if key=='factor': continue
                judgments[key]={'judgment':item['judgment'],'reason':'','source':item['source'],'evidence':item['evidence']}
            if not has_reference(self.folder):
                judgments['speaker']={'judgment':'不适用','reason':'','source':'not_applicable'}
            factor=manual.get('factor')
            if judgments!=old.get('judgments') or (factor and factor!=old.get('dominant_factor')):
                human={'saved_at':stamp,'elapsed_seconds':now-self.started,
                       'seconds_since_session_start':now-self.started,
                       'timing_basis':'远程选项首次稳定识别的时间；非精确点击反应时间',
                       'model_available_before_observation':bool(self.data.get('model')),
                       'model_visible_before_save':True if any(item.get('state')=='verified' for item in self.data.get('auto_actions',[]))
                           or any(item['source']=='user_submit_confirmation' for item in manual.values())
                           else None if self.data.get('model') else False,
                       'judgments':judgments}
                if factor: human['dominant_factor']=factor
                elif old.get('dominant_factor'): human['dominant_factor']=old['dominant_factor']
                values.update(human=human,feedback_history=[*self.data['feedback_history'],human])
        self.update(**values)

    def save_auto_action(self, dimension, judgment, state, evidence):
        self.update(auto_actions=[*self.data.get('auto_actions',[]),
            {'dimension':dimension,'judgment':judgment,'state':state,'evidence':str(evidence),
             'recorded_at':datetime.now().astimezone().isoformat()}])

    def save_human(self, judgments, began, model_visible):
        with self.lock:
            self._save_human(judgments,began,model_visible)

    def _save_human(self, judgments, began, model_visible):
        resolved = {}
        for key in DIMENSIONS:
            item = judgments[key]
            answer = item['judgment']
            source = 'human_selection'
            if key=='speaker' and not has_reference(self.folder):
                resolved[key]={'judgment':'不适用','reason':'无参考音频','source':'not_applicable'}
                continue
            if not answer:
                if not self.data['model']:
                    raise ValueError('模型尚未结束：请等待结果，或选择自己的判断。理由可以留空。')
                answer = self.data['model']['judgments'][key]['judgment']
                source = 'model_confirmation' if answer != '需复核' else 'unresolved'
                if self.data['model']['state'] not in ('draft_ready','partial_ready'):
                    answer, source = '需复核', 'unresolved'
            resolved[key] = {'judgment': answer, 'reason': item.get('reason', '').strip(), 'source': source}
        now = time.monotonic()
        human={'saved_at': datetime.now().astimezone().isoformat(),
                          'elapsed_seconds': now-began, 'seconds_since_session_start': now-self.started,
                          'model_visible_before_save': model_visible, 'judgments': resolved}
        self.update(human=human, feedback_history=[*self.data['feedback_history'], human])

    def save_model(self, status, first_round_finished, analysis_started, recovered=False):
        now = time.monotonic()
        completed_at=status.get('completed_at',datetime.now().astimezone().isoformat())
        if recovered:
            now -= (datetime.now().astimezone()-datetime.fromisoformat(completed_at)).total_seconds()
        judgments = model_judgments(self.folder)
        if status['state'] not in ('draft_ready','partial_ready'):
            for item in judgments.values():
                if item['judgment']=='不适用': continue
                item['candidate_before_failure'] = item['judgment']
                item['judgment'] = '需复核'
                item['reason'] = '分析未完整成功：'+status.get('reason','请检查模型日志')+'；'+item['reason']
        for stage,reason in status.get('stage_errors',{}).items():
            dimension={'asr':'pronunciation','quality':'quality','speaker':'speaker','naturalness':'naturalness'}.get(stage)
            if dimension:
                judgments[dimension]['judgment']='需复核'
                judgments[dimension]['reason']=reason
        self.update(model={'saved_at': completed_at, 'state': status['state'],
                           'reason': status.get('reason'), 'judgments': judgments,
                           'elapsed_seconds': status.get('elapsed_seconds',0) if recovered else now-analysis_started, 'pipeline': status},
                    timings={'first_round_seconds': first_round_finished-self.started,
                             'model_ready_since_session_seconds': now-self.started,
                             'first_round_to_model_seconds': now-first_round_finished})
