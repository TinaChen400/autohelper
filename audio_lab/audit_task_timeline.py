"""Reconstruct legacy attempt times without labelling retries as repeated tasks."""
from datetime import datetime
import json
from pathlib import Path


def read(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def audit(sessions,date):
    rows=[];groups=[];last_known=None
    for folder in sorted(sessions.glob('integrated_session_'+date.replace('-','')+'*')):
        if any(s in folder.name for s in ('_timing_rechecked','_progress_rechecked')):continue
        record=read(folder/'comparison.json');metadata=read(folder/'session.json')
        started=record.get('started_at')
        marker=dict(metadata.get('question_marker',[]) or [])
        row={'session':folder.name,'started_at':started,'question_marker':marker,
             'submit_clicked_at':record.get('submission_observation',{}).get('clicked_at'),
             'server_success_verified':record.get('submission_observation',{}).get('server_success_verified',False)}
        rows.append(row)
        if not marker.get('prompt') or not marker.get('run'):continue
        near=False
        if last_known and started and last_known['started_at']:
            near=0<=(datetime.fromisoformat(started)-datetime.fromisoformat(last_known['started_at'])).total_seconds()<=900
        if (last_known and near and marker==last_known['question_marker'] and
                not last_known['server_success_verified']):
            group=groups[-1]
            group['sessions'].append(row['session'])
            group['last_started_at']=started
        else:
            groups.append({'question_marker':marker,'first_started_at':started,'last_started_at':started,
                           'sessions':[row['session']]})
        last_known=row
    retries=[{**group,'classification':'likely_same_question_retry',
              'basis':'相邻已识别题目编号相同、间隔不超过15分钟、未确认上次提交成功；时间本身不证明音频相同'}
             for group in groups if len(group['sessions'])>1]
    return {'date':date,'attempt_count':len(rows),'attempts':rows,'likely_retry_groups':retries,
            'note':'旧记录没有持久题目ID，分组仅为推断；未改写原始记录。'}


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    report=audit(root/'audio_lab'/'sessions','2026-10-12')
    path=root/'artifacts'/'task_timeline_20261012.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'attempts':report['attempt_count'],'likely_retry_groups':len(report['likely_retry_groups']),
                      'report':str(path)},ensure_ascii=True))
