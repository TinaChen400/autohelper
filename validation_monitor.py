"""Read-only continuous-run audit; count actual submissions and task failures."""
import argparse
from collections import Counter
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CODE=('integrated_app.py','continuous_runner.py','remote_ratings.py','rating_reader.py','question_identity.py',
      'human_mouse.py','text_follow.py','audio_lab/decision_engine.py','audio_lab/live_follow.py','audio_lab/task_context.py',
      'audio_lab/live_capture.py','audio_lab/durable_io.py','audio_lab/segment_session.py',
      'health_monitor.py','compact_ui.py','audio_lab/comparison_record.py')
CODE=(*CODE,'question_progress.py','submit_monitor.py')


def inspect(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    started=datetime.fromisoformat(manifest['started_at'])
    finished=datetime.fromisoformat(manifest['finished_at']) if manifest.get('finished_at') else None
    rows=[]
    for path in (ROOT/'artifacts').glob('session_*/events.jsonl'):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row=json.loads(line);stamp=datetime.fromisoformat(row['time']).astimezone()
                if stamp>=started and (finished is None or stamp<=finished):rows.append({**row,'session':path.parent.name})
            except (ValueError,KeyError):pass
    rows.sort(key=lambda row:row['time'])
    events=Counter(row['event'] for row in rows)
    failures=[row for row in rows if row['event'] in ('capture_validation_failed','capture_repair_pending',
                                                     'first_round_retry','review_pending','submission_incomplete')]
    tasks=[]
    for name in sorted({row['session'] for row in rows}):
        related=[row for row in rows if row['session']==name]
        recording=next((row['recording_session'] for row in related if row.get('recording_session')),'integrated_'+name)
        path=ROOT/'audio_lab/sessions'/recording/'comparison.json'
        if not path.exists():continue
        try:data=json.loads(path.read_text(encoding='utf-8'))
        except ValueError:continue
        submitted=any(row['event']=='auto_submit_clicked' or
                      row['event']=='submission_pending' and row.get('actor')=='user' for row in related)
        model_ready=[row for row in related if row['event']=='analysis_completed']
        presses=[row for row in related if row['event']=='submission_pending']
        ready_to_submit=(datetime.fromisoformat(presses[0]['time'])-datetime.fromisoformat(model_ready[-1]['time'])).total_seconds() if presses and model_ready else None
        capture=(ROOT/'audio_lab/sessions'/('integrated_'+name)/'capture_status.json')
        try:capture_data=json.loads(capture.read_text(encoding='utf-8'))
        except (OSError,ValueError):capture_data={}
        tasks.append({'session':name,'marker':data.get('question_marker'),
            'task_id':data.get('task_tracking',{}).get('task_id'),
            'submitted':submitted,'server_verified':data.get('submission_observation',{}).get('server_success_verified',False),
            'missing':data.get('rating_completion',{}).get('missing'),
            'consistent':data.get('preference_consistency',{}).get('consistent'),
            'model_seconds':(data.get('model') or {}).get('elapsed_seconds'),
            'model_ready_to_submit_seconds':ready_to_submit,
            'submission_actor':data.get('submission_observation',{}).get('actor'),
            'progress':data.get('question_progress',{}).get('phase'),
            'timings':data.get('timings',{}),'capture_state':capture_data.get('state'),
            'last_event':related[-1]['event']})
    elapsed=((finished or datetime.now().astimezone())-started).total_seconds()
    zones=Counter(str(row.get('zone')) for row in rows if row['event'] in ('mouse_park','mouse_gaze'))
    submitted=[task for task in tasks if task['submitted']]
    alerts=[]
    health_path=ROOT/'artifacts'/'health_alerts.jsonl'
    if health_path.exists():
        for line in health_path.read_text(encoding='utf-8').splitlines():
            try:
                alert_row=json.loads(line);stamp=datetime.fromisoformat(alert_row['time'])
                if stamp>=started and (finished is None or stamp<=finished):alerts.append(alert_row)
            except (ValueError,KeyError):pass
    observed_wait=0
    runtime=ROOT/'artifacts'/'assistant_runtime_status.json'
    try:
        status=json.loads(runtime.read_text(encoding='utf-8'))
        stamp=datetime.fromisoformat(status['updated_at'])
        waiting=status.get('running') is True and any(word in status.get('worker_status','') for word in ('失去焦点','等待远程画面恢复'))
        state_path=folder/'observed_wait.json'
        state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        observed_wait=state.get('seconds',0)
        if state.get('waiting') and waiting and state.get('pid')==status['pid']:
            observed_wait+=max(0,(stamp-datetime.fromisoformat(state['sampled_at'])).total_seconds())
        state_path.write_text(json.dumps({'seconds':observed_wait,'waiting':waiting,'pid':status['pid'],
                                         'sampled_at':stamp.isoformat()}),encoding='utf-8')
    except (OSError,ValueError,KeyError,TypeError):pass
    report={'started_at':manifest['started_at'],'finished_at':manifest.get('finished_at'),'elapsed_seconds':elapsed,'events':dict(events),
        'observed_focus_wait_seconds':observed_wait,'observed_nonwaiting_seconds':max(0,elapsed-observed_wait),
        'submitted_count':len(submitted),'verified_count':sum(task['server_verified'] for task in submitted),
        'incomplete_submissions':[task['session'] for task in submitted if task['missing']],
        'inconsistent_submissions':[task['session'] for task in submitted if task['consistent'] is not True],
        'recovery_reasons':dict(Counter(row.get('reason','') for row in rows if row['event']=='continuous_recovery')),
        'failure_events':failures,
        'health_alerts':alerts,
        'health_alert_levels':dict(Counter(row['alert']['level'] for row in alerts if row.get('alert'))),
        'gaze_zones':dict(zones),'transcript_follow_count':events['transcript_follow'],'tasks':tasks,
        'last_event':rows[-1] if rows else None,'build_unchanged':all(sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in manifest['code_sha256'].items())}
    (folder/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--start',action='store_true');parser.add_argument('folder',type=Path)
    args=parser.parse_args();args.folder.mkdir(parents=True,exist_ok=True)
    if args.start:
        manifest={'started_at':datetime.now().astimezone().isoformat(),'requested_minutes':30,'minimum_submissions':10,
                  'code_sha256':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in CODE}}
        (args.folder/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report=inspect(args.folder)
    print(json.dumps({key:report[key] for key in ('elapsed_seconds','submitted_count','verified_count','incomplete_submissions',
        'inconsistent_submissions','recovery_reasons','gaze_zones','transcript_follow_count','build_unchanged','last_event')},ensure_ascii=True))
