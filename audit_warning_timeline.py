"""Correlate persisted warning transitions with task events and capture results."""
from collections import Counter
from datetime import datetime, timedelta
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read_rows(path):
    try:lines=path.read_text(encoding='utf-8',errors='replace').splitlines()
    except OSError:return []
    result=[]
    for line in lines:
        try:result.append(json.loads(line))
        except ValueError:pass
    return result

def read_json(path):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}

def stamp(value):return datetime.fromisoformat(value).replace(tzinfo=None)

def audit():
    alerts=read_rows(ROOT/'artifacts/health_alerts.jsonl')
    end=max(stamp(r['time']) for r in alerts)
    start=end-timedelta(minutes=35)
    alerts=[r for r in alerts if start<=stamp(r['time'])<=end and r.get('alert')]
    events=[]
    for path in (ROOT/'artifacts').glob('session_20261013_*/events.jsonl'):
        events.extend({**r,'session':path.parent.name} for r in read_rows(path))
    events.sort(key=lambda r:r['time'])
    selected=[r for r in events if start<=stamp(r['time'])<=end]
    contexts=[]
    for alert in alerts:
        task=alert.get('task_id')
        if not task:continue
        related=[r for r in events if r.get('task_id')==task]
        before=[r for r in related if stamp(r['time'])<=stamp(alert['time'])]
        folder=ROOT/'audio_lab/sessions'/next((r['recording_session'] for r in related if r.get('recording_session')),'missing')
        capture=read_json(folder/'capture_status.json');pipeline=read_json(folder/'pipeline_status.json')
        significant=[r for r in before if r['event'] not in ('manual_input_pause','mouse_gaze','mouse_park','transcript_follow','playback_sample','player_durations')]
        contexts.append({'time':alert['time'],'task_id':task,'alert':alert['alert'],
            'capture':{k:capture.get(k) for k in ('state','seconds','pcm_peak','nonzero_samples_total')},
            'pipeline':{k:pipeline.get(k) for k in ('state','completed_at','elapsed_seconds','stage_errors')},
            'last_events':significant[-5:]})
    tasks={r['task_id'] for r in selected if r.get('task_id')}
    summary={'start':start.isoformat(),'end':end.isoformat(),'tasks_with_events':len(tasks),
        'warning_transitions':sum(r['alert']['level']=='warning' for r in alerts),
        'error_transitions':sum(r['alert']['level']=='error' for r in alerts),
        'titles':dict(Counter(r['alert']['title'] for r in alerts)),
        'submitted_tasks':len({r['task_id'] for r in selected if r['event']=='auto_submit_clicked'}),
        'next_question_tasks':len({r['task_id'] for r in selected if r['event']=='next_question_ready'}),
        'failures':dict(Counter(r.get('reason','') for r in selected if r['event'] in ('continuous_recovery','capture_repair_pending','first_round_retry','review_pending')))}
    report={'summary':summary,'contexts':contexts}
    (ROOT/'artifacts/warning_timeline_20261013.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for row in contexts:
        print(json.dumps({'time':row['time'],'alert':row['alert']['title'],'capture':row['capture'],
            'pipeline':row['pipeline'],'last_events':[{k:r.get(k) for k in ('time','event','reason','phase','result')} for r in row['last_events']]},ensure_ascii=False))

if __name__=='__main__':audit()
