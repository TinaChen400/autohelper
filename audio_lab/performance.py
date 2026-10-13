"""Account for wall-clock phases without adding overlapping durations together."""
from datetime import datetime
import json
from pathlib import Path


def write_performance(folder, artifact, completed_at=None):
    folder,artifact=Path(folder),Path(artifact)
    rows=[json.loads(line) for line in (artifact/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    def stamp(value):return datetime.fromisoformat(value).replace(tzinfo=None)
    def first(event):return next((stamp(row['time']) for row in rows if row['event']==event),None)
    def seconds(start,end):return round((end-start).total_seconds(),3) if start and end else None
    if first('play_click') is None:
        path=folder/'performance.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    end=completed_at or datetime.now()
    pipeline=json.loads((folder/'pipeline_status.json').read_text(encoding='utf-8'))
    model_end=stamp(pipeline['completed_at']) if pipeline.get('completed_at') else None
    replay_index=next((i for i,row in enumerate(rows) if row['event']=='replay_plan'),len(rows))
    full_end=next((stamp(row['time']) for row in reversed(rows[:replay_index])
                   if row['event']=='chain_completed'),None)
    # Older event streams/tests may not contain per-player completion events.
    full_end=full_end or first('replay_plan')
    report={'total_until_rating_check_seconds':seconds(first('session_start'),end),
            'startup_location_context_seconds':seconds(first('session_start'),first('play_click')),
            'first_complete_playback_seconds':seconds(first('play_click'),full_end),
            'analysis_seconds':pipeline.get('elapsed_seconds'),
            'model_ready_to_rating_check_seconds':seconds(model_end,end),
            'rating_location_fill_check_seconds':seconds(first('rating_scroll_start'),end),
            'rating_phase_seconds':{phase:round(sum(row['seconds'] for row in rows if
                    row['event']=='rating_phase_timing' and row['phase']==phase),3)
                for phase in ('observer_stop','navigation','recognition','click_and_verify','click_dispatch','selection_verification','recovery')},
            'player_location_attempts':[row for row in rows if row['event']=='player_location_timing'],
            'model_stage_timings':pipeline.get('stage_timings',{}),
            'clicks':[{'time':r['time'],'event':r['event'],'target':r.get('target',r.get('dimension')),
                      'judgment':r.get('judgment'),'click':r.get('click')} for r in rows
                     if r['event'] in ('play_click','model_rating_click')],
            'note':'Wall time; background analysis and supplemental playback overlap. Do not sum overlapping phases.',
            'submitted_by_assistant':False}
    (folder/'performance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
