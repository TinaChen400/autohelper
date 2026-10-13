"""Find similar text and reuse only identical audio with human evidence."""
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from task_tracking import record_relation


def read(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):return {}


def fingerprint(folder):
    context=read(folder/'task_context.json')
    text=re.sub(r'\s+',' ',context.get('target_text','')).strip().casefold()
    order=read(folder/'session.json').get('player_order',[])
    hashes={name:hashlib.sha256((folder/f'{name}.wav').read_bytes()).hexdigest()
            for name in order if (folder/f'{name}.wav').exists()}
    return {'text':text,'instructions':context.get('instructions',''),
            'player_order':order,'audio_sha256':hashes}


def find_history(folder):
    folder=Path(folder);current=fingerprint(folder)
    current_record={**read(folder/'session.json'),**read(folder/'comparison.json')}
    matches=[];reused={}
    if current['text']:
        for prior in folder.parent.iterdir():
            if not prior.is_dir() or prior==folder or not (prior/'comparison.json').exists():continue
            old=fingerprint(prior)
            similarity=SequenceMatcher(None,current['text'],old['text']).ratio()
            if similarity<.85:continue
            exact=(current==old and bool(current['player_order']) and
                   set(current['audio_sha256'])==set(current['player_order']))
            record=read(prior/'comparison.json')
            relation=record_relation(current_record,{**read(prior/'session.json'),**record})
            same_attempt_group=relation in ('same_question_retry','likely_same_question_retry')
            eligible=exact and not same_attempt_group and record.get('submission_observation',{}).get('server_success_verified') is True
            matches.append({'session':prior.name,'text_similarity':round(similarity,4),
                            'identical_text_and_audio':exact,'reusable':eligible,'occurrence_relation':relation,
                            'cross_question_repeat_candidate':exact and not same_attempt_group})
            if eligible and not reused:
                reused={key:{**item,'reason':'复用相同正文与音频的已提交人工答案',
                             'history_session':prior.name}
                        for key,item in record.get('remote_ratings',{}).items()
                        if item.get('source') in ('human_remote','user_submit_confirmation')
                        and item.get('judgment')!='需复核'}
    report={'fingerprint':current,'matches':sorted(matches,key=lambda m:m['text_similarity'],reverse=True)[:20],
            'reused_judgments':reused}
    (folder/'question_history.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report
