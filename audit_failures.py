"""Read-only daily fault audit, with attempts grouped by task identity."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent

def category(reason):
    for key,words in (
        ('capture_no_signal',('未采集到声音','全零','录音进程退出','录音设备')),
        ('capture_boundary',('声音间隙','末尾证据','切片长度','音频段顺序')),
        ('capture_clock',('计时与画面','起始计时','有效播放进度','录音完整性','总时长变化','可靠读取 Input 的计时','起点和总时长')),
        ('target_text',('目标正文','完整正文','正文有未读清')),
        ('focus_or_occlusion',('焦点','遮挡','最小化','窗口尺寸','窗口移动')),
        ('identity_or_layout',('题号','题目编号','本题身份','题目内容或布局','播放器位置','题目身份','画面变化','同题核对','定位音频区','未能确认题型','题目已变化','评价区域变化')),
        ('playback_pause',('确认全部暂停',)),
        ('rating_incomplete',('漏项','未看清评价','选齐','未确认','评价尚未','核对齐')),
        ('review_unresolved',('复核','无可靠结论','无法判断')),
        ('user_stop',('Esc','停止按钮','已取消'))):
        if any(word in reason for word in words):return key
    return 'other'

def audit(day):
    faults=[];invalid=[];submitted=set();verified=set();tasks=defaultdict(set)
    for path in sorted((ROOT/'artifacts').glob(f'session_{day}_*/events.jsonl')):
        for number,line in enumerate(path.read_text(encoding='utf-8',errors='replace').splitlines(),1):
            try:row=json.loads(line)
            except ValueError:invalid.append({'file':str(path),'line':number});continue
            task=row.get('task_id',path.parent.name)
            tasks[task].add(path.parent.name)
            event=row.get('event')
            if event=='auto_submit_clicked':submitted.add(task)
            if event=='next_question_ready':verified.add(task)
            # Count each outer recovery once; inner validation/retry entries
            # are supporting evidence, not additional failed tasks.
            if event in ('continuous_recovery','review_pending','capture_repair_pending'):
                reason=row.get('reason','')
                faults.append({'time':row.get('time'),'task_id':task,'session':path.parent.name,
                    'event':event,'category':category(reason),'reason':reason})
    grouped={key:{'recovery_events':sum(row['category']==key for row in faults),
        'distinct_tasks':len({row['task_id'] for row in faults if row['category']==key}),
        'reasons':dict(Counter(row['reason'] for row in faults if row['category']==key))}
        for key in sorted({row['category'] for row in faults})}
    return {'day':day,'note':'Recovery events are not unique failed questions; category is a text-based audit label, not a proven root cause.',
        'submitted_tasks':len(submitted),'verified_submissions':len(submitted&verified),
        'categories':grouped,'multi_attempt_tasks':{k:sorted(v) for k,v in tasks.items() if len(v)>1},
        'faults':faults,'invalid_event_lines':invalid}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('day');args=parser.parse_args()
    result=audit(args.day);out=ROOT/'artifacts'/f'failure_audit_{args.day}.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('day','submitted_tasks','verified_submissions','categories','multi_attempt_tasks')},ensure_ascii=True))
