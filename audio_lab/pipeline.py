"""Offline worker launched after the first complete recorded round."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parent


def merge_stage_reports(folder, manifest, stages):
    """Merge only each isolated worker's owned fields; never inherit old votes."""
    if __package__:
        from .evaluate_clips import new_report
    else:
        from evaluate_clips import new_report
    report=new_report(manifest)
    fields={'asr':('transcripts','transcription_backend','transcription_details'),
            'quality':('quality',),'speaker':('speaker_similarity','speaker_applicable')}
    for stage in stages:
        path=folder/f'{stage}_report.json'
        if stage not in fields or not path.exists():continue
        part=json.loads(path.read_text(encoding='utf-8'))
        for field in fields[stage]:
            if field in part:report[field]=part[field]
        report['timings'].update(part.get('timings',{}))
        report['models'].update(part.get('models',{}))
    if manifest.get('task_type')=='ab_only':report['speaker_applicable']=False
    (folder/'model_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def run_stage(folder, stage, script, options):
    began=time.monotonic()
    log_path=folder/f'{stage}.log'
    environment=os.environ.copy()
    environment.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONIOENCODING='utf-8',
                       OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if stage!='naturalness':options=[*options,'--output',str(folder/f'{stage}_report.json')]
    with log_path.open('w',encoding='utf-8') as log:
        try:
            result=subprocess.run([sys.executable,str(ROOT/script),str(folder),*options],
                                  stdout=log,stderr=subprocess.STDOUT,env=environment,timeout=300)
        except subprocess.TimeoutExpired:
            log.write(f'\n[worker_timeout stage={stage} seconds=300]\n')
            return time.monotonic()-began,'模型阶段超过 300 秒，等待助手重新生成；详见 '+log_path.name
        log.write(f'\n[worker_exit stage={stage} code={result.returncode} hex={result.returncode & 0xffffffff:08x}]\n')
    return time.monotonic()-began, describe_failure(log_path) if result.returncode else None


def describe_failure(log_path):
    detail=log_path.read_text(encoding='utf-8',errors='replace')
    if '应用程序控制策略' in detail or 'application control policy' in detail.lower() or '\\u5e94\\u7528\\u7a0b\\u5e8f\\u63a7\\u5236' in detail:
        return 'Windows 应用控制阻止模型依赖加载；详见 '+log_path.name
    lines=[line.strip() for line in detail.splitlines() if line.strip()]
    return (lines[-1][:220] if lines else '分析进程异常退出')+'；详见 '+log_path.name

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path); args=parser.parse_args()
    folder=args.folder.resolve()
    began=time.monotonic()
    started_at=datetime.now().astimezone().isoformat()
    timings={}
    errors={}
    def status(state,**extra):
        if state in ('draft_ready','partial_ready','needs_review'):
            extra['completed_at']=datetime.now().astimezone().isoformat()
        (folder/'pipeline_status.json').write_text(json.dumps({'state':state,'analysis_started_at':started_at,
            'elapsed_seconds':time.monotonic()-began,'stage_timings':timings,**extra},ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        (folder/'evaluation_decisions.json').unlink(missing_ok=True)
        status('segmenting')
        if __package__:
            from .segment_session import segment
        else:
            from segment_session import segment
        manifest=segment(folder) or {}
        timings['segmenting']=time.monotonic()-began
        stages=[('naturalness','naturalness_trial.py',['--responses-only']),
                ('asr','evaluate_clips.py',['--stage','asr']),
                ('quality','evaluate_clips.py',['--stage','quality']),
                ('speaker','evaluate_clips.py',['--stage','speaker'])]
        stages=[item for item in stages if not (item[0]=='speaker' and manifest.get('task_type')=='ab_only')]
        for stage,script,options in stages:
            (folder/('naturalness.json' if stage=='naturalness' else f'{stage}_report.json')).unlink(missing_ok=True)
        # Two independent workers retain the same models and repetitions, but
        # overlap inference. Separate reports prevent concurrent JSON writes.
        status('analyzing_parallel',stage_errors=errors)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(run_stage,folder,*item):item[0] for item in stages}
            for future in as_completed(futures):
                stage=futures[future]
                try:
                    elapsed,error=future.result()
                    timings['analyzing_'+stage]=elapsed
                    if error:errors[stage]=error
                except Exception as exc:errors[stage]=str(exc)
                status('analyzing_parallel',stage_errors=errors,completed_stages=list(timings))
        merge_stage_reports(folder,manifest,[item[0] for item in stages])
        status('deciding',stage_errors=errors)
        decision_started=time.monotonic()
        try:
            if __package__:
                from .decision_engine import build_decisions
            else:
                from decision_engine import build_decisions
            decisions=build_decisions(folder)
            if decisions['judgments']['pronunciation']['judgment']=='需复核':
                status('reviewing_pronunciation',stage_errors=errors)
                review_log=folder/'pronunciation_review.log'
                with review_log.open('w',encoding='utf-8') as output:
                    result=subprocess.run([sys.executable,str(ROOT/'pronunciation_review.py'),str(folder)],
                        stdout=output,stderr=subprocess.STDOUT,timeout=150)
                if result.returncode==0:build_decisions(folder)
        except Exception as exc:
            errors['decisions']=str(exc)
        timings['deciding']=time.monotonic()-decision_started
        expected_stages=4 if manifest and manifest.get('task_type')=='ab_only' else len(stages)+1
        state='draft_ready' if not errors else 'partial_ready' if len(errors)<expected_stages else 'needs_review'
        status(state,stage_errors=errors,reason='；'.join(f'{name}: {reason}' for name,reason in errors.items()),
               report=str(folder/'model_comparison.json'),naturalness=str(folder/'naturalness.json'),
               decisions=str(folder/'evaluation_decisions.json'),
               note='逐项自动候选和依据已保存；界面流程负责逐项核对与提交，用户可以修改或提前提交。')
    except Exception as exc:
        status('needs_review',reason=str(exc))

if __name__=='__main__': main()
