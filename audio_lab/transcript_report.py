"""Readable transcript evidence, shared by the UI and offline inspection."""
import json
from pathlib import Path


def transcript_report(folder):
    folder=Path(folder)
    path=folder/'model_comparison.json'
    try:
        report=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return '本题转写结果尚未生成。首轮完整播放结束后会自动分析。'
    backend=report.get('transcription_backend',{})
    lines=['录音：'+folder.name,
           '转写模型：'+backend.get('model','尚未加载'),
           '运行设备：'+backend.get('device','未知'),
           '说明：以下文字是语音识别结果，识别差异不等于发音错误；需核对页面目标正文。','']
    for name in report.get('manifest',{}).get('bounds_seconds',{}):
        detail=report.get('transcription_details',{}).get(name,{})
        elapsed=report.get('timings',{}).get('asr_'+name)
        timing=f'；转写 {elapsed:.2f} 秒' if elapsed is not None else ''
        duration=detail.get('source_seconds')
        length=f'（录音 {duration:.2f} 秒{timing}）' if duration is not None else ''
        lines.extend([name+length,report.get('transcripts',{}).get(name,'等待转写结果'), ''])
    try:
        status=json.loads((folder/'pipeline_status.json').read_text(encoding='utf-8'))
        error=status.get('stage_errors',{}).get('asr')
        if error: lines.extend(['转写异常：'+error,'其他维度结果仍单独保留。'])
    except (OSError,ValueError): pass
    return '\n'.join(lines)
