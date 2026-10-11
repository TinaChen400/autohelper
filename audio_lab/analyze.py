"""Independent WAV checks, optional local ASR and a human-review draft."""
import argparse
import html
import json
import math
from pathlib import Path
import re
import wave
import numpy as np


def wav_metrics(path):
    with wave.open(str(path),'rb') as source:
        rate, channels, count = source.getframerate(), source.getnchannels(), source.getnframes()
        if source.getsampwidth() != 2: raise ValueError('第一版仅支持 PCM 16-bit WAV，请先转换文件。')
        data = np.frombuffer(source.readframes(count),dtype='<i2').astype(np.float64)/32768
    if not count: raise ValueError(f'空录音：{path.name}')
    data = data.reshape(-1,channels)
    rms = float(np.sqrt(np.mean(data**2)))
    return {'duration_seconds':count/rate,'sample_rate':rate,'channels':channels,
            'peak_dbfs':20*math.log10(max(float(np.max(np.abs(data))),1e-12)),
            'rms_dbfs':20*math.log10(max(rms,1e-12)),
            'near_full_scale_fraction':float(np.mean(np.abs(data)>=.999)),
            'low_amplitude_sample_fraction':float(np.mean(np.max(np.abs(data),axis=1)<.003)),
            'note':'仅录音诊断指标；低幅度比例不等于语音停顿率，不能作为音质或自然度分数。'}


def words(text): return re.findall(r"\w+(?:['’]\w+)?",text.lower(),flags=re.UNICODE)


def word_error(reference,hypothesis):
    a,b = words(reference),words(hypothesis)
    if not a: return None
    row = list(range(len(b)+1))
    for i,word in enumerate(a,1):
        next_row = [i]
        for j,other in enumerate(b,1):
            next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(word!=other)))
        row = next_row
    return row[-1]/len(a)


def analyze(folder, model=None, device='cpu', reference=None):
    folder = Path(folder)
    result = {'stage':'offline_human_review','ratings_applied':False,'submitted':False,'clips':{},
              'target_text':reference,'limitations':['未自动识别录音边界或音频是否完整','转写差异不是发音错误的确证','尚未加载音色相似度、音质或自然度评分模型']}
    engine = None
    if model:
        from faster_whisper import WhisperModel
        engine = WhisperModel(str(Path(model).resolve(strict=True)),device=device,
                              compute_type='int8' if device=='cpu' else 'float16',local_files_only=True)
    for name in ('Input','A','B'):
        path = folder/f'{name}.wav'
        metrics = wav_metrics(path)
        item = {'file':path.name,'recording_checks':metrics,'transcript':None,'target_word_error_rate':None}
        if engine:
            segments,info = engine.transcribe(str(path),beam_size=5,vad_filter=True)
            item['segments'] = [{'start':s.start,'end':s.end,'text':s.text} for s in segments]
            item['transcript'] = ''.join(s['text'] for s in item['segments']).strip()
            item['language'] = info.language
            if reference and name in ('A','B'):
                item['target_word_error_rate'] = word_error(reference,item['transcript'])
        result['clips'][name] = item
    result['evaluation_draft'] = {name:{'choice':None,'reason':None,'status':'needs_human_review'} for name in
        ('Audio and Recording Quality','Persona Likeness (Speaker Similarity)','Pronunciation Faithfulness','Naturalness','Overall Preference','Dominant Factor')}
    output = folder/'analysis.json'
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    blocks = []
    for name,item in result['clips'].items():
        transcript = item['transcript'] or '未运行转写模型'
        blocks.append(f'<section><h2>{name}</h2><audio controls src="{name}.wav"></audio><p>{html.escape(transcript)}</p><pre>{html.escape(json.dumps(item["recording_checks"],indent=2,ensure_ascii=False))}</pre></section>')
    page = '<!doctype html><meta charset="utf-8"><title>独立音频对比 · 待人工评价</title><style>body{font:16px sans-serif;max-width:1000px;margin:30px auto;background:#f7f8fa}section{padding:20px;background:white;margin:12px 0}pre{white-space:pre-wrap}audio{width:95%}</style><h1>Input / A / B 独立对比</h1><p>这是诊断和转写报告；没有模型评分，没有填写远程页面。</p>'
    if reference: page += '<h2>页面目标文本</h2><p>'+html.escape(reference)+'</p>'
    page += ''.join(blocks)+'<p>请在 analysis.json 的 evaluation_draft 中记录你的选择和理由，保留原始录音以便复核。</p>'
    (folder/'report.html').write_text(page,encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('session',type=Path)
    parser.add_argument('--model',type=Path,help='已下载的 faster-whisper 模型本地目录，不自动下载')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--reference',type=Path,help='本题页面要求朗读的目标文本 UTF-8 文件；不使用 Input 转写代替')
    args = parser.parse_args()
    analyze(args.session,args.model,args.device,args.reference.read_text(encoding='utf-8') if args.reference else None)
    print(args.session/'report.html')
