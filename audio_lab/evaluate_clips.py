"""Run local models on reviewed Input/A/B clips; never fills a remote form."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import wave
import numpy as np

ROOT=Path(__file__).resolve().parent


def new_report(manifest):
    return {'manifest':manifest,'transcripts':{},'quality':{},'speaker_similarity':{},
            'timings':{},'models':{},'ratings_applied':False,'submitted':False,
            'limitations':['ASR差异不等于发音错误。','本地模型候选尚未经本任务人工标注校准。']}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    parser.add_argument('--stage',choices=('all','asr','quality','speaker'),default='all')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    folder=args.folder.resolve()
    manifest=json.loads((folder/'clip_manifest.json').read_text(encoding='utf-8'))
    report={'manifest':manifest,'transcripts':{},'quality':{},'speaker_similarity':{},
            'timings':{},'ratings_applied':False,'submitted':False,
            'limitations':['音画传输延迟尚未校准，切分需回听核对。','ASR差异不等于发音错误。',
                           '音质和音色分数未经本任务人工标注校准，不代表准确率。',
                           '未使用自然度或整体偏好模型。']}
    output=args.output or folder/'model_comparison.json'
    if args.output:
        report=new_report(manifest)
    elif (folder/'model_comparison.json').exists():
        report=json.loads((folder/'model_comparison.json').read_text(encoding='utf-8'))
    for stage,field in {'asr':'transcripts','quality':'quality','speaker':'speaker_similarity'}.items():
        if args.stage in ('all',stage): report[field]={}
    if args.stage in ('all','asr'):
        for field in ('transcription_backend','transcription_details'):
            report.pop(field,None)
        for key in list(report['timings']):
            if key.startswith(('whisper_','asr_')): report['timings'].pop(key)
        report.setdefault('models',{}).pop('transcription',None)
    def save(): output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    clips={}
    for name in manifest['bounds_seconds']:
        with wave.open(str(folder/f'{name}.wav')) as source:
            rate=source.getframerate()
            audio=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2').reshape(-1,source.getnchannels()).astype(np.float32).mean(axis=1)/32768
            clips[name]=(audio,rate)
    if args.stage in ('all','asr'):
        transcribe(folder,report,save,tuple(clips))
        if args.stage!='all': return
    if args.stage in ('all','quality'):
        score_quality(clips,report,save)
        if args.stage!='all': return
    if 'Input' in clips:
        score_speaker(clips,report,save)
    else:
        report['speaker_applicable']=False
        save()


def transcribe(folder,report,save,order=('Input','A','B')):
    from torch_asr import transcribe_session
    transcribe_session(folder,report,save,order)


def score_quality(clips,report,save):
    report.setdefault('models',{})['quality']='SigMOS V1'
    spec=importlib.util.spec_from_file_location('sigmos_local',ROOT/'models'/'sigmos'/'sigmos.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    quality=module.SigMOS(str(ROOT/'models'/'sigmos'))
    for name,(audio,rate) in clips.items():
        t=time.perf_counter()
        report['quality'][name]=quality.run(audio,sr=rate)
        report['timings'][f'sigmos_{name}']=time.perf_counter()-t
        save()


def score_speaker(clips,report,save):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    from scipy.signal import resample_poly
    torch.set_num_threads(4)
    encoder=EncoderClassifier.from_hparams(source=str(ROOT/'models'/'ecapa'),savedir=str(ROOT/'models'/'ecapa-runtime'),
        overrides={'pretrained_path':(ROOT/'models'/'ecapa').as_posix()},run_opts={'device':'cpu'},local_strategy=LocalStrategy.COPY)
    embeddings={}
    for name,(audio,rate) in clips.items():
        t=time.perf_counter()
        from math import gcd
        divisor=gcd(rate,16000)
        resampled=resample_poly(audio,16000//divisor,rate//divisor).copy()
        embeddings[name]=encoder.encode_batch(torch.from_numpy(resampled).unsqueeze(0)).reshape(1,-1)
        report['timings'][f'ecapa_{name}']=time.perf_counter()-t
    report['speaker_similarity']={n:float(torch.nn.functional.cosine_similarity(embeddings['Input'],embeddings[n]).item()) for n in ('A','B')}
    report.setdefault('models',{})['speaker']='ECAPA VoxCeleb'
    save()
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
