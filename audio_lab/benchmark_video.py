"""Reproducible local-model trial on the original recording, never clicks UI."""
import importlib.util
import json
from pathlib import Path
import time
import wave
import shutil
from datetime import datetime
import numpy as np
from gpu_runtime import cuda_paths

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT/'sessions'/'original_video'
MODELS = ROOT/'models'


def main():
    if (FOLDER/'benchmark.json').exists():
        shutil.copy2(FOLDER/'benchmark.json',FOLDER/f'benchmark_previous_{datetime.now():%Y%m%d_%H%M%S}.json')
    handles = cuda_paths()
    with wave.open(str(FOLDER/'source.wav'),'rb') as source:
        rate = source.getframerate()
        samples = np.frombuffer(source.readframes(source.getnframes()),dtype='<i2')
    # Original video is approximately 2 fps. Audio shows quiet gaps after the
    # UI transitions (~1.50 and ~9.52), hence use the gaps at 1.9 and 9.9.
    bounds = {'Input':(0,1.9),'A':(1.9,9.9),'B':(9.9,16.65)}
    for name,(start,end) in bounds.items():
        with wave.open(str(FOLDER/f'{name}.wav'),'wb') as output:
            output.setnchannels(1); output.setsampwidth(2); output.setframerate(rate)
            output.writeframes(samples[round(start*rate):round(end*rate)].astype('<i2').tobytes())
    target = 'No, there is not a dependence between chromatin organization and dorsoventral gene expression in Drosophila.'
    (FOLDER/'target.txt').write_text(target,encoding='utf-8')
    report = {'source':'awesun_20261010090504_-1.mp4','boundaries':bounds,
              'input_is_partial':True,'boundary_method':'visual state transitions plus waveform quiet gaps; manual review required',
              'target_text':target,'timings':{},'transcripts':{},'quality':{},'speaker_similarity':None,
              'ratings_applied':False,'submitted':False}
    report['model_sources'] = {'transcription':'dropbox-dash/faster-whisper-large-v3-turbo',
        'quality':'microsoft/SIG-Challenge ICASSP2024 SigMOS V1',
        'speaker':'speechbrain/spkrec-ecapa-voxceleb'}
    report['execution'] = {'whisper':'CUDA float16','sigmos':'ONNX CPU','ecapa':'PyTorch CPU, 4 threads'}
    from faster_whisper import WhisperModel
    t = time.perf_counter()
    model = WhisperModel(str(MODELS/'whisper-turbo'),device='cuda',compute_type='float16',local_files_only=True)
    report['timings']['whisper_load_seconds'] = time.perf_counter()-t
    for run in range(2):
        rows = {}
        for name in bounds:
            t = time.perf_counter()
            segments,info = model.transcribe(str(FOLDER/f'{name}.wav'),language='en',beam_size=5,vad_filter=True)
            text = ''.join(segment.text for segment in segments).strip()
            rows[name] = {'text':text,'seconds':time.perf_counter()-t}
        report['timings'][f'whisper_run_{run+1}'] = rows
        report['transcripts'] = {n:r['text'] for n,r in rows.items()}
        (FOLDER/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('Whisper',run+1,rows,flush=True)
    spec = importlib.util.spec_from_file_location('sigmos_local',MODELS/'sigmos'/'sigmos.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    t = time.perf_counter(); quality = module.SigMOS(str(MODELS/'sigmos'))
    report['timings']['sigmos_load_seconds'] = time.perf_counter()-t
    for name,(start,end) in bounds.items():
        t = time.perf_counter()
        report['quality'][name] = quality.run(samples[round(start*rate):round(end*rate)].astype(np.float32)/32768,sr=rate)
        report['timings'][f'sigmos_{name}_seconds'] = time.perf_counter()-t
    report['speaker_similarity_status'] = 'reference incomplete: exploratory only, not a reliable vote'
    (FOLDER/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
    from scipy.signal import resample_poly
    torch.set_num_threads(4)
    t = time.perf_counter()
    encoder = EncoderClassifier.from_hparams(source=str(MODELS/'ecapa'),savedir=str(MODELS/'ecapa-runtime'),
        overrides={'pretrained_path':(MODELS/'ecapa').as_posix()},run_opts={'device':'cpu'},local_strategy=LocalStrategy.COPY)
    report['timings']['ecapa_load_seconds'] = time.perf_counter()-t
    embeddings = {}
    for name,(start,end) in bounds.items():
        t = time.perf_counter()
        audio = resample_poly(samples[round(start*rate):round(end*rate)].astype(np.float32)/32768,1,3).copy()
        embeddings[name] = encoder.encode_batch(torch.from_numpy(audio).unsqueeze(0)).reshape(1,-1)
        report['timings'][f'ecapa_{name}_seconds'] = time.perf_counter()-t
    report['speaker_similarity'] = {name:float(torch.nn.functional.cosine_similarity(embeddings['Input'],embeddings[name]).item()) for name in ('A','B')}
    (FOLDER/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
