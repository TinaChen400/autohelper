"""Incremental transcription of the active loopback session, without UI input."""
import json
from pathlib import Path
import time
import numpy as np
from gpu_runtime import cuda_paths

ROOT = Path(__file__).resolve().parent

def main():
    folder = Path(json.loads((ROOT/'sessions'/'active_live.json').read_text())['folder'])
    output = folder/'live_analysis.json'
    report = {'state': 'loading_model', 'chunks': [], 'ratings_applied': False,
              'note': '实时转写不等于评价；尚未确认 Input/A/B 边界。'}
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    save()
    handles = cuda_paths()
    from faster_whisper import WhisperModel
    engine = WhisperModel(str(ROOT/'models'/'whisper-turbo'), device='cuda', compute_type='float16', local_files_only=True)
    report['state'] = 'waiting_for_audio'
    save()
    processed = 0
    while True:
        # Read PCM directly because the running WAV writer finalizes its header only on stop.
        with (folder/'whole.wav').open('rb') as stream:
            stream.seek(44)
            pcm = stream.read()
        count = len(pcm)//4
        status = json.loads((folder/'capture_status.json').read_text(encoding='utf-8'))
        ended = status['state'] in ('saved', 'failed')
        if count-processed >= 48000*5 or (ended and count > processed):
            start = max(0, processed-48000)
            samples = np.frombuffer(pcm[start*4:count*4], dtype='<i2').reshape(-1,2).mean(axis=1).astype(np.float32)/32768
            # Exact 48 kHz -> 16 kHz anti-aliased resampling.
            from scipy.signal import resample_poly
            audio = resample_poly(samples, 1, 3)
            if float(np.max(np.abs(audio), initial=0)) > .001:
                began = time.monotonic()
                segments, info = engine.transcribe(audio, beam_size=5, vad_filter=True)
                texts = [{'start': start/48000+s.start, 'end': start/48000+s.end, 'text': s.text} for s in segments]
                report['chunks'].append({'start': start/48000, 'end': count/48000, 'segments': texts, 'processing_seconds': time.monotonic()-began})
                report['state'] = 'analyzing_audio'
            else:
                report['state'] = 'waiting_for_audio'
            processed = count
            report['processed_seconds'] = count/48000
            save()
        if ended:
            report['state'] = 'finished'
            save()
            break
        time.sleep(1)

if __name__ == '__main__':
    main()
