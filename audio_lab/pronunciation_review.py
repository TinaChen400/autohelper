"""One bounded, separate decode of existing response WAVs; never replay UI."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def review(folder):
    if __package__:
        from .torch_asr import TorchTranscriber,read_wav
    else:
        from torch_asr import TorchTranscriber,read_wav
    began=time.monotonic();engine=TorchTranscriber()
    result={'state':'running','transcripts':{},'details':{},'method':'separate_beam3_decode',
            'started_at':datetime.now().astimezone().isoformat()}
    for name in ('A','B'):
        audio,rate=read_wav(folder/f'{name}.wav')
        decoded=engine.transcribe(audio,rate,beams=3)
        result['transcripts'][name]=decoded['text'];result['details'][name]=decoded
    result.update(state='complete',elapsed_seconds=time.monotonic()-began)
    (folder/'pronunciation_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    args=parser.parse_args()
    try:review(args.folder)
    except Exception as exc:
        (args.folder/'pronunciation_review.json').write_text(json.dumps({'state':'failed','reason':str(exc)}),encoding='utf-8')
        raise
