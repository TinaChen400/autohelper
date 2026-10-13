"""Measure offline ASR on recorded clips without starting remote interaction."""
import argparse
import json
from pathlib import Path
import sys
import time
from torch_asr import TorchTranscriber,read_wav


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('clips',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    began=time.perf_counter()
    engine=TorchTranscriber()
    results={'load_seconds':time.perf_counter()-began,'clips':[]}
    for path in args.clips:
        began=time.perf_counter()
        audio,rate=read_wav(path)
        result=engine.transcribe(audio,rate)
        result.update(path=str(path),elapsed_seconds=time.perf_counter()-began)
        results['clips'].append(result)
        print(json.dumps(result,ensure_ascii=True),flush=True)
    results['blocked_dependencies_loaded']=[name for name in ('av','ctranslate2') if name in sys.modules]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
