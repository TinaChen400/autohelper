"""UTMOSv2 naturalness trial. Downloads official weights only when explicitly run."""
import argparse
import json
import os
from pathlib import Path
import time
import wave
import numpy as np

ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'models'/'utmos-hf')
os.environ['TORCH_HOME']=str(ROOT/'models'/'utmos-torch')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('folder',type=Path)
    parser.add_argument('--responses-only',action='store_true')
    args=parser.parse_args()
    import torch
    import utmosv2
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    t=time.perf_counter()
    checkpoint=hf_hub_download('sarulab-speech/UTMOSv2','fold0_s42_best_model.pth')
    model=utmosv2.create_model(pretrained=True,checkpoint_path=checkpoint,device='cpu')
    report={'model':'sarulab-speech/UTMOSv2','config':'fusion_stage3','fold':0,
            'device':'cpu','load_and_download_seconds':time.perf_counter()-t,
            'scores':{},'ratings_applied':False,'note':'自然度MOS预测；不能定位重音错误，未按当前任务校准。'}
    for name in (('A','B') if args.responses_only else ('Input','A','B')):
        with wave.open(str(args.folder/f'{name}.wav')) as source:
            rate=source.getframerate()
            x=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2').reshape(-1,source.getnchannels()).astype(np.float32).mean(axis=1)/32768
        t=time.perf_counter()
        result=model.predict(data=x,sr=rate,device='cpu',num_workers=0,num_repetitions=3,remove_silent_section=False,verbose=False)
        report['scores'][name]={'mos':float(np.asarray(result).reshape(-1)[0]),'seconds':time.perf_counter()-t}
        (args.folder/'naturalness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
