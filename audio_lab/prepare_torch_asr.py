"""One-time download of the official Distil-Whisper checkpoint, no audio upload."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'models'/'asr-hf')
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'


def main():
    from huggingface_hub import snapshot_download
    repo='distil-whisper/distil-small.en'
    revision='9e4a67ca4569c30be43a3fe7fba1621e504f0093'
    destination=ROOT/'models'/'distil-small.en'
    print('Downloading official model revision '+revision,flush=True)
    snapshot_download(repo,revision=revision,local_dir=destination,
        allow_patterns=['*.json','model.safetensors','*.txt'],max_workers=2)
    (destination/'source.json').write_text(json.dumps({'repo_id':repo,'revision':revision,
        'backend':'transformers-pytorch','audio_upload':False},indent=2),encoding='utf-8')
    print('Model ready: '+str(destination),flush=True)


if __name__=='__main__': main()
