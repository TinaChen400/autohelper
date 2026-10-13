"""Local English WAV transcription using the supported PyTorch Whisper API."""
from math import gcd
import json
from pathlib import Path
import time
import wave
import numpy as np

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'models'/'distil-small.en'


def read_wav(path):
    with wave.open(str(path)) as source:
        if source.getsampwidth()!=2: raise ValueError('转写需要 16-bit PCM WAV。')
        rate=source.getframerate()
        audio=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2')
        audio=audio.reshape(-1,source.getnchannels()).astype(np.float32).mean(axis=1)/32768
    return audio,rate


def resample(audio,rate):
    if rate<=0 or len(audio)==0: raise ValueError('录音为空或采样率无效。')
    if rate==16000: return np.asarray(audio,dtype=np.float32)
    from scipy.signal import resample_poly
    divisor=gcd(rate,16000)
    return resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)


class TorchTranscriber:
    def __init__(self,model_path=MODEL,threads=4):
        import torch
        from transformers import WhisperProcessor,WhisperForConditionalGeneration
        path=Path(model_path)
        if not (path/'model.safetensors').exists():
            raise RuntimeError('本地转写模型未准备好；请先运行 audio_lab/prepare_torch_asr.py。')
        self.torch=torch
        self.source=json.loads((path/'source.json').read_text(encoding='utf-8')) if (path/'source.json').exists() else {}
        torch.set_num_threads(threads)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype=torch.float16 if self.device=='cuda' else torch.float32
        self.processor=WhisperProcessor.from_pretrained(path,local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(path,local_files_only=True,
            use_safetensors=True).to(device=self.device,dtype=self.dtype).eval()

    def transcribe(self,audio,rate,beams=1):
        audio=resample(audio,rate)
        duration=len(audio)/16000
        # No truncation: Whisper's documented sequential long-form generation
        # processes every 30-second window, including the final partial window.
        inputs=self.processor.feature_extractor(audio,sampling_rate=16000,return_tensors='pt',
            truncation=False,padding='longest',return_attention_mask=True)
        inputs={key:value.to(self.device) for key,value in inputs.items()}
        inputs['input_features']=inputs['input_features'].to(dtype=self.dtype)
        with self.torch.inference_mode():
            output=self.model.generate(**inputs,return_timestamps=True,return_segments=True,
                condition_on_prev_tokens=False,num_beams=beams,max_new_tokens=440)
        sequences=output['sequences']
        text=self.processor.batch_decode(sequences,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        segments=[]
        for item in output['segments'][0]:
            segments.append({'start':min(duration,float(item['start'])),'end':min(duration,float(item['end'])),
                'text':self.processor.decode(item['tokens'],skip_special_tokens=True,clean_up_tokenization_spaces=False).strip()})
        return {'text':text,'segments':segments,'source_seconds':duration,
                'backend':'transformers-pytorch','model':'distil-whisper/distil-small.en',
                'device':self.device,'language':'en','truncated_input':False}


def transcribe_session(folder,report,save,order):
    began=time.perf_counter()
    engine=TorchTranscriber()
    report['timings']['asr_load']=time.perf_counter()-began
    report['transcription_backend']={'engine':'transformers-pytorch',
        'model':'distil-whisper/distil-small.en','device':engine.device,'language':'en',
        'revision':engine.source.get('revision'),'offline':True}
    report.setdefault('models',{})['transcription']='distil-whisper/distil-small.en (PyTorch)'
    report['transcription_details']={}
    for name in order:
        began=time.perf_counter()
        audio,rate=read_wav(Path(folder)/f'{name}.wav')
        result=engine.transcribe(audio,rate)
        report['transcripts'][name]=result['text']
        report['transcription_details'][name]=result
        report['timings'][f'asr_{name}']=time.perf_counter()-began
        save()
