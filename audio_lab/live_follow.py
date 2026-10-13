"""Independent partial-audio recognition for cursor guidance; no UI input."""
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np
from torch_asr import TorchTranscriber


def run(folder):
    engine=TorchTranscriber(threads=2)
    previous=None
    while not (folder/'STOP').exists() and not (folder/'FOLLOW_STOP').exists():
        try:
            request=json.loads((folder/'follow_request.json').read_text(encoding='utf-8'))
            if request['sequence']==previous:time.sleep(.15);continue
            previous=request['sequence']
            with wave.open(str(folder/'whole.wav')) as source:
                rate=source.getframerate();channels=source.getnchannels()
                start=max(0,int(request['start_seconds']*rate))
                stop=min(source.getnframes(),int(request['end_seconds']*rate))
                if stop-start<rate:time.sleep(.15);continue
                source.setpos(start)
                raw=source.readframes(stop-start)
            audio=np.frombuffer(raw,dtype='<i2').reshape(-1,channels).astype(np.float32).mean(axis=1)/32768
            if np.sqrt(np.mean(audio**2))<.003:continue
            result=engine.transcribe(audio,rate)
            payload={**request,'text':result['text'],'decoded_at':time.monotonic(),
                     'source':'live_recorded_audio_asr','audio_end_seconds':stop/rate}
            temporary=folder/'live_follow.tmp'
            temporary.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
            temporary.replace(folder/'live_follow.json')
        except (OSError,ValueError,KeyError,wave.Error):time.sleep(.2)


if __name__=='__main__':run(Path(sys.argv[1]))
