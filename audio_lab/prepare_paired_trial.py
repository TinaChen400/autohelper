"""Reviewed bounds for the first paired trial; not an automatic segmenter."""
import json
from pathlib import Path
import wave

p=Path(__file__).resolve().parent/'sessions'/'paired_20261011_203027_238752'
bounds={'Input':[11.0,19.33],'A':[19.33,24.15],'B':[24.15,29.1]}
with wave.open(str(p/'whole.wav')) as source:
    rate=source.getframerate(); channels=source.getnchannels()
    raw=source.readframes(source.getnframes())
for name,(a,b) in bounds.items():
    with wave.open(str(p/f'{name}.wav'),'wb') as out:
        out.setnchannels(channels); out.setsampwidth(2); out.setframerate(rate)
        out.writeframes(raw[round(a*rate)*channels*2:round(b*rate)*channels*2])
target="And uh, I thought we did a really good job of that and I'm really proud of that first half."
(p/'target.txt').write_text(target,encoding='utf-8')
(p/'clip_manifest.json').write_text(json.dumps({'bounds_seconds':bounds,
    'boundary_method':'page progress checked against 10ms audio RMS; cuts in inter-clip quiet gaps',
    'target_text':target,'target_source':'screen/00041_A.png visual transcription',
    'complete_confirmed':False,'note':'B ends near saved file end; no final idle screenshot; listening review required'},indent=2),encoding='utf-8')
print('Three clips prepared')
