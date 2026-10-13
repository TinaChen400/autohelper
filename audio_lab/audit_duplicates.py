"""Read-only audio repeat audit, allowing recording offset and silence."""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np


def load(path):
    with wave.open(str(path),'rb') as wav:
        rate=wav.getframerate();channels=wav.getnchannels();width=wav.getsampwidth()
        if width!=2:raise ValueError(f'Unsupported PCM width: {width}')
        signal=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float64)
    signal=signal.reshape(-1,channels).mean(axis=1)
    # Average bins before reducing the rate; keep waveform polarity.
    step=max(1,rate//4000);signal=signal[:len(signal)//step*step].reshape(-1,step).mean(axis=1)
    rate=rate/step
    block=max(1,round(rate*.02));count=len(signal)//block
    rms=np.sqrt((signal[:count*block].reshape(-1,block)**2).mean(axis=1))
    active=np.flatnonzero(rms>max(4,rms.max()*.025))
    if not len(active):return None
    signal=signal[max(0,(active[0]-5)*block):min(len(signal),(active[-1]+6)*block)]
    return signal-signal.mean(),rate


def compare(a,b):
    if a is None or b is None:return None
    x,rx=a;y,ry=b
    if rx!=ry or min(len(x),len(y))/max(len(x),len(y))<.8:return 0
    size=1<<(len(x)+len(y)-1).bit_length()
    corr=np.fft.irfft(np.fft.rfft(x,size)*np.fft.rfft(y[::-1],size),size)[:len(x)+len(y)-1]
    # Search only shifts of up to 0.75 seconds, with >=80% shared content.
    shifts=np.arange(-round(rx*.75),round(rx*.75)+1)
    start=np.maximum(0,shifts);ystart=np.maximum(0,-shifts)
    length=np.minimum(len(x)-start,len(y)-ystart)
    valid=length>=min(len(x),len(y))*.8
    shifts,start,ystart,length=[v[valid] for v in (shifts,start,ystart,length)]
    ex=np.r_[0,np.cumsum(x*x)];ey=np.r_[0,np.cumsum(y*y)]
    denom=np.sqrt((ex[start+length]-ex[start])*(ey[ystart+length]-ey[ystart]))
    scores=np.abs(corr[len(y)-1+shifts])/np.maximum(denom,1)
    return round(float(scores.max()),5)


def main():
    sessions=Path(__file__).resolve().parent/'sessions'
    folders=sorted(sessions.glob('integrated_session_20261012*'))
    originals=[f for f in folders if not any(s in f.name for s in ('_timing_rechecked','_progress_rechecked'))]
    complete=[];incomplete=[];errors=[]
    for f in originals:
        if not all((f/f'{n}.wav').exists() for n in ('Input','A','B')):
            incomplete.append(f.name);continue
        try:
            metadata=json.loads((f/'session.json').read_text(encoding='utf-8')) if (f/'session.json').exists() else {}
            complete.append((f,metadata,{n:load(f/f'{n}.wav') for n in ('Input','A','B')}))
        except Exception as exc:errors.append({'session':f.name,'error':str(exc)})
    repeats=[];partials=[];exact=[]
    for index,(f,meta,audio) in enumerate(complete):
        for g,other,other_audio in complete[index+1:]:
            scores={n:compare(audio[n],other_audio[n]) for n in ('Input','A','B')}
            if any(s is not None and s>=.9 for s in scores.values()):
                item={'first':f.name,'second':g.name,'first_marker':meta.get('question_marker'),
                      'second_marker':other.get('question_marker'),'waveform_correlations':scores}
                (repeats if all(s is not None and s>=.9 for s in scores.values()) else partials).append(item)
            hashes=lambda p:tuple(hashlib.sha256((p/f'{n}.wav').read_bytes()).hexdigest() for n in ('Input','A','B'))
            if hashes(f)==hashes(g):exact.append([f.name,g.name])
    report={'date':'2026-10-12','session_folders':len(folders),'original_sessions':len(originals),
        'excluded_recheck_copies':len(folders)-len(originals),'complete_audio_sessions':len(complete),
        'incomplete_sessions':incomplete,'errors':errors,'three_audio_repeat_pairs':repeats,
        'partial_repeat_pairs':partials,'byte_identical_pairs':exact,
        'method':'All three independently aligned PCM waveforms must correlate >=0.90. Silent audio cannot match. Recheck copies excluded.'}
    path=Path(__file__).resolve().parents[1]/'artifacts'/'duplicate_audit_20261012.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':main()
