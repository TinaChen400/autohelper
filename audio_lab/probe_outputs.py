"""Read-only output loopback diagnostic. Does not change Windows settings."""
import argparse
import json
import numpy as np
import soundcard as sc


def probe(seconds=2):
    result=[]
    default=sc.default_speaker()
    for speaker in sc.all_speakers():
        item={'device':speaker.name,'id':speaker.id,'default':bool(default and speaker.id==default.id)}
        try:
            microphone=sc.get_microphone(id=speaker.id,include_loopback=True)
            with microphone.recorder(samplerate=48000,channels=2) as source:
                block=source.record(numframes=round(48000*seconds))
            item.update(peak=float(np.max(np.abs(block))),rms=float(np.sqrt(np.mean(block**2))),
                        nonzero_samples=int(np.count_nonzero(block)))
        except Exception as exc:item['error']=str(exc)
        result.append(item)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seconds',type=float,default=2)
    args=parser.parse_args()
    if not .1<=args.seconds<=5:parser.error('seconds must be between 0.1 and 5')
    print(json.dumps(probe(args.seconds),ensure_ascii=True,indent=2))
