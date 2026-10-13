"""Bounded whole-session system-output recording; never controls the click app."""
from datetime import datetime
import json
from pathlib import Path
import time
import wave
import argparse

import numpy as np
try:
    from .durable_io import checkpoint, write_json
except ImportError:
    from durable_io import checkpoint, write_json

ROOT = Path(__file__).resolve().parent


def main():
    import soundcard as sc
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    folder = args.folder or ROOT / 'sessions' / ('live_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True, exist_ok=bool(args.folder))
    status_path = folder / 'capture_status.json'
    speaker = sc.default_speaker()
    if speaker is None:
        raise RuntimeError('No default output device')
    mic = sc.get_microphone(id=speaker.id, include_loopback=True)
    info = {'device': speaker.name, 'device_id':speaker.id, 'folder': str(folder), 'max_seconds': 600,
            'source': 'host_system_output_loopback', 'started_at': datetime.now().astimezone().isoformat(),
            'boundaries_confirmed': False,'pcm_peak':0,'nonzero_samples_total':0,'signal_seconds':0}
    def status(state, **extra):
        info.update(state=state, **extra)
        write_json(status_path, info)
    write_json(ROOT / 'sessions' / 'active_live.json', {'folder': str(folder)})
    frames = 0
    try:
        with mic.recorder(samplerate=48000, channels=2, blocksize=4800) as source, \
                open(folder / 'whole.wav', 'w+b', buffering=0) as raw, wave.open(raw, 'wb') as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(48000)
            status('recording', seconds=0)
            last_update = time.monotonic()
            while frames < 48000 * 600 and not (folder / 'STOP').exists():
                block = source.record(numframes=4800)
                pcm=(np.clip(block,-1,1)*32767).astype('<i2')
                peak=int(np.max(np.abs(pcm.astype(np.int32))))
                info['pcm_peak']=max(info['pcm_peak'],peak)
                info['nonzero_samples_total']+=int(np.count_nonzero(pcm))
                if peak:info['signal_seconds']+=len(block)/48000
                out.writeframes(pcm.tobytes())
                frames += len(block)
                if frames == len(block):
                    info['audio_zero_monotonic'] = time.monotonic()-len(block)/48000
                    info['clock_note'] = 'Host capture clock estimate; remote audiovisual delay is not calibrated.'
                if time.monotonic() - last_update >= 1:
                    checkpoint(raw)
                    status('recording', seconds=frames / 48000, checkpoint_seconds=frames / 48000,
                           recent_peak=float(np.max(np.abs(block))))
                    last_update = time.monotonic()
            checkpoint(raw)
        status('saved', seconds=frames / 48000)
    except Exception as exc:
        status('failed', seconds=frames / 48000, error=str(exc))
        raise


if __name__ == '__main__':
    main()
