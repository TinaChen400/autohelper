"""Decode original timestamps and export audio/video evidence without UI use."""
import argparse
import json
from pathlib import Path
import wave
import av
import numpy as np


def prepare(video, destination):
    destination.mkdir(parents=True,exist_ok=True)
    audio = av.open(str(video))
    resampler = av.AudioResampler(format='s16',layout='mono',rate=48000)
    chunks = []
    origin = None
    for frame in audio.decode(audio=0):
        if origin is None: origin = float(frame.time or 0)
        for block in resampler.resample(frame): chunks.append(block.to_ndarray().reshape(-1))
    for block in resampler.resample(None): chunks.append(block.to_ndarray().reshape(-1))
    samples = np.concatenate(chunks)
    with wave.open(str(destination/'source.wav'),'wb') as output:
        output.setnchannels(1); output.setsampwidth(2); output.setframerate(48000)
        output.writeframes(samples.astype('<i2').tobytes())
    container = av.open(str(video))
    observations = []
    next_at = 0
    for packet in container.demux(video=0):
        try: frames = packet.decode()
        except av.error.InvalidDataError: continue
        for frame in frames:
            at = float(frame.time or 0)
            if at < next_at: continue
            filename = f'frame_{len(observations):03d}.npz'
            np.savez_compressed(destination/filename,image=frame.to_ndarray(format='bgr24'))
            observations.append({'time':at,'file':filename})
            next_at = at+.24
    (destination/'video_manifest.json').write_text(json.dumps({'video':str(video.resolve()),'audio_origin':origin,
        'audio_seconds':len(samples)/48000,'frames':observations},indent=2),encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video',type=Path)
    parser.add_argument('destination',type=Path)
    args = parser.parse_args()
    prepare(args.video,args.destination)
