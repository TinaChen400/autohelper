"""Conservative first-round segmentation using visual progress and audio gaps."""
import json
from pathlib import Path
import wave
import numpy as np


class CaptureUnavailable(ValueError):
    """No captured signal; repeating clicks cannot repair the audio route."""


def progress_interval(run, name, end_confirmed=False):
    def interval(row):
        end=row['audio_seconds']-row['readings'][name]['current']
        return end-1,end
    low=max(interval(row)[0] for row in run)
    high=min(interval(row)[1] for row in run)
    if low<=high+.25: return low,high,[]
    # One digit can be read one second ahead (e.g. 15 as 16) in two adjacent
    # frames. Recover only a bounded internal cluster with >=95% consensus;
    # the original reading remains in the journal and audio gaps still gate
    # every boundary below. Short/sustained/conflicting traces stay rejected.
    if len(run)>=20:
        for count in (1,2):
            if (len(run)-count)/len(run)<.95:continue
            for index in range(1,len(run)-count):
                removed=run[index:index+count]
                current=removed[0]['readings'][name]['current']
                total=removed[0]['readings'][name]['total']
                if not 1<current<total-1:continue
                if any(row['readings'][name]['current']!=current for row in removed):continue
                if not run[index-1]['readings'][name]['current']<=current<=run[index+count]['readings'][name]['current']:continue
                kept=run[:index]+run[index+count:]
                values={r['readings'][name]['current'] for r in kept}
                if len(values)<6 or min(values)>1 or max(values)<total-1:continue
                a=max(interval(r)[0] for r in kept);b=min(interval(r)[1] for r in kept)
                distances=[max(a-interval(row)[1],interval(row)[0]-b,0) for row in removed]
                if a<=b+.25 and all(.25<distance<=1.1 for distance in distances):
                    return a,b,[{'audio_seconds':row['audio_seconds'],'current':current,
                        'reason':'至多两帧连续内部计时偏差；至少95%样本、六种进度及首尾一致，仍需声音边界验证'} for row in removed]
    # A compressed 25 can be read as the terminal 26 for two frames. Unlike
    # an internal digit, this needs an independently observed idle end and
    # at least two later terminal samples consistent with the complete clock.
    # Never infer completion from an early saturated timer alone.
    if end_confirmed and len(run)>=20:
        total=run[-1]['readings'][name]['total']
        for count in (1,2):
            if (len(run)-count)/len(run)<.95:continue
            for index in range(1,len(run)-count-1):
                removed=run[index:index+count]
                if any(row['readings'][name]['current']!=total for row in removed):continue
                prefix=run[:index];tail=run[index+count:]
                values=[row['readings'][name]['current'] for row in prefix]
                if len(set(values))<6 or values[0]>1 or values[-1]<total-2:continue
                if any(row['readings'][name]['current']!=total for row in tail):continue
                if not 0<tail[0]['audio_seconds']-removed[0]['audio_seconds']<=1.25:continue
                kept=prefix+tail
                a=max(interval(row)[0] for row in kept);b=min(interval(row)[1] for row in kept)
                distances=[max(a-interval(row)[1],interval(row)[0]-b,0) for row in removed]
                if a<=b+.25 and all(.25<distance<=1.1 for distance in distances):
                    return a,b,[{'audio_seconds':row['audio_seconds'],'current':total,
                        'reason':'至多两帧提前读成结束值；至少95%完整计时、后续两帧结束值及独立停止态确认，仍需声音边界验证'} for row in removed]
    # Historical recordings can contain one reused startup read attached to a
    # newer frame. Recover only a single stale zero, corroborated by >=4 other
    # samples (>=80%), near-end progress, and subsequent audio-boundary checks.
    for index,row in enumerate(run):
        if index==0 or row['readings'][name]['current']!=0: continue
        kept=run[:index]+run[index+1:]
        if len(kept)<4 or len(kept)/len(run)<.8: continue
        if not any(r['readings'][name]['current']==0 for r in run[:index]): continue
        total=row['readings'][name]['total']
        later=run[index+1:]
        if not later or max(r['readings'][name]['current'] for r in later)<total-1: continue
        low=max(interval(r)[0] for r in kept)
        high=min(interval(r)[1] for r in kept)
        if low<=high+.25:
            return low,high,[{'audio_seconds':row['audio_seconds'],'current':0,
                             'reason':'单个滞后的起始读数；其他至少四个进度样本一致，仍需声音边界验证'}]
    # A briefly frozen/nonzero OCR timer can repeat an old integer for one or
    # two additional frames. Require a small contiguous tail of that same
    # integer, >=85% agreement, four distinct progress values and both ends.
    # Multiple stale zero samples and sustained timing changes stay rejected.
    if len(run)>=10:
        for count in (1,2):
            if (len(run)-count)/len(run)<.85: continue
            for index in range(1,len(run)-count):
                removed=run[index:index+count]
                current=removed[0]['readings'][name]['current']
                total=removed[0]['readings'][name]['total']
                if not 0<current<total-1: continue
                if any(row['readings'][name]['current']!=current for row in removed): continue
                if run[index-1]['readings'][name]['current']!=current: continue
                if run[index+count]['readings'][name]['current']<=current: continue
                kept=run[:index]+run[index+count:]
                values={row['readings'][name]['current'] for row in kept}
                if len(values)<4 or min(values)>1 or max(values)<total-1: continue
                low=max(interval(row)[0] for row in kept)
                high=min(interval(row)[1] for row in kept)
                if low<=high+.25:
                    return low,high,[{'audio_seconds':row['audio_seconds'],'current':current,
                        'reason':'短段重复的非零计时滞后；至少85%样本、四种进度及首尾一致，仍需声音边界验证'} for row in removed]
    raise ValueError(f'{name}: 计时与画面不一致。')


def first_progress_run(usable,name):
    """Recover a brief false zero only when the whole clock corroborates it."""
    run=[];last=-1
    for index,row in enumerate(usable):
        current=row['readings'][name]['current']
        if current<last:
            # A genuine restart changes the subsequent start-time interval.
            # Do not treat it as an OCR fault merely because it reads zero.
            for count in (1,2):
                if index+count>=len(usable) or len(usable)<20:continue
                removed=usable[index:index+count]
                if any(r['readings'][name]['current']!=0 for r in removed):continue
                if last<=2 or usable[index+count]['readings'][name]['current']<last:continue
                if not 0<usable[index+count]['audio_seconds']-run[-1]['audio_seconds']<=1.75:continue
                kept=usable[:index]+usable[index+count:]
                if len(kept)/len(usable)<.90:continue
                values=[r['readings'][name]['current'] for r in kept]
                total=row['readings'][name]['total']
                if len(set(values))<6 or values[0]>1 or values[-1]<total-1:continue
                if any(a>b for a,b in zip(values,values[1:])):continue
                low=max(r['audio_seconds']-r['readings'][name]['current']-1 for r in kept)
                high=min(r['audio_seconds']-r['readings'][name]['current'] for r in kept)
                if low>high+.25:continue
                return kept,[{'audio_seconds':r['audio_seconds'],'current':0,
                    'reason':'中途至多两帧归零；1.75秒内恢复原进度，至少90%完整单调计时一致，仍需声音边界验证'} for r in removed]
            return run,[]
        run.append(row);last=current
    return run,[]


def segment(folder):
    folder=Path(folder)
    rows=[json.loads(line) for line in (folder/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
    order=tuple(json.loads((folder/'session.json').read_text(encoding='utf-8'))['player_order']) if (folder/'session.json').exists() else ('Input','A','B')
    if order not in (('Input','A','B'),('A','B')): raise ValueError('不支持的录音题型。')
    with wave.open(str(folder/'whole.wav')) as source:
        rate=source.getframerate(); channels=source.getnchannels()
        samples=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2').reshape(-1,channels)
    # A fully silent capture is not a sequence of inter-segment gaps. Treating
    # it that way chooses the same huge gap for all players and hides the
    # actual capture failure behind a misleading ordering error.
    if not samples.size or not np.any(samples):
        device='当前输出设备'
        try:device=json.loads((folder/'capture_status.json').read_text(encoding='utf-8')).get('device',device)
        except (OSError,ValueError):pass
        raise CaptureUnavailable(f'未采集到声音：{device} 的整段录音为全零；请确认向日葵远程声音已传到本机。未启动模型评分。')
    mono=samples.astype(np.float32).mean(axis=1)/32768
    step=max(1,rate//100)
    energy=np.sqrt(np.mean(mono[:len(mono)//step*step].reshape(-1,step)**2,axis=1))
    quiet=energy<.0015
    starts=np.flatnonzero(np.diff(np.r_[False,quiet,False].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,quiet,False].astype(int))==-1)
    short_gaps=[(a*step/rate,b*step/rate) for a,b in zip(starts,ends) if b-a>=4]
    gaps=[gap for gap in short_gaps if gap[1]-gap[0]>=.08-1e-8]
    estimates={}
    previous=-1
    for name in order:
        active=[r for r in rows if r.get('label')==name and r['audio_seconds']>previous]
        usable=[r for r in active if r.get('readings',{}).get(name,{}).get('current') is not None]
        if len(usable)<2: raise ValueError(f'{name}: 有效播放进度不足，不能自动切分。')
        first=usable[0]; reading=first['readings'][name]
        total=reading['total']
        if not total or any(r['readings'][name]['total']!=total for r in usable):
            raise ValueError(f'{name}: 总时长不一致。')
        startup_excluded=[]
        # On autoplay, the pause icon can appear one frame before the timer
        # clears the previous completed play. Require two actual zero frames
        # immediately afterwards; never infer zero from a high first reading.
        if (len(usable)>=3 and reading['current']==total
                and all(r['readings'][name]['current']==0 for r in usable[1:3])
                and 0<usable[1]['audio_seconds']-first['audio_seconds']<=.75
                and 0<usable[2]['audio_seconds']-first['audio_seconds']<=1.25):
            startup_excluded=[{'audio_seconds':first['audio_seconds'],'current':reading['current'],
                'reason':'自动播放切换时保留上一轮结束计时；随后两帧明确归零，仍需完整进度和声音边界验证'}]
            usable=usable[1:]
            first=usable[0];reading=first['readings'][name]
        if reading['current']>2:
            raise ValueError(f"{name}: 首个可用计时已到 {reading['current']} 秒，缺少起始计时证据（可能是早期识别失败），不能确认完整录音。")
        # Use only the first continuous play, rejecting progress regressions.
        run,regression_excluded=first_progress_run(usable,name)
        last=run[-1]['readings'][name]['current']
        ended=any(r['audio_seconds']>=run[-1]['audio_seconds'] and
                  r['audio_seconds']-run[-1]['audio_seconds']<=1.25 and
                  r.get('readings',{}).get(name,{}).get('state')=='idle' and
                  r['readings'][name].get('current')==total and
                  r['readings'][name].get('total')==total for r in rows)
        low,high,excluded=progress_interval(run,name,end_confirmed=ended)
        center=(low+high)/2
        candidates=[g for g in gaps if g[1]>=low-.5 and g[0]<=high+.5]
        boundary_minimum_ms=80
        # Autoplay can leave less than 80 ms between complete clips. Accept
        # an actual >=40 ms quiet gap only inside a narrow, complete visual
        # start interval with bounded clock tolerance; never cut continuous speech.
        values={r['readings'][name]['current'] for r in run}
        # A three-second clip normally supplies active 0/1/2, then idle 3.
        # Requiring four *active* integers made the short-gap fallback
        # unreachable even with a fully observed end and a real 50 ms gap.
        enough_progress=(len(values)>=4 or
                         total<=3 and ended and set(range(total))<=values and len(run)>=4)
        if (not candidates and name!=order[0] and -.25<=high-low<=.5
                and enough_progress and min(values)<=1 and last>=total-1):
            # progress_interval already permits <=250 ms clock disagreement.
            # Do not reject that accepted interval just because its ends
            # cross. Allow 100 ms of AV/energy-frame quantization around it;
            # still require an actual >=40 ms quiet gap and complete progress.
            candidates=[g for g in short_gaps if g[1]>=min(low,high)-.1 and g[0]<=max(low,high)+.1]
            if candidates:boundary_minimum_ms=40
        if not candidates: raise ValueError(f'{name}: 边界附近没有确认的声音间隙。')
        gap=min(candidates,key=lambda g:abs(min(max(center,g[0]),g[1])-center))
        begin=max(0,gap[1]-.05) if name==order[0] else sum(gap)/2
        estimates[name]={'start':begin,'total':total,'visual_start_interval':[low,high],
                         'gap':gap,'boundary_minimum_ms':boundary_minimum_ms,'last_progress':last,
                         'excluded_stale_samples':startup_excluded+regression_excluded+excluded}
        previous=run[0]['audio_seconds']
    positions=[estimates[n]['start'] for n in order]
    if positions[0]<0 or positions[-1]>=len(samples)/rate or any(a>=b for a,b in zip(positions,positions[1:])):
        raise ValueError('音频段顺序不成立。')
    c=positions[-1]
    # B must reach its near-end progress and leave some quiet audio at the tail.
    end_gaps=[g for g in gaps if g[0]>c+estimates['B']['total']-.8]
    if estimates['B']['last_progress']<estimates['B']['total']-1 or not end_gaps:
        raise ValueError('B 末尾证据不足，请补听完整B。')
    end=min(end_gaps,key=lambda g:g[0])[0]+.1
    end=min(end,len(samples)/rate)
    bounds={name:[positions[index],positions[index+1] if index+1<len(order) else end] for index,name in enumerate(order)}
    for name,(start,stop) in bounds.items():
        duration=stop-start
        if abs(duration-estimates[name]['total'])>1.75:
            raise ValueError(f'{name}: 切片长度与画面时长差异过大。')
    # Write only after every segment passes the checks.
    for name,(start,stop) in bounds.items():
        with wave.open(str(folder/f'{name}.wav'),'wb') as out:
            out.setnchannels(channels); out.setsampwidth(2); out.setframerate(rate)
            out.writeframes(samples[round(start*rate):round(stop*rate)].tobytes())
    manifest={'player_order':list(order),'task_type':'ab_only' if len(order)==2 else 'reference_ab',
              'bounds_seconds':bounds,'estimates':estimates,'complete_confirmed':False,
              'boundary_method':'visual integer progress intervals + nearby audio quiet gaps',
              'target_text':None,'note':'自动候选切片；传输音画延迟和切分完整性仍需人工回听验证。'}
    (folder/'clip_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
