"""Successor corroboration for near-end OCR regressions in integration mode."""
from playback import RoundTracker


class ResilientTracker(RoundTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.peak={}
        self.corrections=[]

    def observe(self,readings,now):
        for name in self.expected:
            r=readings.get(name)
            if r and r.state=='active' and r.current is not None and r.total:
                old=self.peak.get(name)
                if old is None or r.current>old[0]:
                    self.peak[name]=(r.current,r.total,now+r.total-r.current)
        for index,name in enumerate(self.expected):
            r=readings.get(name)
            if not r or r.state!='active': continue
            for prior in self.expected[:index]:
                segment=self.segments[prior]
                peak=self.peak.get(prior)
                idle=readings.get(prior)
                # A successor corroborates the old near-end timeline. Do not
                # treat a regressing timer alone as either completion or seek.
                if (peak and segment.started and not segment.done and idle and idle.state=='idle'
                        and peak[0]>=peak[1]-3 and now>=peak[2]-.75):
                    segment.expected_end=peak[2]
                    self.corrections.append({'target':prior,'peak_progress':peak[0],
                        'successor':name,'note':'successor + idle + near-end timeline corroboration'})
        return super().observe(readings,now)
