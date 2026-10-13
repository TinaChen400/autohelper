"""Lightweight manual Submit monitor survives rating-observer replacement.

Only recent, screenshot-verified Submit geometry is used. It observes local
input and never activates a window or sends mouse/keyboard input.
"""
import threading
import time
from remote_ratings import same_layout
from rating_reader import refresh_selections,submit_enabled


class ManualSubmitMonitor:
    def __init__(self,flow):
        self.flow=flow;self.lock=threading.Lock();self.stopped=threading.Event()
        self.screen=None;self.image=None;self.geometry=None;self.observed_at=0
        self.invalidated=False
        self.thread=threading.Thread(target=self.run,daemon=True)

    def publish(self,screen,image,geometry,now=None):
        if not screen.get('submit_boxes'):return
        with self.lock:
            self.screen={**screen};self.image=image.copy();self.geometry=geometry
            self.invalidated=False
            self.observed_at=time.monotonic() if now is None else now

    def observe_frame(self,image,geometry,now=None):
        with self.lock:
            if self.screen is None:return
            if self.invalidated:return
            if geometry!=self.geometry or not same_layout(self.image,image,self.screen['rows'],self.screen.get('guard_boxes',())):
                self.observed_at=0;return
            rows=refresh_selections(image,self.screen['rows'])
            chosen={row['key'] for row in self.screen['rows'] if row.get('selection_state')=='selected'}
            visible=[row for row in rows if row['key'] in chosen]
            if len(visible)>=2 and all(row.get('selection_state')=='none' for row in visible):
                self.invalidated=True;self.observed_at=0;return
            self.screen={**self.screen,'rows':rows,
                         'submit_enabled':submit_enabled(image,self.screen['submit_boxes'])}
            self.image=image.copy();self.observed_at=time.monotonic() if now is None else now

    def click(self,point,geometry,now=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if (self.invalidated or self.screen is None or geometry!=self.geometry or not self.observed_at
                    or not 0<=now-self.observed_at<=2 or not self.screen.get('submit_enabled')):return False
            px,py=point
            if not any(x<=px<=x+w and y<=py<=y+h for x,y,w,h in self.screen['submit_boxes']):return False
            screen=self.screen;image=self.image.copy()
        return self.flow.note_submission('user',screen,image,'continuous_manual_click_monitor')

    def run(self):
        from assistant import u,client
        from dynamic_app import foreground
        from human_mouse import HumanMouse
        held=bool(u.GetAsyncKeyState(0x01)&0x8000)
        refreshed=0
        while not self.stopped.wait(.02) and not self.flow.event.is_set():
            if time.monotonic()-refreshed>=.2:
                with self.lock:observe=(self.screen is not None and not self.invalidated
                    and time.monotonic()-self.observed_at>.3)
                if observe:
                    try:
                        image,geometry=self.flow.grab(self.flow.hwnd)
                        self.observe_frame(image,geometry)
                    except Exception:
                        with self.lock:self.observed_at=0
                refreshed=time.monotonic()
            down=bool(u.GetAsyncKeyState(0x01)&0x8000)
            if down and not held and foreground(self.flow.hwnd):
                geometry=client(self.flow.hwnd)
                with self.lock:
                    shape=None if self.image is None else self.image.shape
                if shape is not None:
                    x,y,w,h=geometry;px,py=HumanMouse.position()
                    try:self.click(((px-x)/w*shape[1],(py-y)/h*shape[0]),geometry)
                    except OSError as exc:
                        self.flow.log('submit_monitor_error',reason=str(exc))
            held=down
