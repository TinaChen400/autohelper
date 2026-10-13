"""Cancelable smooth Windows input with manual takeover detection."""
import ctypes as C
import math
import random
import time
import threading
from assistant import u, W, client

u.GetCursorPos.argtypes = [C.POINTER(W.POINT)]
u.GetAsyncKeyState.argtypes = [C.c_int]


class Cancelled(RuntimeError):
    pass


def smooth_path(start, end, steps, bend, easing='smooth'):
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = max(1, math.hypot(dx, dy))
    normal = (-dy/length*bend, dx/length*bend)
    result = []
    for i in range(1, steps+1):
        t = i/steps
        s = (t*t*t*(10-15*t+6*t*t) if easing=='gentle' else
             (1-math.cos(math.pi*t))/2 if easing=='cosine' else t*t*(3-2*t))
        curve = math.sin(math.pi*s)
        result.append((round(start[0]+dx*s+normal[0]*curve), round(start[1]+dy*s+normal[1]*curve)))
    return result


def safe_click_point(box, rng):
    # Central 52% stays inside even a circular play control at every corner.
    # Avoid the tiny old +/-10% cluster while keeping borders untouched.
    x,y,w,h=box
    return (x+w*rng.uniform(.24,.76), y+h*rng.uniform(.24,.76))


class HumanMouse:
    def __init__(self, hwnd, event, rng=None):
        self.hwnd, self.event = hwnd, event
        self.rng = rng or random.Random()
        self.geometry = client(hwnd)
        self.expected = self.position()
        self.on_pause = None
        self.interference_count = 0
        self.gaze_lock=threading.RLock()

    @staticmethod
    def position():
        point = W.POINT()
        if not u.GetCursorPos(C.byref(point)): raise RuntimeError('无法读取鼠标位置。')
        return point.x, point.y

    def check_context(self, require_focus=True):
        if self.event.is_set() or u.GetAsyncKeyState(0x1B) & 0x8000:
            raise Cancelled('已停止（停止按钮或 Esc）。')
        if require_focus and u.GetAncestor(u.GetForegroundWindow(), 2) != u.GetAncestor(self.hwnd, 2):
            raise Cancelled('所选向日葵窗口失去键盘鼠标焦点；音频可能继续播放，但助手已停止自动操作。请返回向日葵后重新开始。')
        if client(self.hwnd) != self.geometry:
            raise Cancelled('窗口移动或缩放，自动流程已暂停，请重新开始。')

    def check_observation(self):
        """Read-only playback polling must never wait for manual pointer use."""
        self.check_context(require_focus=False)
        lock=getattr(self,'gaze_lock',None)
        if lock:
            with lock:actual=self.position();expected=self.expected
        else:actual=self.position();expected=self.expected
        held=bool((u.GetAsyncKeyState(0x01)|u.GetAsyncKeyState(0x02))&0x8000)
        interfered=math.dist(actual,expected)>7 or held
        if interfered and not getattr(self,'observation_interfered',False):
            callback=getattr(self,'on_interference',None)
            if callback:callback(expected=self.expected,actual=actual,held=held,observation_continues=True)
            if self.on_pause:self.on_pause('检测到人工鼠标操作：继续录音和播放观察，自动点击等待停稳后再进行。')
        self.observation_interfered=interfered
        return interfered

    def wait_observation(self,seconds):
        deadline=time.monotonic()+seconds
        while True:
            self.check_observation()
            remaining=deadline-time.monotonic()
            if remaining<=0:return
            self.event.wait(min(.04,remaining))

    def check(self):
        self.check_context(require_focus=False)
        focused=u.GetAncestor(u.GetForegroundWindow(),2)==u.GetAncestor(self.hwnd,2)
        resumed=False
        if not focused:
            if self.on_pause:self.on_pause('远程窗口失去焦点：仅等待鼠标操作，已保存录音和后台分析继续。返回远程后自动核对继续。')
            while not focused:
                if self.event.wait(.1):raise Cancelled('已停止（停止按钮）。')
                self.check_context(require_focus=False)
                focused=u.GetAncestor(u.GetForegroundWindow(),2)==u.GetAncestor(self.hwnd,2)
            self.expected=self.position()
            self.interference_count=getattr(self,'interference_count',0)+1
            resumed=True
        actual = self.position()
        if math.dist(actual, self.expected) > 7 or (u.GetAsyncKeyState(0x01) | u.GetAsyncKeyState(0x02)) & 0x8000:
            self.interference_count = getattr(self,'interference_count',0)+1
            callback=getattr(self,'on_interference',None)
            if callback:callback(expected=self.expected,actual=actual,
                held=bool((u.GetAsyncKeyState(0x01)|u.GetAsyncKeyState(0x02))&0x8000))
            if self.on_pause: self.on_pause('你正在使用鼠标，暂停操作；松开并停稳 1.2 秒后继续。')
            stable_since = time.monotonic()
            while True:
                if self.event.wait(.05) or u.GetAsyncKeyState(0x1B) & 0x8000:
                    raise Cancelled('已停止（停止按钮或 Esc）。')
                current = self.position()
                held = (u.GetAsyncKeyState(0x01) | u.GetAsyncKeyState(0x02)) & 0x8000
                focused = u.GetAncestor(u.GetForegroundWindow(), 2) == u.GetAncestor(self.hwnd, 2)
                if not focused:
                    stable_since=time.monotonic()
                    actual=current
                    continue
                if current != actual or held:
                    stable_since = time.monotonic()
                actual = current
                if client(self.hwnd) != self.geometry:
                    raise Cancelled('窗口尺寸或位置变化，请重新定位。')
                if time.monotonic()-stable_since >= 1.2:
                    self.expected = actual
                    self.observation_interfered=False
                    if self.on_pause: self.on_pause('鼠标已停稳，继续核对当前画面。')
                    return True
        return resumed

    def wait(self, seconds):
        deadline = time.monotonic()+seconds
        while True:
            self.check()
            remaining = deadline-time.monotonic()
            if remaining <= 0: return
            self.event.wait(min(.04, remaining))

    def screen_point(self, point, shape):
        x, y, w, h = self.geometry
        return round(x+point[0]/shape[1]*w), round(y+point[1]/shape[0]*h)

    def observation_glide(self,point,shape,now=None):
        """Schedule a smooth cursor path without waiting or changing focus."""
        now=time.monotonic() if now is None else now
        focused=u.GetAncestor(u.GetForegroundWindow(),2)==u.GetAncestor(self.hwnd,2)
        if not focused or self.check_observation():
            self.observation_path=None;return False
        if point is not None:
            end=self.screen_point(point,shape)
            self.observation_path=(now,self.rng.uniform(.8,1.3),self.expected,end,self.rng.uniform(-10,10))
        path=getattr(self,'observation_path',None)
        if not path:return False
        if hasattr(self,'gaze_lock'):
            worker=getattr(self,'gaze_thread',None)
            if not worker or not worker.is_alive():
                self.gaze_thread=threading.Thread(target=self._gaze_worker,daemon=True)
                self.gaze_thread.start()
            return True
        return self._gaze_step(now)

    def _gaze_worker(self):
        while not self.event.wait(.016):
            with self.gaze_lock:
                if not self.observation_path:return
                if (u.GetAncestor(u.GetForegroundWindow(),2)!=u.GetAncestor(self.hwnd,2) or
                        self.check_observation() or client(self.hwnd)!=self.geometry):
                    self.observation_path=None;return
                self._gaze_step(time.monotonic())

    def _gaze_step(self,now):
        path=getattr(self,'observation_path',None)
        if not path:return False
        began,duration,start,end,bend=path
        fraction=min(1,max(0,(now-began)/duration))
        # Reuse the same curve formula, but each playback poll advances it.
        steps=100;index=max(0,min(99,round(fraction*99)))
        target=smooth_path(start,end,steps,bend)[index]
        if not u.SetCursorPos(*target):return False
        self.expected=target
        if fraction>=1:self.observation_path=None
        return True

    def move(self, point, shape, slow=False, quick=False):
        lock=getattr(self,'gaze_lock',None)
        if lock:
            with lock:self.observation_path=None
        else:self.observation_path=None
        self.check()
        end = self.screen_point(point, shape)
        duration = self.rng.uniform(.32, .65)+min(.35, math.dist(self.expected, end)/2400)
        if slow: duration = self.rng.uniform(.8, 1.3)
        if quick: duration = self.rng.uniform(.20, .38)+min(.18,math.dist(self.expected,end)/3800)
        steps = max(12, round(duration*60))
        bend_limit=min(18,math.dist(self.expected,end)*.16)
        path = smooth_path(self.expected, end, steps, self.rng.uniform(-bend_limit,bend_limit),
                           easing=self.rng.choice(('smooth','gentle','cosine')))
        for target in path:
            if self.check():
                return self.move(point, shape, slow=slow, quick=quick)
            if not u.SetCursorPos(*target): raise RuntimeError('鼠标移动失败。')
            self.expected = target
            self.event.wait(duration/steps)
        self.check()

    def click_box(self, box, shape, before_press=None, quick=False):
        x, y, w, h = box
        point = safe_click_point(box,self.rng)
        self.move(point, shape, quick=quick)
        self.wait(self.rng.uniform(.02, .04) if quick else self.rng.uniform(.12, .32))
        self.check()
        screen = W.POINT(*self.expected)
        if u.GetAncestor(u.WindowFromPoint(screen), 2) != u.GetAncestor(self.hwnd, 2):
            raise Cancelled('点击位置被遮挡，已停止。')
        if before_press: before_press()
        if self.check(): return self.click_box(box, shape, before_press, quick=quick)
        self.last_click={'point':list(point),'box':list(box)}
        u.mouse_event(0x0002, 0, 0, 0, 0)
        try:
            # Do not treat our own held button as manual takeover.
            if self.event.wait(self.rng.uniform(.06, .13)):
                raise Cancelled('点击过程已取消。')
        finally:
            u.mouse_event(0x0004, 0, 0, 0, 0)
            # Allow the queued mouse-up to reach GetAsyncKeyState before the
            # next guard; our own held bit must not trigger a 1.2 s takeover.
            self.event.wait(.02)

    def scroll(self, shape, down=True, notches=2):
        self.move((shape[1]*self.rng.uniform(.73, .91), shape[0]*self.rng.uniform(.38, .73)), shape)
        burst=self.rng.randint(2,4)
        for index in range(notches):
            self.check()
            # AweSun can coalesce grouped wheel deltas into one remote notch.
            # Send short, separated notches rather than rescanning a half form.
            u.mouse_event(0x0800, 0, 0, C.c_uint32(-120 if down else 120).value, 0)
            self.wait(self.rng.uniform(.08, .19))
            if (index+1)%burst==0 and index+1<notches:self.wait(self.rng.uniform(.22,.55))
        self.wait(self.rng.uniform(.3, .65))
