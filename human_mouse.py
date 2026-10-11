"""Cancelable smooth Windows input with manual takeover detection."""
import ctypes as C
import math
import random
import time
from assistant import u, W, client

u.GetCursorPos.argtypes = [C.POINTER(W.POINT)]
u.GetAsyncKeyState.argtypes = [C.c_int]


class Cancelled(RuntimeError):
    pass


def smooth_path(start, end, steps, bend):
    dx, dy = end[0]-start[0], end[1]-start[1]
    length = max(1, math.hypot(dx, dy))
    normal = (-dy/length*bend, dx/length*bend)
    result = []
    for i in range(1, steps+1):
        t = i/steps
        s = t*t*(3-2*t)
        curve = math.sin(math.pi*s)
        result.append((round(start[0]+dx*s+normal[0]*curve), round(start[1]+dy*s+normal[1]*curve)))
    return result


class HumanMouse:
    def __init__(self, hwnd, event, rng=None):
        self.hwnd, self.event = hwnd, event
        self.rng = rng or random.Random()
        self.geometry = client(hwnd)
        self.expected = self.position()
        self.on_pause = None

    @staticmethod
    def position():
        point = W.POINT()
        if not u.GetCursorPos(C.byref(point)): raise RuntimeError('无法读取鼠标位置。')
        return point.x, point.y

    def check(self):
        if self.event.is_set() or u.GetAsyncKeyState(0x1B) & 0x8000:
            raise Cancelled('已停止（停止按钮或 Esc）。')
        if u.GetAncestor(u.GetForegroundWindow(), 2) != u.GetAncestor(self.hwnd, 2):
            raise Cancelled('你切换了窗口，自动流程已暂停。')
        if client(self.hwnd) != self.geometry:
            raise Cancelled('窗口移动或缩放，自动流程已暂停，请重新开始。')
        actual = self.position()
        if math.dist(actual, self.expected) > 7 or (u.GetAsyncKeyState(0x01) | u.GetAsyncKeyState(0x02)) & 0x8000:
            if self.on_pause: self.on_pause('你正在使用鼠标，暂停操作；松开并停稳 1.2 秒后继续。')
            stable_since = time.monotonic()
            while True:
                if self.event.wait(.05) or u.GetAsyncKeyState(0x1B) & 0x8000:
                    raise Cancelled('已停止（停止按钮或 Esc）。')
                current = self.position()
                held = (u.GetAsyncKeyState(0x01) | u.GetAsyncKeyState(0x02)) & 0x8000
                focused = u.GetAncestor(u.GetForegroundWindow(), 2) == u.GetAncestor(self.hwnd, 2)
                if current != actual or held or not focused:
                    stable_since = time.monotonic()
                actual = current
                if client(self.hwnd) != self.geometry:
                    raise Cancelled('窗口尺寸或位置变化，请重新定位。')
                if time.monotonic()-stable_since >= 1.2:
                    self.expected = actual
                    if self.on_pause: self.on_pause('鼠标已停稳，继续核对当前画面。')
                    return True
        return False

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

    def move(self, point, shape, slow=False, quick=False):
        self.check()
        end = self.screen_point(point, shape)
        duration = self.rng.uniform(.3, .65)+min(.35, math.dist(self.expected, end)/2200)
        if slow: duration = self.rng.uniform(.8, 1.3)
        if quick: duration = self.rng.uniform(.10, .16)
        steps = max(12, round(duration*60))
        path = smooth_path(self.expected, end, steps, self.rng.uniform(-18, 18))
        for target in path:
            if self.check():
                return self.move(point, shape, slow=slow, quick=quick)
            if not u.SetCursorPos(*target): raise RuntimeError('鼠标移动失败。')
            self.expected = target
            self.event.wait(duration/steps)
        self.check()

    def click_box(self, box, shape, before_press=None, quick=False):
        x, y, w, h = box
        point = (x+w/2+self.rng.uniform(-.1, .1)*w, y+h/2+self.rng.uniform(-.1, .1)*h)
        self.move(point, shape, quick=quick)
        self.wait(self.rng.uniform(.02, .04) if quick else self.rng.uniform(.12, .32))
        self.check()
        screen = W.POINT(*self.expected)
        if u.GetAncestor(u.WindowFromPoint(screen), 2) != u.GetAncestor(self.hwnd, 2):
            raise Cancelled('点击位置被遮挡，已停止。')
        if before_press: before_press()
        if self.check(): return self.click_box(box, shape, before_press, quick=quick)
        u.mouse_event(0x0002, 0, 0, 0, 0)
        try:
            # Do not treat our own held button as manual takeover.
            if self.event.wait(self.rng.uniform(.06, .13)):
                raise Cancelled('点击过程已取消。')
        finally:
            u.mouse_event(0x0004, 0, 0, 0, 0)

    def scroll(self, shape, down=True, notches=2):
        self.move((shape[1]*self.rng.uniform(.78, .87), shape[0]*self.rng.uniform(.51, .64)), shape)
        for _ in range(notches):
            self.check()
            u.mouse_event(0x0800, 0, 0, C.c_uint32(-120 if down else 120).value, 0)
            self.wait(self.rng.uniform(.13, .3))
        self.wait(self.rng.uniform(.5, 1.0))
