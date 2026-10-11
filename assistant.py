"""Windows remote-player calibration and visual playback verification."""
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'calibration.json'
u = C.WinDLL('user32', use_last_error=True)
g = C.WinDLL('gdi32', use_last_error=True)
try:
    u.SetProcessDpiAwarenessContext(C.c_void_p(-4))
except (AttributeError, OSError):
    u.SetProcessDPIAware()

# Explicit signatures are required for 64-bit Windows handles.
u.GetDC.argtypes = [W.HWND]; u.GetDC.restype = W.HDC
u.ReleaseDC.argtypes = [W.HWND, W.HDC]
u.GetClientRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
u.ClientToScreen.argtypes = [W.HWND, C.POINTER(W.POINT)]
u.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
u.GetWindowTextLengthW.argtypes = [W.HWND]
u.IsWindowVisible.argtypes = [W.HWND]
u.IsWindow.argtypes = [W.HWND]
u.IsIconic.argtypes = [W.HWND]
u.GetForegroundWindow.restype = W.HWND
u.GetAncestor.argtypes = [W.HWND, W.UINT]; u.GetAncestor.restype = W.HWND
u.WindowFromPoint.argtypes = [W.POINT]; u.WindowFromPoint.restype = W.HWND
u.SetForegroundWindow.argtypes = [W.HWND]
u.ShowWindow.argtypes = [W.HWND, C.c_int]
u.SetCursorPos.argtypes = [C.c_int, C.c_int]
g.CreateCompatibleDC.argtypes = [W.HDC]; g.CreateCompatibleDC.restype = W.HDC
g.CreateCompatibleBitmap.argtypes = [W.HDC, C.c_int, C.c_int]; g.CreateCompatibleBitmap.restype = W.HBITMAP
g.SelectObject.argtypes = [W.HDC, W.HANDLE]; g.SelectObject.restype = W.HANDLE
g.BitBlt.argtypes = [W.HDC, C.c_int, C.c_int, C.c_int, C.c_int, W.HDC, C.c_int, C.c_int, W.DWORD]
g.DeleteObject.argtypes = [W.HANDLE]
g.DeleteDC.argtypes = [W.HDC]

class BitmapInfo(C.Structure):
    _fields_ = [('size', W.DWORD), ('width', W.LONG), ('height', W.LONG),
                ('planes', W.WORD), ('bits', W.WORD), ('compression', W.DWORD),
                ('image_size', W.DWORD), ('xppm', W.LONG), ('yppm', W.LONG),
                ('used', W.DWORD), ('important', W.DWORD)]

g.GetDIBits.argtypes = [W.HDC, W.HBITMAP, W.UINT, W.UINT, C.c_void_p, C.POINTER(BitmapInfo), W.UINT]

def windows():
    found = []
    callback_type = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    def visit(hwnd, _):
        if u.IsWindowVisible(hwnd):
            n = u.GetWindowTextLengthW(hwnd)
            text = C.create_unicode_buffer(n + 1)
            u.GetWindowTextW(hwnd, text, n + 1)
            if text.value:
                found.append((hwnd, text.value))
        return True
    callback = callback_type(visit)
    u.EnumWindows(callback, 0)
    return sorted(found, key=lambda pair: not any(s in pair[1].lower() for s in ('向日葵', 'awesun', 'sunlogin')))

def client(hwnd):
    if not u.IsWindow(hwnd) or u.IsIconic(hwnd):
        raise RuntimeError('窗口已关闭或最小化，请恢复窗口。')
    rect, point = W.RECT(), W.POINT(0, 0)
    if not u.GetClientRect(hwnd, C.byref(rect)) or not u.ClientToScreen(hwnd, C.byref(point)):
        raise RuntimeError('无法读取窗口位置。')
    if rect.right < 10 or rect.bottom < 10:
        raise RuntimeError('窗口尺寸过小。')
    return point.x, point.y, rect.right, rect.bottom

def capture(hwnd, max_width=1000, max_height=600):
    x, y, width, height = client(hwnd)
    scale = min(1, max_width / width, max_height / height)
    sw, sh = max(1, round(width * scale)), max(1, round(height * scale))
    screen = u.GetDC(None)
    dc = g.CreateCompatibleDC(screen)
    bitmap = g.CreateCompatibleBitmap(screen, sw, sh)
    old = g.SelectObject(dc, bitmap)
    try:
        # StretchBlt samples the visible screen; do not cover the remote window.
        g.SetStretchBltMode.argtypes = [W.HDC, C.c_int]
        g.SetStretchBltMode(dc, 3)
        g.StretchBlt.argtypes = [W.HDC, C.c_int, C.c_int, C.c_int, C.c_int, W.HDC, C.c_int, C.c_int, C.c_int, C.c_int, W.DWORD]
        if not g.StretchBlt(dc, 0, 0, sw, sh, screen, x, y, width, height, 0x00CC0020):
            raise RuntimeError('屏幕截图失败。')
        g.SelectObject(dc, old)
        info = BitmapInfo(C.sizeof(BitmapInfo), sw, -sh, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = C.create_string_buffer(sw * sh * 4)
        if not g.GetDIBits(dc, bitmap, 0, sh, buf, C.byref(info), 0):
            raise RuntimeError('无法读取截图像素。')
        raw = buf.raw
        rgb = bytearray(sw * sh * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = raw[2::4], raw[1::4], raw[0::4]
        return sw, sh, bytes(rgb)
    finally:
        g.SelectObject(dc, old)
        g.DeleteObject(bitmap); g.DeleteDC(dc); u.ReleaseDC(None, screen)

def absolute_box(config, name):
    vx, vy, vw, vh = config['viewport']
    bx, by, bw, bh = config['buttons'][name]['box']
    return vx + bx * vw, vy + by * vh, bw * vw, bh * vh

def signature(frame, box):
    width, height, rgb = frame
    x, y, w, h = box
    result = []
    for row in range(24):
        py = min(height - 1, max(0, int((y + (row + .5) * h / 24) * height)))
        for col in range(24):
            px = min(width - 1, max(0, int((x + (col + .5) * w / 24) * width)))
            offset = (py * width + px) * 3
            result.extend(rgb[offset:offset + 3])
    return result

def distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)

class App:
    def __init__(self):
        self.root = tk.Tk(); self.root.title('向日葵播放助手 · 标定与验证')
        self.hwnd = None; self.frame = None; self.mode = None; self.busy = False
        self.config = {'viewport': [0, 0, 1, 1], 'buttons': {}}
        if CONFIG.exists():
            try:
                saved = json.loads(CONFIG.read_text(encoding='utf-8'))
                if len(saved['viewport']) == 4 and isinstance(saved['buttons'], dict):
                    self.config = saved
            except (ValueError, KeyError, TypeError):
                pass
        bar = ttk.Frame(self.root); bar.pack(fill='x', padx=8, pady=8)
        self.window = ttk.Combobox(bar, width=58, state='readonly'); self.window.pack(side='left')
        ttk.Button(bar, text='刷新窗口', command=self.refresh).pack(side='left')
        ttk.Button(bar, text='锁定并截图（2秒）', command=self.lock).pack(side='left')
        self.status = tk.StringVar(value='先选择向日葵远程窗口。截图时请保持窗口可见，助手不能遮住它。')
        ttk.Label(self.root, textvariable=self.status, wraplength=980).pack(fill='x', padx=8)
        tools = ttk.Frame(self.root); tools.pack(fill='x', padx=8, pady=6)
        ttk.Button(tools, text='框选远程画面', command=lambda: self.select('viewport')).pack(side='left')
        for name in ('Input', 'A', 'B'):
            ttk.Button(tools, text=f'框选 {name} 按钮', command=lambda n=name: self.select(n)).pack(side='left')
        ttk.Button(tools, text='重新截图（2秒）', command=lambda: self.delayed(self.snapshot)).pack(side='left')
        self.canvas = tk.Canvas(self.root, width=900, height=450, background='#222'); self.canvas.pack(padx=8, pady=4)
        self.canvas.bind('<ButtonPress-1>', self.start_box)
        self.canvas.bind('<B1-Motion>', self.drag_box)
        self.canvas.bind('<ButtonRelease-1>', self.end_box)
        for name in ('Input', 'A', 'B'):
            row = ttk.Frame(self.root); row.pack(fill='x', padx=8, pady=3)
            ttk.Label(row, text=name, width=8).pack(side='left')
            ttk.Button(row, text='采集停止样本（2秒）', command=lambda n=name: self.delayed(lambda: self.sample(n, 'idle'))).pack(side='left')
            ttk.Button(row, text='采集播放样本（2秒）', command=lambda n=name: self.delayed(lambda: self.sample(n, 'active'))).pack(side='left')
            ttk.Button(row, text='点击播放并验证', command=lambda n=name: self.play(n)).pack(side='left')
        ttk.Button(self.root, text='停止验证（不会点击远程暂停）', command=self.stop).pack(pady=5)
        ttk.Label(self.root, text='仅操作播放按钮；不会填写评价或提交。缩放后先检查标框；远程布局或黑边变化需重新框选。').pack(padx=8, pady=5)
        self.refresh()

    def report(self, message):
        self.status.set(message)

    def refresh(self):
        self.items = windows()
        self.window['values'] = [f'{title}  [窗口 {handle}]' for handle, title in self.items]
        if self.items: self.window.current(0)

    def lock(self):
        if self.busy: return
        index = self.window.current()
        if index < 0: return
        self.hwnd = self.items[index][0]
        self.delayed(self.snapshot)

    def delayed(self, action):
        if self.busy: return
        self.busy = True
        self.report('2 秒后采集，请把向日葵窗口放在可见位置，移开鼠标。')
        def run():
            self.busy = False
            try: action()
            except Exception as exc: self.report(str(exc))
        self.root.after(2000, run)

    def snapshot(self):
        if not self.hwnd: raise RuntimeError('请先锁定窗口。')
        self.frame = capture(self.hwnd)
        width, height, rgb = self.frame
        self.photo = tk.PhotoImage(data=f'P6\n{width} {height}\n255\n'.encode() + rgb, format='PPM')
        self.canvas.configure(width=width, height=height)
        self.canvas.delete('all'); self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        self.draw_boxes()
        self.report('截图已更新。绿色为远程画面，橙色为按钮。请检查位置是否正确。')

    def draw_boxes(self):
        if not self.frame: return
        width, height, _ = self.frame
        boxes = [('viewport', self.config['viewport'], '#22ee99')]
        boxes += [(name, absolute_box(self.config, name), '#ffbb33') for name in self.config['buttons']]
        for name, (x, y, w, h), color in boxes:
            self.canvas.create_rectangle(x * width, y * height, (x + w) * width, (y + h) * height, outline=color, width=2)
            self.canvas.create_text(x * width + 3, y * height + 3, text=name, fill=color, anchor='nw')

    def select(self, mode):
        if self.busy: return
        if not self.frame: return self.report('请先锁定窗口并截图。')
        self.mode = mode
        self.report(f'在截图上拖动框选 {mode}。按钮应紧贴播放图标，避免包含不断变化的计时文字。')

    def start_box(self, event):
        if self.mode and not self.busy:
            self.origin = (event.x, event.y)
            self.canvas.delete('selection')

    def drag_box(self, event):
        if self.mode and hasattr(self, 'origin'):
            self.canvas.delete('selection')
            self.canvas.create_rectangle(*self.origin, event.x, event.y, outline='cyan', tags='selection')

    def end_box(self, event):
        if not self.mode or not hasattr(self, 'origin') or self.busy: return
        width, height, _ = self.frame
        x0, y0 = self.origin; del self.origin
        x1, x2 = sorted((max(0, min(width, x0)), max(0, min(width, event.x))))
        y1, y2 = sorted((max(0, min(height, y0)), max(0, min(height, event.y))))
        if x2 - x1 < 5 or y2 - y1 < 5: return self.report('框选过小，请重新拖动。')
        box = [x1 / width, y1 / height, (x2 - x1) / width, (y2 - y1) / height]
        if self.mode == 'viewport':
            self.config = {'viewport': box, 'buttons': {}}
        else:
            vx, vy, vw, vh = self.config['viewport']
            if box[0] < vx or box[1] < vy or box[0] + box[2] > vx + vw or box[1] + box[3] > vy + vh:
                return self.report('按钮必须位于远程画面内。')
            self.config['buttons'][self.mode] = {'box': [(box[0]-vx)/vw, (box[1]-vy)/vh, box[2]/vw, box[3]/vh]}
        self.mode = None; self.save()
        self.canvas.delete('all'); self.canvas.create_image(0, 0, anchor='nw', image=self.photo); self.draw_boxes()
        self.report('位置已保存。请分别采集按钮的停止和播放中样本。')

    def save(self):
        CONFIG.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding='utf-8')

    def sample(self, name, state):
        if name not in self.config['buttons']: raise RuntimeError(f'请先框选 {name} 按钮。')
        self.snapshot()
        self.config['buttons'][name][state] = signature(self.frame, absolute_box(self.config, name))
        self.save()
        self.report(f'{name} 的 {"播放中" if state == "active" else "停止"} 样本已保存。')

    def classify(self, name, frame):
        button = self.config['buttons'][name]
        current = signature(frame, absolute_box(self.config, name))
        separation = distance(button['idle'], button['active'])
        if separation < 5: return 'invalid', 0, 0
        idle, active = distance(current, button['idle']), distance(current, button['active'])
        limit, margin = max(8, separation * .65), max(2, separation * .2)
        if active < limit and active + margin < idle: return 'active', idle, active
        if idle < limit and idle + margin < active: return 'idle', idle, active
        return 'unknown', idle, active

    def play(self, name):
        if self.busy: return
        button = self.config['buttons'].get(name, {})
        if not all(key in button for key in ('box', 'idle', 'active')):
            return self.report(f'先完成 {name} 的按钮框选及两种状态样本采集。')
        if not self.hwnd: return self.report('请先锁定窗口。')
        self.busy = True
        u.ShowWindow(self.hwnd, 9); u.SetForegroundWindow(self.hwnd)
        self.report('准备检查窗口和停止状态，随后点击一次。')
        self.root.after(900, lambda: self.click_checked(name))

    def click_checked(self, name):
        try:
            if not self.busy: return
            if u.GetAncestor(u.GetForegroundWindow(), 2) != u.GetAncestor(self.hwnd, 2):
                raise RuntimeError('向日葵未成为前台窗口，已取消点击。')
            state, _, _ = self.classify(name, capture(self.hwnd))
            if state != 'idle':
                raise RuntimeError(f'当前按钮未匹配停止样本（{state}），已取消点击，请检查标定。')
            x, y, width, height = client(self.hwnd)
            bx, by, bw, bh = absolute_box(self.config, name)
            point = W.POINT(round(x + (bx + bw/2)*width), round(y + (by + bh/2)*height))
            if u.GetAncestor(u.WindowFromPoint(point), 2) != u.GetAncestor(self.hwnd, 2):
                raise RuntimeError('按钮被其他窗口遮挡，已取消点击。')
            u.SetCursorPos(point.x, point.y)
            u.mouse_event(0x0002, 0, 0, 0, 0); u.mouse_event(0x0004, 0, 0, 0, 0)
            # Move off the button to avoid interpreting hover as a playback change.
            u.SetCursorPos(x + 3, y + 3)
            self.deadline = time.monotonic() + 8
            self.matches = 0
            self.root.after(350, lambda: self.poll(name))
        except Exception as exc:
            self.busy = False; self.report(str(exc))

    def poll(self, name):
        if not self.busy: return
        try:
            if u.GetAncestor(u.GetForegroundWindow(), 2) != u.GetAncestor(self.hwnd, 2):
                raise RuntimeError('窗口失去前台焦点，验证已停止。')
            state, idle, active = self.classify(name, capture(self.hwnd))
            self.matches = self.matches + 1 if state == 'active' else 0
            if self.matches >= 3:
                self.busy = False
                self.report(f'{name}：连续 3 次匹配播放中图标，界面确认播放已开始。尚未检测声音。')
            elif time.monotonic() >= self.deadline:
                self.busy = False
                self.report(f'{name}：8 秒内未确认播放。不会重试点击；请检查远程延迟、遮挡和样本。')
            else:
                self.report(f'{name} 验证中：{state}，停止差异 {idle:.1f}，播放差异 {active:.1f}')
                self.root.after(300, lambda: self.poll(name))
        except Exception as exc:
            self.busy = False; self.report(str(exc))

    def stop(self):
        self.busy = False
        self.report('验证已停止。远程音频可能仍在播放。')

if __name__ == '__main__':
    App().root.mainloop()
