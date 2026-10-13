"""Screenshot-only remote player assistant. No browser APIs or DOM access."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog

import cv2
from PIL import Image, ImageTk

from assistant import capture, client, windows, u, W
from vision import Detector, frame_bgr, glyph_state, annotate, stable_target, scene_difference
from human_mouse import HumanMouse
from workflow import Workflow, Settings

ROOT = Path(__file__).resolve().parent


def foreground(hwnd):
    return u.GetAncestor(u.GetForegroundWindow(), 2) == u.GetAncestor(hwnd, 2)


def visible_at(hwnd, point):
    return u.GetAncestor(u.WindowFromPoint(point), 2) == u.GetAncestor(hwnd, 2)


def park_pointer(hwnd, label=None, shape=None):
    """Move INSIDE the remote content, so Sunlogin forwards the mouse move.

    Moving to the host title bar leaves the remote pointer on the play icon.
    No click is generated here.
    """
    if not foreground(hwnd): return
    x, y, w, h = client(hwnd)
    fx, fy = .72, .3
    if label is not None and shape is not None:
        lx, ly, lw, lh = label
        fx, fy = (lx+lw/2)/shape[1], (ly+lh/2)/shape[0]
    point = W.POINT(round(x+w*fx), round(y+h*fy))
    if visible_at(hwnd, point): u.SetCursorPos(point.x, point.y)


def grab(hwnd):
    if not u.IsWindow(hwnd) or u.IsIconic(hwnd):
        raise RuntimeError('远程窗口不可见或已最小化，等待画面恢复。')
    geometry = client(hwnd)
    x, y, w, h = geometry
    # Check multiple points; a covered window must not be interpreted as the page.
    for fx, fy in ((.1, .1), (.5, .1), (.9, .1), (.1, .5), (.5, .5), (.9, .5), (.1, .9), (.5, .9), (.9, .9)):
        if not visible_at(hwnd, W.POINT(round(x+w*fx), round(y+h*fy))):
            raise RuntimeError('向日葵画面存在遮挡，已暂停。请移开遮挡窗口。')
    image = frame_bgr(capture(hwnd, max_width=1920, max_height=1200))
    if client(hwnd) != geometry:
        raise RuntimeError('截图时窗口发生移动或缩放，请重试。')
    return image, geometry


def patches_match(first, second, box, limit=6):
    if first.shape != second.shape: return False
    x, y, w, h = box
    return scene_difference(first[y:y+h, x:x+w], second[y:y+h, x:x+w]) <= limit


class DynamicApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('向日葵播放助手 v0.14 · 短听暂停遮挡恢复')
        self.root.geometry('1100x900')
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.messages = queue.Queue()
        self.cancel = threading.Event()
        self.generation = 0
        self.detector = None
        self.last = None
        self.items = []
        self.hwnd = None
        self.status = tk.StringVar(value='选择向日葵窗口，点击“开始实时识别”。初次加载 OCR 需要等待。')
        row = ttk.Frame(self.root); row.pack(fill='x', padx=8, pady=8)
        self.combo = ttk.Combobox(row, state='readonly', width=65); self.combo.pack(side='left', fill='x', expand=True)
        ttk.Button(row, text='刷新窗口', command=self.refresh).pack(side='left')
        row = ttk.Frame(self.root); row.pack(fill='x', padx=8)
        ttk.Button(row, text='开始实时识别', command=self.monitor).pack(side='left')
        ttk.Button(row, text='停止', command=self.stop).pack(side='left')
        ttk.Button(row, text='离线查看截图／视频', command=self.offline).pack(side='left')
        ttk.Button(row, text='保存当前识别结果', command=self.save).pack(side='left')
        config = ttk.LabelFrame(self.root, text='首轮 B 结束 → 约 1 秒后重播 Input → 随机重听 A/B → 下滚查看要求')
        config.pack(fill='x', padx=8, pady=5)
        self.replay_min = tk.StringVar(value='2'); self.replay_max = tk.StringVar(value='4')
        self.pause_min = tk.StringVar(value='1'); self.pause_max = tk.StringVar(value='3')
        self.scroll_max = tk.StringVar(value='12')
        self.after_b_pause = tk.StringVar(value='1')
        for label, variable in [('重听最少', self.replay_min), ('最多', self.replay_max), ('停顿最少秒', self.pause_min), ('最多秒', self.pause_max), ('最多滚动段', self.scroll_max)]:
            ttk.Label(config, text=label).pack(side='left', padx=3)
            ttk.Entry(config, textvariable=variable, width=4).pack(side='left')
        ttk.Button(config, text='开始连续流程（本题）', command=self.run_workflow).pack(side='left', padx=8)
        timing = ttk.Frame(self.root); timing.pack(fill='x', padx=8)
        ttk.Label(timing, text='首轮 B 结束后，重播 Input 的停顿（秒）：').pack(side='left')
        ttk.Entry(timing, textvariable=self.after_b_pause, width=5).pack(side='left')
        ttk.Label(timing, text='重听次数含第一遍 Input；后续随机 A/B。以实际计时和图标为准。').pack(side='left', padx=8)
        short = ttk.Frame(self.root); short.pack(fill='x', padx=8, pady=3)
        self.short_enabled = tk.BooleanVar(value=True)
        self.scroll_after = tk.BooleanVar(value=True)
        self.short_count = tk.StringVar(value='6')
        self.short_min = tk.StringVar(value='0.4')
        self.short_max = tk.StringVar(value='1.2')
        ttk.Checkbutton(short, text='追加开头随机短听', variable=self.short_enabled).pack(side='left')
        for label, variable in [('次数', self.short_count), ('最短秒', self.short_min), ('最长秒', self.short_max)]:
            ttk.Label(short, text=label).pack(side='left', padx=3)
            ttk.Entry(short, textvariable=variable, width=5).pack(side='left')
        ttk.Checkbutton(short, text='结束后下滚查看要求', variable=self.scroll_after).pack(side='left', padx=10)
        ttk.Label(self.root, textvariable=self.status, wraplength=1020).pack(fill='x', padx=8, pady=6)
        self.canvas = tk.Canvas(self.root, background='#202020', height=540)
        self.canvas.pack(fill='both', expand=True, padx=8)
        self.summary = tk.StringVar(value='尚未识别。绿色框：标签；橙色框：播放按钮。')
        ttk.Label(self.root, textvariable=self.summary, wraplength=1020).pack(fill='x', padx=8, pady=6)
        self.dimensions = tk.StringVar(value='评价维度：连续流程结束时读取；此版本不填写或提交。')
        ttk.Label(self.root, textvariable=self.dimensions, wraplength=1020).pack(fill='x', padx=8, pady=3)
        row = ttk.Frame(self.root); row.pack(fill='x', padx=8, pady=6)
        for name in ('Input', 'A', 'B'):
            ttk.Button(row, text=f'播放 {name} 并验证', command=lambda n=name: self.play(n)).pack(side='left', padx=4)
        ttk.Label(self.root, text='连续流程会控制鼠标：移动鼠标、切换窗口或按 Esc 即暂停。尚不采集音频、不评价、不提交。').pack(pady=4)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()
        self.root.after(100, self.drain)

    def refresh(self):
        old_index=self.combo.current()
        previous=self.items[old_index][0] if 0<=old_index<len(self.items) else None
        # Older helper instances also contain “向日葵” in their titles. They
        # must never become the remote-control target after a refresh/restart.
        self.items = [(hwnd,title) for hwnd,title in windows()
                      if not title.startswith(('向日葵助手','向日葵播放助手'))]
        self.combo['values'] = [f'{title} [{hwnd}]' for hwnd, title in self.items]
        if self.items:
            selected=next((i for i,(hwnd,_) in enumerate(self.items) if hwnd==previous),None)
            if selected is None:
                selected=next((i for i,(_,title) in enumerate(self.items) if title.upper().startswith('KVM')),0)
            self.combo.current(selected)

    def select_window(self):
        i = self.combo.current()
        if i < 0: raise RuntimeError('请先选择向日葵远程控制窗口。')
        hwnd, title = self.items[i]
        if u.GetAncestor(hwnd, 2) == u.GetAncestor(self.root.winfo_id(), 2):
            raise RuntimeError('不能选择助手自身。请选择实际向日葵远程控制窗口。')
        self.hwnd = hwnd
        return hwnd

    def start_task(self, action):
        self.cancel.set()
        self.cancel = threading.Event()
        self.generation += 1
        token, event = self.generation, self.cancel
        def send(kind, payload):
            self.messages.put((token, kind, payload))
        def work():
            try:
                if event.is_set(): return
                if self.detector is None and not getattr(self,'defer_detector_loading',False):
                    send('status', '加载本地 OCR 模型…')
                    self.detector = Detector()
                if not event.is_set(): action(event, send)
            except Exception as exc:
                send('status', f'已停止：{exc}')
        self.task_future=self.pool.submit(work)

    def activate(self, hwnd):
        u.ShowWindow(hwnd, 9)
        u.SetForegroundWindow(hwnd)

    def run_workflow(self):
        try:
            settings = Settings(int(self.replay_min.get()), int(self.replay_max.get()),
                                float(self.pause_min.get()), float(self.pause_max.get()), int(self.scroll_max.get()),
                                float(self.after_b_pause.get()), short_enabled=self.short_enabled.get(),
                                short_count=int(self.short_count.get()), short_min=float(self.short_min.get()),
                                short_max=float(self.short_max.get()), scroll_after=self.scroll_after.get())
            settings.validate()
            hwnd = self.select_window()
            self.activate(hwnd)
        except Exception as exc:
            self.status.set(f'无法开始：{exc}'); return
        self.dimensions.set('评价维度：等待首轮播放与重听完成。')
        def work(event, send):
            if event.wait(.6): return
            Workflow(hwnd, self.detector, event, send, grab, settings, ROOT).run()
        self.start_task(work)

    def monitor(self):
        try:
            hwnd = self.select_window()
            self.activate(hwnd)
        except Exception as exc:
            self.status.set(str(exc)); return
        def work(event, send):
            if event.wait(.6): return
            park_pointer(hwnd)
            if event.wait(.4): return
            last_image, targets, issues = None, {}, []
            ocr_at, last_geometry = 0, None
            while not event.is_set():
                image, geometry = grab(hwnd)
                changed = last_image is None or geometry != last_geometry or scene_difference(last_image, image) > 1.2
                # Periodically refresh OCR even if a small layout change escapes
                # the scene threshold. Tracking is never used to authorize a click.
                if changed or time.monotonic()-ocr_at > 2:
                    targets, issues, seconds = self.detector.detect(image)
                    ocr_at = time.monotonic()
                    mode = f'OCR {seconds:.2f} 秒'
                else:
                    targets = {n: replace(t, state=glyph_state(image, t.button)) for n, t in targets.items()}
                    mode = '局部图标跟踪'
                if event.is_set(): return
                send('result', (image, targets, issues, mode, 'live'))
                last_image, last_geometry = image, geometry
                if event.wait(.2): return
        self.start_task(work)

    def play(self, name):
        try:
            hwnd = self.select_window()
            self.activate(hwnd)
        except Exception as exc:
            self.status.set(str(exc)); return
        def work(event, send):
            send('status', f'{name}：正在点击前重新识别，两次核对通过后才点击。')
            if event.wait(.6): return
            park_pointer(hwnd)
            if event.wait(.4): return
            mouse = HumanMouse(hwnd, event)
            from vision import locate_players
            first, targets1, _, _ = locate_players(self.detector, lambda: grab(hwnd)[0], mouse, lambda text: send('status', text))
            _, geo1 = grab(hwnd)
            if event.is_set(): return
            if name not in targets1: raise RuntimeError(f'{name} 标签与按钮对应不明确，没有点击。')
            if targets1[name].state != 'idle': raise RuntimeError(f'{name} 未处于可识别的停止状态，没有点击。')
            if event.wait(.2): return
            second, geo2 = grab(hwnd)
            targets2, issues, seconds = self.detector.detect(second)
            if event.is_set(): return
            if name not in targets2 or not stable_target(targets1[name], targets2[name]):
                raise RuntimeError('两次识别结果不一致，没有点击。')
            if geo1 != geo2 or scene_difference(first, second) > 2:
                raise RuntimeError('页面或窗口仍在变化，没有点击。')
            target = targets2[name]
            # OCR takes time. Grab a new frame immediately before the input event.
            fresh, geo3 = grab(hwnd)
            if geo3 != geo2 or scene_difference(second, fresh) > 2:
                raise RuntimeError('识别期间页面或窗口变化，没有点击。')
            if not patches_match(second, fresh, target.label) or glyph_state(fresh, target.button) != 'idle':
                raise RuntimeError('标签或播放图标已变化，没有点击。')
            bx, by, bw, bh = target.button
            x, y, w, h = geo3
            point = W.POINT(round(x+(bx+bw/2)/fresh.shape[1]*w), round(y+(by+bh/2)/fresh.shape[0]*h))
            if not foreground(hwnd) or not visible_at(hwnd, point):
                raise RuntimeError('目标失去焦点或被遮挡，没有点击。')
            if event.is_set(): return
            def before_press():
                latest, geometry = grab(hwnd)
                # The remote pointer may cover the icon once the mouse moves
                # over it. The preceding unobstructed idle check remains valid
                # only while the label and scene remain stable.
                if (geometry != geo3 or scene_difference(fresh, latest) > 2
                        or not patches_match(fresh, latest, target.label)
                        or glyph_state(latest, target.button) == 'active'):
                    raise RuntimeError('点击前页面变化，已取消点击。')
            mouse.click_box(target.button, fresh.shape, before_press=before_press)
            lx, ly, lw, lh = target.label
            mouse.move((min(fresh.shape[1]*.97, lx+lw+bw*2), ly+lh/2), fresh.shape)
            send('status', f'{name} 已点击一次，正在检查暂停图标。')
            deadline, matches = time.monotonic()+8, 0
            while time.monotonic() < deadline:
                if event.wait(.3): return
                image, geometry = grab(hwnd)
                if geometry != geo3 or not patches_match(fresh, image, target.label):
                    raise RuntimeError('验证时窗口或目标标签变化，无法确认；不会重复点击。')
                if scene_difference(fresh, image) > 5:
                    raise RuntimeError('验证时页面发生较大变化，无法确认；不会重复点击。')
                state = glyph_state(image, target.button)
                matches = matches+1 if state == 'active' else 0
                targets2[name] = replace(target, state=state)
                send('result', (image, targets2, issues, f'{name} 播放验证 {matches}/3', 'live'))
                if matches >= 3:
                    send('status', f'{name}：连续三次识别到暂停图标，界面确认播放开始；尚未检测声音。')
                    return
            raise RuntimeError(f'{name} 已点击，但 8 秒内未看清暂停图标；音频可能已播放。不会重复点击。请移开鼠标或放大向日葵画面后重试。')
        self.start_task(work)

    def offline(self):
        path = filedialog.askopenfilename(initialdir=ROOT/'sample', filetypes=[('截图或视频', '*.png *.jpg *.jpeg *.bmp *.mp4'), ('全部文件', '*.*')])
        if not path: return
        def work(event, send):
            if Path(path).suffix.lower() == '.mp4':
                video = cv2.VideoCapture(path)
                try:
                    video.set(cv2.CAP_PROP_POS_FRAMES, int(video.get(cv2.CAP_PROP_FRAME_COUNT))//3)
                    success, image = video.read()
                    if not success: raise RuntimeError('无法读取视频。')
                finally: video.release()
            else: image = cv2.imread(path)
            if image is None: raise RuntimeError('无法读取截图。')
            targets, issues, seconds = self.detector.detect(image)
            if not event.is_set(): send('result', (image, targets, issues, f'离线 OCR {seconds:.2f} 秒', 'offline'))
        self.start_task(work)

    def render(self, payload):
        image, targets, issues, mode, source = payload
        self.last = payload
        shown = annotate(image, targets)
        rgb = cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)
        picture = Image.fromarray(rgb)
        picture.thumbnail((max(200, self.canvas.winfo_width()), max(200, self.canvas.winfo_height())))
        self.photo = ImageTk.PhotoImage(picture)
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        labels = [f'{"参考音频 Input" if n == "Input" else "回答 " + n}: {"就绪（可播放）" if t.state == "idle" else "播放中（暂停图标）" if t.state == "active" else "锁定，需先播放前一段" if t.state == "locked" else "播放状态尚未确认"}' for n, t in targets.items()]
        self.summary.set('；'.join(labels+issues) or '未识别到目标。')
        self.status.set(f'{mode}。' + ('离线结果不能作为点击依据；点击时始终重新识别实际窗口。' if source == 'offline' else '移动鼠标或按 Esc 可暂停。' if source == 'workflow' else '识别框仅供预览，播放操作会再次核对。'))

    def drain(self):
        # Avoid displaying a backlog of obsolete preview frames.
        pending = []
        while True:
            try:
                token, kind, payload = self.messages.get_nowait()
                if token == self.generation: pending.append((kind, payload))
            except queue.Empty: break
        for kind, payload in pending:
            if kind == 'status': self.status.set(payload)
            elif kind == 'health_event':
                handler=getattr(self,'handle_health_event',None)
                if handler:handler(payload)
            elif kind == 'dimensions':
                self.dimensions.set('已读取评价维度：'+'；'.join(row['name'] for row in payload) if payload else '暂未读到完整评价行。')
            else: self.render(payload)
        self.root.after(100, self.drain)

    def save(self):
        if self.last is None: return self.status.set('还没有识别结果。')
        image, targets, issues, mode, source = self.last
        directory = ROOT/'artifacts'; directory.mkdir(exist_ok=True)
        stamp = time.strftime('%Y%m%d_%H%M%S')
        path = directory/f'preview_{stamp}.png'
        cv2.imwrite(str(path), annotate(image, targets))
        path.with_suffix('.json').write_text(json.dumps({'source': source, 'mode': mode, 'targets': {n: asdict(t) for n, t in targets.items()}, 'issues': issues}, ensure_ascii=False, indent=2), encoding='utf-8')
        self.status.set(f'已保存：{path}')

    def stop(self):
        self.cancel.set(); self.generation += 1
        self.status.set('已停止识别／点击验证。已开始的远程音频不会自动暂停。')

    def close(self):
        self.cancel.set(); self.pool.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


if __name__ == '__main__':
    DynamicApp().root.mainloop()
