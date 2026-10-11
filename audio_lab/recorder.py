"""Independent loopback recorder. No click application imports or UI control."""
from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import wave

ROOT = Path(__file__).resolve().parent


class Recorder:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('独立音频实验 · 手动录制 Input / A / B')
        self.root.geometry('700x390')
        self.session = ROOT/'sessions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.session.mkdir(parents=True)
        self.thread = None
        self.stop = threading.Event()
        self.events = queue.Queue()
        self.label = tk.StringVar(value='Input')
        self.status = tk.StringVar(value='先开始录音，再手动播放一段完整音频；播完后点停止。')
        ttk.Label(self.root,text='只录本机输出声音，不操作向日葵。请分别录一遍完整的 Input、A、B。',wraplength=660).pack(pady=10)
        self.device = ttk.Combobox(self.root,state='readonly',width=75)
        self.device.pack(padx=12)
        self.speakers = []
        ttk.Button(self.root,text='刷新输出设备',command=self.refresh).pack(pady=5)
        row = ttk.Frame(self.root); row.pack(pady=10)
        ttk.Combobox(row,textvariable=self.label,values=('Input','A','B'),state='readonly',width=10).pack(side='left')
        ttk.Button(row,text='开始录这一段',command=self.start).pack(side='left',padx=8)
        ttk.Button(row,text='停止并保存',command=self.stop.set).pack(side='left')
        ttk.Label(self.root,textvariable=self.status,wraplength=660).pack(pady=8)
        ttk.Label(self.root,text=f'本题录音目录：\n{self.session}',wraplength=660).pack(pady=8)
        ttk.Label(self.root,text='录的是所选设备的系统声音；通知、其他播放器声音也会被录入。\n每次启动新建一题；同名文件不会覆盖。',wraplength=660).pack()
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        self.root.after(100,self.drain)
        self.refresh()

    def refresh(self):
        if self.thread and self.thread.is_alive(): return
        try:
            import soundcard as sc
            self.speakers = sc.all_speakers()
            self.device['values'] = [s.name for s in self.speakers]
            if self.speakers: self.device.current(0)
        except Exception as exc: self.status.set(f'音频设备加载失败：{exc}')

    def start(self):
        if self.thread and self.thread.is_alive(): return
        index = self.device.current()
        if index < 0: self.status.set('请先选择实际播放向日葵声音的输出设备。'); return
        label = self.label.get()
        path = self.session/f'{label}.wav'
        if path.exists():
            self.status.set(f'{label} 已保存。请选下一段；重录时先另存旧文件或重启新建一题。'); return
        self.stop.clear()
        speaker = self.speakers[index]
        def worker():
            try:
                import numpy as np
                import soundcard as sc
                microphone = sc.get_microphone(id=speaker.id,include_loopback=True)
                # Windows loopback uses stereo; single-channel capture can be
                # corrupted according to SoundCard's documented WASAPI caveat.
                frames = 0
                with microphone.recorder(samplerate=48000,channels=2,blocksize=4800) as source, wave.open(str(path),'wb') as out:
                    out.setnchannels(2); out.setsampwidth(2); out.setframerate(48000)
                    self.events.put(f'正在录 {label}：现在手动播放；完整结束后点停止。')
                    while not self.stop.is_set() and frames < 48000*600:
                        block = source.record(numframes=4800)
                        pcm = (np.clip(block,-1,1)*32767).astype('<i2')
                        out.writeframes(pcm.tobytes())
                        frames += len(block)
                metadata = {'label':label,'device':speaker.name,'sample_rate':48000,'channels':2,'seconds':frames/48000,
                            'source':'host_system_output_loopback','boundary':'manual','complete_clip_confirmed':False}
                path.with_suffix('.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
                self.events.put(f'已保存 {label}，{frames/48000:.1f} 秒。请回听确认完整，再录下一段。')
            except Exception as exc:
                self.events.put(f'录音失败：{exc}；若生成不完整文件，请检查后另存，再重录。')
        self.thread = threading.Thread(target=worker,daemon=True)
        self.thread.start()

    def drain(self):
        while not self.events.empty(): self.status.set(self.events.get_nowait())
        self.root.after(100,self.drain)

    def close(self):
        self.stop.set()
        if self.thread and self.thread.is_alive():
            self.status.set('正在保存录音，请稍候…')
            self.root.after(200,self.close)
        else: self.root.destroy()


if __name__ == '__main__':
    app = Recorder()
    app.root.mainloop()
