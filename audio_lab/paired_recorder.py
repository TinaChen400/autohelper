"""Independent audio + screenshot observer UI. Does not click remote controls."""
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from assistant import windows
from dynamic_app import grab
from vision import Detector
from playback import read_players
import cv2


class Panel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('独立实验：页面与录音同步对应')
        self.root.geometry('740x300')
        self.items = windows()
        self.combo = ttk.Combobox(self.root, state='readonly', width=95,
                                 values=[f'{title} [{hwnd}]' for hwnd,title in self.items])
        self.combo.pack(padx=10,pady=10)
        matches = [i for i,(_,title) in enumerate(self.items) if title.strip() == 'KVMQ1']
        if len(matches)==1: self.combo.current(matches[0])
        self.status = tk.StringVar(value='选择实际向日葵被控画面窗口，点开始，然后切回向日葵播放 Input→A→B。')
        ttk.Label(self.root,textvariable=self.status,wraplength=700).pack(pady=12)
        ttk.Button(self.root,text='开始同步录音与页面识别',command=self.start).pack(pady=5)
        ttk.Button(self.root,text='结束并保存',command=self.stop).pack(pady=5)
        ttk.Label(self.root,text='只观察画面与录音，不移动鼠标、不点击、不提交。\n遮挡或识别失败会记录为未知，恢复画面后继续；最长10分钟。').pack(pady=10)
        self.folder = None
        self.worker = None
        self.message = ''
        self.root.after(200,self.refresh)
        self.root.protocol('WM_DELETE_WINDOW',self.close)

    def refresh(self):
        if self.message: self.status.set(self.message)
        self.root.after(200,self.refresh)

    def start(self):
        if self.worker and self.worker.is_alive(): return
        if self.combo.current()<0:
            self.message='请先选择被控画面窗口。'; return
        hwnd,title=self.items[self.combo.current()]
        folder=ROOT/'sessions'/('paired_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        folder.mkdir(parents=True)
        self.folder=folder
        (folder/'screen').mkdir()
        python=ROOT/'.venv'/'Scripts'/'python.exe'
        self.error_stream=(folder/'audio_error.log').open('w',encoding='utf-8')
        self.audio=subprocess.Popen([str(python),str(ROOT/'live_capture.py'),'--folder',str(folder)],
                                    stdout=subprocess.DEVNULL,stderr=self.error_stream,creationflags=subprocess.CREATE_NO_WINDOW)
        self.worker=threading.Thread(target=self.observe,args=(folder,hwnd,title),daemon=True)
        self.worker.start()

    def observe(self,folder,hwnd,title):
        try:
            detector=Detector()
            previous=None
            index=0
            deadline=time.monotonic()+600
            with (folder/'timeline.jsonl').open('a',encoding='utf-8') as timeline:
                while time.monotonic()<deadline and not (folder/'STOP').exists():
                    status_path=folder/'capture_status.json'
                    try:
                        status=json.loads(status_path.read_text(encoding='utf-8'))
                    except (OSError,ValueError):
                        time.sleep(.2); continue
                    if status['state'] in ('failed','saved'): break
                    zero=status.get('audio_zero_monotonic')
                    if zero is None:
                        time.sleep(.2); continue
                    t0=time.monotonic()
                    row={'window':title,'audio_seconds':max(0,t0-zero),'observed_monotonic':t0}
                    try:
                        image,_=grab(hwnd)
                        screenshot_at=time.monotonic()
                        targets,issues,elapsed=detector.detect(image)
                        readings=read_players(detector,image,targets)
                        active=[name for name,r in readings.items() if r.state=='active']
                        label=active[0] if len(active)==1 and len(readings)==3 else 'unknown'
                        if not active and len(readings)==3 and all(r.state=='idle' for r in readings.values()): label='idle'
                        row.update(label=label,readings={k:asdict(v) for k,v in readings.items()},issues=issues,
                                   capture_end_seconds=screenshot_at-zero,observation_seconds=time.monotonic()-t0)
                        # Retain visual evidence for transitions, including first detection.
                        if label!=previous:
                            filename=f'screen/{index:05d}_{label}.png'
                            cv2.imwrite(str(folder/filename),image)
                            row['screenshot']=filename
                        previous=label
                        self.message=f'录音中：{row["audio_seconds"]:.1f} 秒；画面：{label}。请保持向日葵画面可见。'
                    except Exception as exc:
                        row.update(label='unknown',error=str(exc))
                        previous=None
                        self.message=f'录音继续，页面暂时未知：{exc}'
                    timeline.write(json.dumps(row,ensure_ascii=False)+'\n'); timeline.flush()
                    index+=1
                    time.sleep(.25)
        except Exception as exc:
            self.message=f'页面识别失败，录音已请求保存：{exc}'
        finally:
            (folder/'STOP').touch()
            self.audio.wait(timeout=15)
            self.error_stream.close()
            self.message=f'已结束并保存：{folder}。边界仍须核对，未生成评价。'

    def stop(self):
        if self.folder: (self.folder/'STOP').touch()

    def close(self):
        self.stop()
        if self.worker and self.worker.is_alive(): self.root.after(200,self.close)
        else: self.root.destroy()


if __name__=='__main__':
    Panel().root.mainloop()
