"""Phase 1: follow autoplay, replay, read rating rows, stop. No audio scoring."""
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import random
import time
import cv2

from human_mouse import HumanMouse, Cancelled
from playback import Reading, RoundTracker, read_players, PlaybackEvidenceError
from rating_reader import read_rating_page
from vision import stable_target, glyph_state, scene_difference, locate_players


class ClickRecheck(RuntimeError):
    """A pre-press check failed; no mouse-down has occurred."""


class ControlChanged(RuntimeError):
    """A short-preview control changed before mouse-down."""


class ReplayFinished(RuntimeError):
    """Analysis finished before a supplemental mouse press."""


class LocationUnconfirmed(RuntimeError):
    """Read-only location may be attempted again before any playback."""


class PlaybackDidNotStart(PlaybackEvidenceError):
    pass


class PageUnconfirmed(RuntimeError):
    """One changed frame is re-observed, never used for a mouse press."""


@dataclass
class Settings:
    replay_min: int = 2
    replay_max: int = 4
    pause_min: float = 1.0
    pause_max: float = 3.0
    max_scrolls: int = 12
    after_b_pause: float = 1.0
    short_enabled: bool = False
    short_count: int = 6
    short_min: float = .4
    short_max: float = 1.2
    scroll_after: bool = True
    player_order: tuple = ('Input','A','B')
    auto_mode: bool = False

    def validate(self):
        if self.player_order not in (('Input','A','B'),('A','B')): raise ValueError('不支持的题型。')
        if not 1 <= self.replay_min <= self.replay_max <= 12: raise ValueError('重听次数需为 1～12（第一遍固定重听 Input），最小值不能超过最大值。')
        if not 0 <= self.pause_min <= self.pause_max <= 15: raise ValueError('停顿需为 0～15 秒，最小值不能超过最大值。')
        if not 1 <= self.max_scrolls <= 30: raise ValueError('最多滚动段数需为 1～30。')
        if not 0 <= self.after_b_pause <= 10: raise ValueError('B 结束后停顿需为 0～10 秒。')
        if not 3 <= self.short_count <= 30: raise ValueError('短听次数需为 3～30。')
        if not .3 <= self.short_min < self.short_max <= 3: raise ValueError('短听时间需为 0.3～3 秒，最短必须小于最长。')


class Workflow:
    def __init__(self, hwnd, detector, event, send, grab, settings, root):
        settings.validate()
        self.hwnd, self.detector, self.event, self.send, self.grab = hwnd, detector, event, send, grab
        self.settings = settings
        self.rng = random.Random()
        self.mouse = HumanMouse(hwnd, event, self.rng)
        self.mouse.on_pause = lambda text: self.status(text)
        self.mouse.on_interference = lambda **data:self.log('manual_input_pause',**data)
        self.directory = Path(root)/'artifacts'/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.targets = {}
        self.identity = None
        self.last_image = None
        self.last_readings = {}
        self.known_readings = {}
        self.locked_start = False
        self.log('session_start', settings=asdict(settings))

    def log(self, kind, **data):
        with (self.directory/'events.jsonl').open('a', encoding='utf-8') as file:
            file.write(json.dumps({'time': datetime.now().isoformat(), 'event': kind,
                **({'task_id':self.task_tracking['task_id']} if getattr(self,'task_tracking',None) else {}),
                **data}, ensure_ascii=False)+'\n')

    def status(self, text):
        self.send('status', text)

    def snapshot(self):
        if getattr(self,'observing_playback',False):
            getattr(self.mouse,'check_observation',self.mouse.check)()
        else:self.mouse.check()
        self.last_image = self.grab(self.hwnd)[0]
        return self.last_image

    def preview(self, image, readings, message):
        from dataclasses import replace
        targets = {name: replace(target, state=readings[name].state) if name in readings else target for name, target in self.targets.items()}
        self.send('result', (image, targets, [], message, 'workflow'))

    def parked(self, image, target=None):
        if target:
            x, y, w, h = target.label
            point = (min(image.shape[1]*.94, x+w+self.rng.uniform(2, 7)*target.button[2]), y+self.rng.uniform(.1, .9)*h)
        else:
            region=self.rng.choice(((.06,.18,.20,.72),(.80,.93,.22,.70),(.30,.70,.12,.22)))
            point=(image.shape[1]*self.rng.uniform(*region[:2]),image.shape[0]*self.rng.uniform(*region[2:]))
        self.mouse.move(point, image.shape,slow=target is None)
        self.mouse.wait(self.rng.uniform(.10,.18))

    def identify(self):
        began=time.monotonic()
        order=self.settings.player_order
        args=(self.detector, self.snapshot, self.mouse, self.status)
        if self.settings.auto_mode and not getattr(self,'mode_detected',False):
            image,targets,issues,_=locate_players(*args,required=None)
            from vision import classify_task
            detected,evidence=classify_task(targets,getattr(self.detector,'last_task_lines',[]),image.shape)
            if detected is None:
                raise LocationUnconfirmed('未能确认题型，没有点击。'+ '；'.join(evidence+issues))
            self.settings.player_order=order=detected
            self.mode_detected=True
            self.log('task_mode_detected',player_order=list(order),evidence=evidence)
            self.status('自动识别题型：'+('仅 A/B（无参考音频）' if len(order)==2 else 'Input/A/B（有参考音频）'))
        else:
            image, targets, issues, _ = locate_players(*args) if 'Input' in order else locate_players(*args,required=order)
        targets={name:target for name,target in targets.items() if name in order}
        self.mouse.check()
        self.targets = targets
        self.locked_start = any(t.state == 'locked' for t in targets.values())
        self.preview(image, {}, '识别当前题目播放器')
        if set(targets) != set(order):
            raise LocationUnconfirmed('本次识别未能确认全部播放器（可见不代表 OCR 已读到）。'+ '；'.join(issues))
        # Static transcript strip changes on a new task, not during playback.
        a = targets[order[0]].button
        boundary = min(targets['A'].label[1], targets['B'].label[1])
        left = max(0, targets[order[0]].label[0]-a[2])
        bottom = boundary-4
        top = a[1]+a[3]+3 if 'Input' in order else max(0,bottom-round(image.shape[0]*.23))
        self.identity = (left, top, image.shape[1]-left, bottom-top) if bottom-top > 8 else None
        self.reference = image.copy()
        cv2.imwrite(str(self.directory/'initial.png'), image)
        readings = read_players(self.detector, image, targets)
        self.known_readings = readings.copy()
        self.log('player_durations', readings={n: asdict(r) for n, r in readings.items()})
        self.log('player_location_timing',seconds=round(time.monotonic()-began,3),
                 attempts=getattr(self.detector,'last_search_profile',[]))
        return image, readings

    def check_page(self, image):
        if image.shape != self.reference.shape: raise RuntimeError('页面尺寸变化，停止本题。')
        boxes = [target.label for name,target in self.targets.items() if not self.locked_start or name == self.settings.player_order[0]]
        if self.identity: boxes.append(self.identity)
        def matches(dx,dy):
            for x,y,w,h in boxes:
                if x+dx<0 or y+dy<0 or x+w+dx>image.shape[1] or y+h+dy>image.shape[0]:return False
                if scene_difference(self.reference[y:y+h,x:x+w],image[y+dy:y+h+dy,x+dx:x+w+dx])>9:return False
            return True
        if matches(0,0):return
        # Remote/browser scroll settling can move every static label and the
        # transcript by one pixel. Require one common offset for all regions,
        # against the original reference (no accumulated drift). A real scroll
        # or changed transcript must still fail before any click.
        if any(matches(dx,dy) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy):return
        raise PageUnconfirmed('题目内容或布局变化，已停止；不会继承上一题进度。')

    def page_snapshot(self):
        for attempt in range(4):
            image=self.snapshot()
            try:
                self.check_page(image)
                return image
            except PageUnconfirmed:
                if attempt==3:raise
                self.log('page_observation_retry',attempt=attempt+1)
                self.status('画面暂未核对清楚，等待短暂遮挡消失后重新观察；暂不点击。')
                # Keep sampling read-only during playback. Waiting for the
                # pointer to settle here would lose audio transitions again.
                wait=getattr(self.mouse,'wait_observation',self.mouse.wait) if getattr(self,'observing_playback',False) else self.mouse.wait
                wait(.15)

    def observe_players(self):
        """Refresh verified player pixels without re-OCRing the whole page."""
        from dataclasses import replace
        image=self.page_snapshot()
        readings=read_players(self.detector,image,self.targets)
        self.targets={name:replace(target,state=readings[name].state) for name,target in self.targets.items()}
        self.known_readings.update({n:r for n,r in readings.items() if r.total is not None})
        return image,readings

    def click(self, name, fast=False):
        for attempt in range(3):
            try:
                return self._click_once(name, fast=fast)
            except ClickRecheck as exc:
                self.log('click_recheck', target=name, attempt=attempt+1, reason=str(exc))
                image = self.snapshot()
                self.check_page(image)
                self.parked(image, self.targets[name])
                if attempt == 2:
                    raise RuntimeError(f'{name} 点击前连续三次核对未通过，未按下鼠标。详情见 click_recheck 日志。') from exc
                self.status(f'{name} 点击前状态暂不一致，已移开鼠标重新核对（{attempt+1}/3）。')
                self.mouse.wait(.35)

    def _click_once(self, name, fast=False):
        if getattr(self, 'supplemental', False) and self.stop_replays():
            raise ReplayFinished()
        if self.targets[name].state == 'locked':
            raise RuntimeError(f'{name} 尚未解锁，不点击灰色按钮。')
        self.status(f'准备自然点击 {name}，正在重新核对标签、状态和时间。')
        image = self.snapshot()
        self.check_page(image)
        # The continuous loop has kept the labels and transcript in view and
        # checked them on every frame. Reuse only those validated coordinates;
        # full OCR is unnecessary after B ends and would add seconds of latency.
        targets = self.targets.copy() if fast else self.detector.detect(image)[0]
        self.mouse.check()
        if name not in targets or not stable_target(self.targets[name], targets[name]):
            # Cached state can differ after a completed round; geometry remains decisive.
            old = self.targets[name]
            from dataclasses import replace
            if name not in targets or not stable_target(replace(old, state=targets[name].state), targets[name]):
                raise RuntimeError(f'{name} 的对应位置发生变化，没有点击。')
        readings = read_players(self.detector, image, targets)
        if any(reading.state == 'active' for reading in readings.values()):
            raise RuntimeError('仍有音频正在播放，没有点击。')
        if name not in readings or readings[name].state != 'idle': raise RuntimeError('目标不是停止状态，没有点击。')
        target = targets[name]
        # Keep the pointer off the icon for the last unobstructed observation.
        bx, by, bw, bh = target.button
        fresh = self.snapshot()
        self.check_page(fresh)
        if scene_difference(image, fresh) > 2 or glyph_state(fresh, target.button) != 'idle':
            raise RuntimeError('移动鼠标期间页面变化，没有点击。')
        def before_press():
            if getattr(self, 'supplemental', False) and self.stop_replays():
                raise ReplayFinished()
            latest = self.snapshot()
            self.check_page(latest)
            if scene_difference(fresh, latest) > 2:
                raise RuntimeError('即将点击时页面变化，已取消点击。')
            # A remote cursor is rendered into the screenshot and may obscure
            # this icon after moving onto it. Use its unchanged timer plus the
            # preceding clear idle sample; other players must still be idle.
            latest_readings = read_players(self.detector, latest, targets)
            if any(r.state != 'idle' for n, r in latest_readings.items() if n != name):
                raise ClickRecheck('其他播放器状态暂不确定：'+str({n: asdict(r) for n, r in latest_readings.items()}))
            now_reading, before = latest_readings[name], readings[name]
            timer_changed = (now_reading.current is not None and before.current is not None
                             and (now_reading.current, now_reading.total) != (before.current, before.total))
            known = self.known_readings.get(name)
            duration_known = before.total is not None or (known is not None and known.total is not None)
            if (not getattr(self,'supplemental',False)
                    and (before.total is None or before.current not in (0,before.total))):
                raise ClickRecheck('首轮起点未确认，未按下播放按钮。')
            if now_reading.state == 'active' or timer_changed or not duration_known:
                raise ClickRecheck(f'目标按下前核对：before={asdict(before)}; now={asdict(now_reading)}; duration_known={duration_known}')
        self.mouse.click_box(target.button, fresh.shape, before_press=before_press)
        self.log('play_click', target=name,click=getattr(self.mouse,'last_click',None))
        self.parked(fresh, target)

    def await_chain(self, expected, initial=None):
        previous=getattr(self,'observing_playback',False)
        self.observing_playback=True
        try:return self._await_chain(expected,initial)
        finally:self.observing_playback=previous

    def _await_chain(self, expected, initial=None):
        tracker = getattr(self, 'tracker_type', RoundTracker)(expected, self.known_readings)
        started = time.monotonic()
        readings = initial
        observed = {}
        guide_at = 0
        missing_since = {name: None for name in expected}
        unclear_since={name:None for name in expected}
        while True:
            if getattr(self,'supplemental',False) and self.stop_replays():
                self.status('模型结果已就绪，核对并暂停补听，准备填写评价。')
                self.quiet_players()
                self.log('replay_interrupted_by_model',expected=list(expected))
                return
            image = self.page_snapshot()
            # A caller's initial readings belong to its older screenshot. Never
            # attach them to this newly captured frame or recording timestamp.
            readings = read_players(self.detector, image, self.targets)
            now = time.monotonic()
            self.last_readings = {name: asdict(reading) for name, reading in readings.items()}
            self.log('playback_sample', expected=list(expected), elapsed=round(now-started, 3), readings=self.last_readings)
            for name in expected:
                reading = readings.get(name, Reading('unknown'))
                if reading.total is not None:
                    missing_since[name] = None
                    self.known_readings[name] = reading
                else:
                    if missing_since[name] is None: missing_since[name] = now
                    if tracker.segments[name].started and tracker.segments[name].total is None and now-missing_since[name] > 12:
                        raise PlaybackEvidenceError(f'{name} 连续 12 秒未读清总时长，无法建立计时。')
                segment=tracker.segments.get(name)
                unclear=bool(segment and segment.started and not segment.done and (reading.state=='unknown' or reading.current is None or reading.total is None))
                if unclear:
                    if unclear_since[name] is None:
                        unclear_since[name]=now
                        self.log('playback_reading_unclear',target=name)
                    if now-unclear_since[name]>=12:
                        raise PlaybackEvidenceError(f'{name} 连续 12 秒未看清播放图标或计时，重新核对完整首轮。')
                elif unclear_since[name] is not None:
                    self.log('playback_reading_recovered',target=name,seconds=round(now-unclear_since[name],3))
                    unclear_since[name]=None
            finished = tracker.observe(readings, now)
            guidance=getattr(self,'playback_guidance',None)
            if guidance:guidance(image,readings,now)
            if (getattr(self,'pending_play_start',None)==expected[0]
                    and not tracker.segments[expected[0]].started and now-started>=4):
                raise PlaybackDidNotStart(f'{expected[0]} 点击后 4 秒仍未确认开始播放。')
            for name, segment in tracker.segments.items():
                state = (segment.started, segment.done)
                if observed.get(name) != state:
                    self.log('playback_state', target=name, started=state[0], done=state[1], reading=asdict(readings[name]))
                    observed[name] = state
            message = tracker.summary(readings)
            self.preview(image, readings, message)
            self.status(message+('；人工操作时继续观察，Esc 停止。' if getattr(self,'observation_only_playback',False) else '；移动鼠标或按 Esc 可停止。'))
            if finished:
                self.log('chain_completed', expected=list(expected))
                cv2.imwrite(str(self.directory/f'chain_end_{time.time_ns()}.png'), image)
                return
            active = next((r for r in readings.values() if r.state == 'active'), None)
            if (active and self.identity and now-guide_at >= 5
                    and not getattr(self,'observation_only_playback',False)):
                x, y, w, h = self.identity
                # Approximate visual reading guidance; no word-level alignment.
                # Keep the cursor in transcript text, away from player controls.
                if h > 12 and active.current is not None and active.total:
                    progress = min(.99, active.current/active.total)
                    rows = max(1, min(5, int(h/24)))
                    row = min(rows-1, int(progress*rows))
                    fraction = (progress*rows) % 1
                    right = min(x+w, image.shape[1]*.85)
                    point = (x+max(1, right-x)*(.08+.8*fraction), y+min(h-4, 8+row*24))
                    self.mouse.move(point, image.shape, slow=not getattr(self,'fast_navigation',False),
                                    quick=getattr(self,'fast_navigation',False))
                guide_at = now
            # Autoplay may have an arbitrary gap. Keep observing instead of
            # timing out the round merely because the next item has not begun.
            # One-second clips need several actual samples, including their
            # autoplay transition. Keep full recording checks unchanged.
            short_transition=any(r.state=='active' and r.total is not None
                and (r.total<=2 or (r.current is not None and r.total-r.current<=1))
                for r in readings.values())
            getattr(self.mouse,'wait_observation',self.mouse.wait)(.08 if short_transition else .35)
            readings = None

    def settle_and_follow(self, selected):
        # A replay may or may not retain autoplay. Observe rather than assume.
        self.await_chain((selected,))
        if getattr(self,'supplemental',False) and self.stop_replays(): return
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if getattr(self,'supplemental',False) and self.stop_replays():
                self.quiet_players()
                return
            image = self.snapshot(); self.check_page(image)
            readings = read_players(self.detector, image, self.targets)
            active = [name for name, reading in readings.items() if reading.state == 'active']
            if active:
                order = self.settings.player_order
                index = order.index(active[0])
                self.status('重听触发了后续自动播放，等待其结束。')
                self.await_chain(order[index:], readings)
                if getattr(self,'supplemental',False) and self.stop_replays(): return
                deadline = time.monotonic()+3
            self.mouse.wait(.35)

    def inspect_ratings(self):
        found, stationary = {}, 0
        for step in range(self.settings.max_scrolls+1):
            image = self.snapshot()
            dimensions, submit = read_rating_page(self.detector, image)
            self.mouse.check()
            for dimension in dimensions:
                key = dimension['name'].lower().strip()
                found[key] = dimension
            cv2.imwrite(str(self.directory/f'rating_page_{step:02d}.png'), image)
            self.send('dimensions', list(found.values()))
            self.send('result', (image, {}, [], f'查看评价区域：已读到 {len(found)} 个维度', 'workflow'))
            self.log('rating_page', step=step, dimensions=dimensions, submit_visible=submit)
            if submit or step == self.settings.max_scrolls or stationary >= 2:
                reason = '已看到 Submit（未点击）' if submit else '达到滚动上限或页面不再移动'
                self.status(f'本轮结束：{reason}，已读到 {len(found)} 个维度；没有填写或提交。记录：{self.directory.name}')
                break
            self.status(f'分段下滚 {step+1}/{self.settings.max_scrolls}，仅查看要求。')
            self.mouse.scroll(image.shape, down=True, notches=self.rng.randint(1, 3))
            after = self.snapshot()
            stationary = stationary+1 if scene_difference(image, after) < .5 else 0
        (self.directory/'dimensions.json').write_text(json.dumps(list(found.values()), ensure_ascii=False, indent=2), encoding='utf-8')

    def preview_control(self, name, box, expected, reset=False):
        if expected=='idle' and getattr(self,'supplemental',False) and self.stop_replays(): raise ReplayFinished()
        image = self.snapshot(); self.check_page(image)
        target = self.targets[name]
        if glyph_state(image, target.button) != expected:
            raise ControlChanged('播放器状态已变化，重新观察而不点击。')
        if expected=='active':
            # Read the timer while the pointer is clear. OCR underneath the
            # pointer in before_press used slow fallback retries and could let
            # a short clip finish before the pause was actually pressed.
            reading=read_players(self.detector,image,{name:target})[name]
            margin=1 if getattr(self,'supplemental',False) and self.stop_replays() else 0
            if reading.current is not None and reading.total is not None and reading.total-reading.current<=margin:
                raise ControlChanged('已接近自然结束，观察下一段，不切换停止按钮。')
        def before():
            if expected=='idle' and getattr(self,'supplemental',False) and self.stop_replays(): raise ReplayFinished()
            latest = self.snapshot(); self.check_page(latest)
            state = glyph_state(latest, target.button)
            if state not in (expected, 'unknown'):
                raise ControlChanged('按下前状态已变化。')
            if scene_difference(image, latest) > 2:
                raise ControlChanged('短听按下前画面变化。')
            if reset:
                from short_listen import reset_control
                # Pointer covers reset glyph here: its location was matched in
                # the clear frame; the rest of the unchanged page is checked.
                if reset_control(image, target) != box:
                    raise ControlChanged('重置图标对应变化。')
        self.mouse.click_box(box, image.shape, before_press=before, quick=True)
        clicked = time.monotonic()
        self.mouse.move((target.button[0]+target.button[2]*3,
                         target.button[1]-target.button[3]), image.shape, quick=True)
        return clicked

    def quiet_players(self):
        """Pause observed autoplay successors too; never blindly toggle idle."""
        deadline, idle_samples = time.monotonic()+8, 0
        states = {}
        while time.monotonic() < deadline:
            image = self.snapshot(); self.check_page(image)
            states = {n:glyph_state(image,t.button) for n,t in self.targets.items()}
            self.log('short_pause_check', states=states)
            active = [n for n,state in states.items() if state == 'active']
            unknown = any(state == 'unknown' for state in states.values())
            if unknown:
                self.status('短听暂停确认：先移开鼠标，重新查看播放图标。')
                self.mouse.move((image.shape[1]*self.rng.uniform(.65,.8), image.shape[0]*.3), image.shape, quick=True)
                self.mouse.wait(.12)
                idle_samples = 0
                continue
            if not active and not unknown:
                idle_samples += 1
                if idle_samples >= 3: return
            else:
                idle_samples = 0
                for name in active:
                    try: self.preview_control(name, self.targets[name].button, 'active')
                    except ControlChanged as exc:
                        self.log('short_pause_recheck', target=name, reason=str(exc))
                        # An aborted pre-press check leaves the remote pointer
                        # on the button. Clear it before inspecting again.
                        self.mouse.move((image.shape[1]*.72,image.shape[0]*.3),image.shape,quick=True)
            self.mouse.wait(.12)
        self.log('short_pause_unconfirmed', states=states)
        raise RuntimeError('移开鼠标复查后仍未确认全部暂停，未切换下一段；详情见 short_pause_check 日志。')

    def opening_previews(self):
        previous=getattr(self,'supplemental',False)
        self.supplemental=True
        try:
            if not self.stop_replays(): self._opening_previews()
        finally:
            self.supplemental=previous

    def _opening_previews(self):
        from short_listen import choose_duration, reset_control
        self.log('short_listen_start')
        self.quiet_players()
        first = list(self.settings.player_order)
        previous = None
        for index in range(self.settings.short_count):
            if self.stop_replays(): break
            name = first[index] if index < len(first) else self.rng.choice([n for n in first if n != previous])
            previous = name
            known = self.known_readings.get(name)
            duration = choose_duration(known.total, self.settings.short_min, self.settings.short_max, self.rng) if known and known.total else None
            if duration is None:
                self.log('short_listen_skipped', target=name, reason='音频过短或时长未知，无法满足短听下限并留出暂停余量')
                continue
            self.status(f'开头短听 {index+1}/{self.settings.short_count}：{name}，目标约 {duration:.2f} 秒。')
            image = self.snapshot(); self.check_page(image)
            box = reset_control(image, self.targets[name])
            if box is None:
                self.log('short_listen_skipped', target=name, reason='未确认圆圈重置图标，不猜坐标')
                continue
            try:
                self.preview_control(name, box, 'idle', reset=True)
                deadline, zero = time.monotonic()+3, False
                while time.monotonic() < deadline:
                    image = self.snapshot(); self.check_page(image)
                    reading = read_players(self.detector, image, {name:self.targets[name]})[name]
                    if reading.state == 'active':
                        self.quiet_players()
                        break
                    if reading.state == 'idle' and reading.current == 0:
                        zero = True; break
                    self.mouse.wait(.08)
                if not zero:
                    self.log('short_listen_skipped', target=name, reason='重置后未确认停止且归零')
                    continue
                clicked = self.preview_control(name, self.targets[name].button, 'idle')
                began = None
                deadline = clicked+3
                while time.monotonic() < deadline:
                    image = self.snapshot(); self.check_page(image)
                    if glyph_state(image, self.targets[name].button) == 'active':
                        began = time.monotonic(); break
                    self.mouse.wait(.04)
                if began is not None:
                    # Count the listening budget from observed start, so a
                    # delayed remote start does not consume the whole preview.
                    self.mouse.wait(max(0, began+duration-time.monotonic()-.15))
                self.quiet_players()
                ended = time.monotonic()
                self.log('short_listen_completed', target=name, requested_seconds=duration,
                         start_confirm_delay=None if began is None else began-clicked,
                         click_to_all_idle_seconds=ended-clicked,
                         note='画面确认延迟，非真实音频延迟或精确音节长度')
            except ReplayFinished:
                self.quiet_players()
                break
            except ControlChanged as exc:
                self.log('short_listen_recheck', target=name, reason=str(exc))
                self.quiet_players()
            self.mouse.wait(self.rng.uniform(.2, .65))
        self.quiet_players()
        self.log('short_listen_end')

    def stop_replays(self):
        return False

    def replay_count(self):
        return self.rng.randint(self.settings.replay_min,self.settings.replay_max)

    def replay_rounds(self):
        count = self.replay_count()
        self.log('replay_plan', count=count)
        self.supplemental = True
        try:
            for i in range(count):
                if self.stop_replays(): break
                name = self.settings.player_order[0] if i == 0 else self.rng.choice(('A', 'B'))
                pause = self.settings.after_b_pause if i == 0 else self.rng.uniform(self.settings.pause_min, self.settings.pause_max)
                self.log('replay_wait', target=name, seconds=pause)
                deadline = time.monotonic() + pause
                while time.monotonic() < deadline:
                    if self.stop_replays(): raise ReplayFinished()
                    self.mouse.wait(min(.1, max(0, deadline-time.monotonic())))
                if self.stop_replays(): break
                self.status(f'额外重听 {i+1}/{count}：{name}')
                self.log('replay_start', index=i+1, count=count, target=name)
                self.click(name, fast=True)
                # Finish any already-started autoplay; do not click pause on
                # model completion or interrupt the recorded first round.
                self.settle_and_follow(name)
                self.log('replay_completed', index=i+1, target=name)
        except ReplayFinished:
            pass
        finally:
            self.supplemental = False

    def on_first_round_complete(self):
        pass

    def on_players_ready(self):
        """Prepare integrations only after the page and timers are verified."""
        pass

    def locate_first_round(self):
        retries=2 if getattr(self,'automatic_recovery',False) else 0
        for attempt in range(retries+1):
            try:
                image,readings=self.identify()
                return self.confirm_initial_timers(image,readings)
            except LocationUnconfirmed as exc:
                self.log('location_retry',attempt=attempt+1,reason=str(exc),will_retry=attempt<retries)
                if attempt==retries:raise
                self.status(f'播放器尚未确认，重新读取画面并定位（{attempt+1}/{retries}），尚未点击播放。')
                self.mouse.wait(.4)
                self.mode_detected=False

    def confirm_initial_timers(self,image,readings):
        active=any(r.state=='active' for r in readings.values())
        unreadable=[n for n,r in readings.items() if r.total is None and self.targets[n].state!='locked']
        for attempt in range(3):
            if not unreadable or active:break
            self.status('首次计时未读全，正在重新采样：'+'/'.join(unreadable))
            self.mouse.wait(.4)
            image=self.snapshot();self.check_page(image)
            readings=read_players(self.detector,image,self.targets)
            self.known_readings.update({n:r for n,r in readings.items() if r.total is not None})
            active=any(r.state=='active' for r in readings.values())
            unreadable=[n for n in readings if self.known_readings.get(n,Reading('unknown')).total is None and self.targets[n].state!='locked']
        if unreadable and not active:
            raise LocationUnconfirmed('无法可靠读取 '+'/'.join(unreadable)+' 的计时，未点击；已尝试重新读取和定位。请放大向日葵中的计时文字。')
        return image,readings

    def capture_recovery_identity(self,image):
        if not getattr(self,'automatic_recovery',False):return
        from audio_lab.task_context import waveform_profiles
        from rating_reader import question_marker
        self.recovery_identity={
            'waveforms':waveform_profiles(image,self.targets,self.settings.player_order),
            'marker':question_marker(getattr(self.detector,'last_task_lines',[]))}

    def verify_recovery_identity(self):
        # Do not re-identify over the old reference: a new task must never
        # inherit its progress or be clicked as part of this recovery.
        import numpy as np
        from audio_lab.task_context import waveform_profiles
        from rating_reader import question_marker
        image=self.snapshot();self.check_page(image)
        proof=getattr(self,'recovery_identity',None)
        if not proof:raise RuntimeError('缺少原题身份依据，未自动重播。')
        targets,_,_=self.detector.detect(image)
        from dataclasses import replace
        if any(name not in targets or not stable_target(replace(old,state=targets[name].state),targets[name])
               for name,old in self.targets.items()):
            raise RuntimeError('恢复前播放器位置尚未确认，未自动重播。')
        marker=question_marker(getattr(self.detector,'last_task_lines',[]))
        if proof['marker'] and marker!=proof['marker']:
            raise RuntimeError('恢复前题目编号变化或未读清，未自动重播。')
        profiles=waveform_profiles(image,targets,self.settings.player_order)
        for name,before in proof['waveforms'].items():
            correlation=float(np.corrcoef(before,profiles[name])[0,1]) if np.std(before)>0 and np.std(profiles[name])>0 else float('nan')
            if not np.isfinite(correlation) or correlation<.85:
                raise RuntimeError(f'恢复前 {name} 音频波形未通过同题核对，未自动重播。')
        self.log('recovery_identity_verified',marker=marker)

    def on_first_round_retry(self,attempt,reason):
        pass

    def validate_first_round(self):
        pass

    def confirmed_first_round(self,readings):
        retries=1 if getattr(self,'automatic_recovery',False) else 0
        for attempt in range(retries+1):
            try:
                self.prepare_first_round(readings)
                self.play_first_round(readings)
                self.validate_first_round()
                return
            except PlaybackEvidenceError as exc:
                self.log('first_round_retry',attempt=attempt+1,reason=str(exc),will_retry=attempt<retries)
                if attempt==retries:
                    self.status('首轮自动恢复次数已用完：'+str(exc))
                    raise
                self.status('首轮证据不完整，正在核对原题；确认后复位并从头完整重播一次。')
                self.verify_recovery_identity()
                self.quiet_players()
                self.verify_recovery_identity()
                self.on_first_round_retry(attempt+1,str(exc))
                _,readings=self.observe_players()
                self.prepare_first_round(readings,force=True)

    def start_and_wait(self,name,expected):
        retries=2 if getattr(self,'automatic_recovery',False) else 0
        for attempt in range(retries+1):
            self.click(name,fast=True)
            self.pending_play_start=name
            try:
                return self.await_chain(expected)
            except PlaybackDidNotStart as exc:
                if attempt==retries:raise
                # Always observe after waiting for manual input to settle.
                # A delayed start is followed; it must never be toggled off.
                _,fresh=self.observe_players()
                if any(r.state=='active' for r in fresh.values()):
                    self.pending_play_start=None
                    return self.await_chain(expected,fresh)
                if any(r.state not in ('idle','locked') for r in fresh.values()) or fresh[name].state!='idle' or fresh[name].current not in (0,fresh[name].total) or fresh[name].total is None:
                    raise PlaybackEvidenceError('补点前未确认停止状态和起点，改为核对后完整重播。') from exc
                self.log('play_start_retry',target=name,attempt=attempt+1,reason=str(exc))
                self.status(f'{name} 未开始播放，状态已重新确认，补点播放（{attempt+1}/{retries}）。')
            finally:self.pending_play_start=None

    def prepare_first_round(self,readings,force=False):
        # A prior pause can leave any player halfway through. Playback resumes
        # that position; reset it before recording a fresh full round.
        if any(r.state=='active' for r in readings.values()):return
        from short_listen import reset_control
        for name in self.settings.player_order:
            reading=readings[name]
            # An unreadable idle timer is not proof of zero. Resolve it before
            # playing, rather than spending a full round on unusable evidence.
            for attempt in range(2):
                if reading.state!='idle' or reading.current is not None:break
                self.parked(self.snapshot(),self.targets[name])
                _,fresh=self.observe_players()
                readings.update(fresh)
                reading=readings[name]
            if reading.state=='idle' and (reading.current is None or reading.total is None):
                raise RuntimeError(f'{name} 开始前未读清起点和总时长，未开始首轮；请保持播放器可见。')
            if reading.state!='idle' or reading.total is None or not (0<(reading.current or 0)<reading.total or force and reading.current!=0):continue
            image=self.snapshot();self.check_page(image)
            box=reset_control(image,self.targets[name])
            if box is None:raise RuntimeError(f'{name} 停在中途但未看清重置图标，未开始首轮。')
            self.preview_control(name,box,'idle',reset=True)
            self.parked(image,self.targets[name])
            deadline=time.monotonic()+3
            confirmed=0
            while time.monotonic()<deadline:
                _,fresh=self.observe_players()
                value=fresh[name]
                confirmed=confirmed+1 if value.state=='idle' and value.current==0 else 0
                if confirmed>=2:
                    readings.update(fresh)
                    self.log('first_round_reset',target=name,previous_current=reading.current)
                    break
                self.mouse.wait(.1)
            else:raise RuntimeError(f'{name} 未确认回到开头，未开始首轮。')

    def play_first_round(self, readings):
        order=self.settings.player_order
        active=[name for name,reading in readings.items() if reading.state=='active']
        if active:
            # Identification and recorder setup take time. Their original
            # active player may already have ended or advanced to a successor.
            # This partial playback is not first-round evidence: pause the
            # currently observed controls, reset partial positions, then start
            # a new complete round after recording is ready. Never relax the
            # strict tracker for the fresh round or relabel old observations.
            self.status('启动时页面已在播放，先暂停并复位，再从头完整播放本题。')
            self.log('startup_playback_reset',readings={n:asdict(r) for n,r in readings.items()})
            self.quiet_players()
            _,readings=self.observe_players()
            if any(r.state=='active' for r in readings.values()):
                raise RuntimeError('启动时未确认原有播放已暂停，未开始首轮。')
            self.prepare_first_round(readings)
        if order==('A','B'):
            # Do not assume this page autoplays B. Await A alone, then inspect
            # whether B has started or needs a verified click after unlocking.
            self.start_and_wait('A',('A',))
            _, fresh=self.observe_players()
            if fresh['B'].state=='active':
                self.await_chain(('B',),fresh)
            elif fresh['B'].current==fresh['B'].total and fresh['B'].total is not None:
                # A very short B may have already finished between observations.
                # Restart it rather than assuming it was captured completely.
                self.start_and_wait('B',('B',))
            else:
                self.start_and_wait('B',('B',))
            return
        self.start_and_wait(order[0],order)

    def run(self):
        try:
            self.status('自动流程开始：录音和分析持续运行，鼠标点击等待焦点。' if getattr(self,'observation_only_playback',False)
                        else '连续流程开始：移动鼠标／切换窗口／Esc 会暂停。')
            image = self.snapshot()
            self.parked(image)
            image, readings = self.locate_first_round()
            self.capture_recovery_identity(image)
            self.on_players_ready()
            # Recorder initialization may span an autoplay transition. Read
            # the actual current frame instead of using identification-time
            # readings to choose where the first recorded round should start.
            _,readings=self.observe_players()
            self.log('recording_start_readings',readings={n:asdict(r) for n,r in readings.items()})
            self.confirmed_first_round(readings)
            self.on_first_round_complete()
            if self.locked_start:
                self.status('首轮播放完成，重新确认播放器已经解锁。')
                self.identify()
                if self.locked_start: raise RuntimeError('首轮完成后仍显示锁定，未重播。')
            self.replay_rounds()
            if self.settings.short_enabled:
                self.opening_previews()
            if self.settings.scroll_after:
                self.log('rating_scroll_start')
                self.inspect_ratings()
            else:
                self.status('本轮完成：重听已结束，按设置保留音频区，不下滚。')
            self.log('session_end')
        except Exception as exc:
            self.log('session_stopped', reason=str(exc), last_readings=self.last_readings)
            if self.last_image is not None:
                cv2.imwrite(str(self.directory/'stopped.png'), self.last_image)
            raise
