"""Timer OCR and playback state; no mouse or desktop side effects."""
from dataclasses import dataclass
import re
import cv2
from vision import glyph_state


@dataclass
class Reading:
    state: str
    current: int | None = None
    total: int | None = None


def parse_timer(text):
    normalized = text.replace('O', '0').replace('o', '0').replace('：', ':')
    normalized = re.sub(r'\s+', '', normalized)
    match = re.fullmatch(r'(\d{1,2}):([0-5]\d)[/|](\d{1,2}):([0-5]\d)', normalized)
    if not match: return None
    a, b, c, d = map(int, match.groups())
    current, total = a*60+b, c*60+d
    if 0 < total <= 3600 and current <= total: return current, total
    return None


def read_players(detector, image, targets):
    readings = {}
    for name, target in targets.items():
        x, y, w, h = target.button
        left = max(0, round(x+w*1.85))
        top, bottom = max(0, y), min(image.shape[0], y+h)
        timer = None
        # Sharpened cubic scaling sometimes merges tiny digits and the slash.
        # Retry a different resampling of the same pixels before giving up.
        for extent, interpolation in ((5.65, cv2.INTER_CUBIC), (6.1, cv2.INTER_CUBIC),
                                      (5.65, cv2.INTER_LINEAR), (6.1, cv2.INTER_LINEAR)):
            right = min(image.shape[1], round(x+w*extent))
            crop = image[top:bottom, left:right]
            if not crop.size: continue
            enlarged = cv2.resize(crop, None, fx=3, fy=3, interpolation=interpolation)
            rows, _ = detector.ocr(enlarged, use_det=False, use_cls=False)
            for text, confidence in (rows or []):
                if confidence >= .75:
                    timer = parse_timer(text)
                    if timer: break
            if timer: break
        readings[name] = Reading(glyph_state(image, target.button), *(timer or (None, None)))
    return readings


class Segment:
    def __init__(self):
        self.started = False
        self.done = False
        self.previous = None
        self.last_progress = None
        self.idle_since = None
        self.ended_samples = 0
        self.total = None
        self.expected_end = None
        self.started_at = None
        self.backward_candidate = None

    def seed(self, reading):
        if reading.total is not None:
            self.total = reading.total

    def near_end(self, now):
        return self.expected_end is not None and now >= self.expected_end

    def observe(self, reading, now):
        if self.done: return
        previous = self.previous
        previous_progress_at = self.last_progress
        reset_candidate = bool(self.started and reading.state == 'idle' and reading.current == 0
                               and previous and previous.current is not None
                               and previous.current >= previous.total-1)
        if reading.state == 'active':
            # Idle timestamps belong to the previous play. Start the stall
            # clock when this play actually begins, even if its first timer
            # is briefly unreadable during the autoplay transition.
            if not self.started:
                self.last_progress = now
                self.started_at = now
                if self.total is not None:
                    self.expected_end = now+self.total
            self.started = True
            self.idle_since = None
            self.ended_samples = 0
        if reading.current is not None:
            if previous and previous.total is not None and reading.total != previous.total:
                raise RuntimeError('音频总时长变化，可能换题或时间误读；停止本轮。')
            # Idle controls commonly retain the last round's final timestamp.
            # Their first active sample legitimately starts at zero. Compare
            # progress only between consecutive active samples of this play.
            if (previous and previous.current is not None and previous.state == 'active' and reading.state == 'active'
                    and reading.current < previous.current):
                startup_reset = (self.started_at is not None and now-self.started_at <= 3
                                 and previous.current >= previous.total-1 and reading.current <= 2)
                if not startup_reset:
                    # One lower OCR value is not proof of a seek. Confirm the
                    # new timeline on a second progressing sample; a genuine
                    # manual seek is then followed rather than ending the run.
                    candidate = self.backward_candidate
                    if candidate is None or not candidate <= reading.current <= candidate+2:
                        self.backward_candidate = reading.current
                        return
                self.backward_candidate = None
                self.last_progress = now
                self.idle_since = None
            else:
                self.backward_candidate = None
            if not reset_candidate and (previous is None or reading.current != previous.current):
                self.last_progress = now
            self.total = reading.total
            if reading.state == 'active':
                self.expected_end = now+reading.total-reading.current
        ended = False
        # Missing timer text is not evidence of a pause. Combine the last
        # observed duration/progress with two clear stopped-icon observations.
        if self.started and reading.state == 'idle' and reading.current is None:
            ended = self.near_end(now)
        if self.started and reading.state == 'idle' and reading.current is not None:
            ended = reading.current == reading.total
            # Some players reset to zero after finishing. Require an observed
            # near-end sample and enough elapsed time to reach the end.
            if (not ended and reading.current == 0 and previous and previous.current is not None
                    and previous.current >= previous.total-1 and previous_progress_at is not None):
                ended = now-previous_progress_at >= previous.total-previous.current
            if not ended:
                if self.idle_since is None: self.idle_since = now
                if now-self.idle_since > 3:
                    raise RuntimeError('音频提前停止或暂停，未判定播完；不会点击下一条。')
        if ended:
            if reading.current == 0: self.last_progress = previous_progress_at
            self.ended_samples += 1
            if self.ended_samples >= 2: self.done = True
        elif reading.state != 'unknown': self.ended_samples = 0
        if reading.current is not None:
            # Preserve the last near-end sample during a reset confirmation.
            if not reset_candidate: self.previous = reading
        if (self.started and reading.state == 'active' and self.last_progress is not None
                and (reading.current is not None or self.total is None or now > self.expected_end+15)
                and now-self.last_progress > 15):
            raise RuntimeError('播放进度超过 15 秒没有前进，已停止等待。')


class RoundTracker:
    def __init__(self, expected=('Input', 'A', 'B'), known=None):
        self.expected = tuple(expected)
        self.segments = {name: Segment() for name in expected}
        for name, reading in (known or {}).items():
            if name in self.segments: self.segments[name].seed(reading)

    def observe(self, readings, now):
        active = [name for name, reading in readings.items() if reading.state == 'active']
        if len(active) > 1: raise RuntimeError('检测到多个播放器同时播放，停止本轮。')
        for name in self.expected:
            if name not in readings: continue
            if name in active:
                if self.segments[name].done: raise RuntimeError('已结束的片段再次开始，停止本轮。')
                prior = self.expected[:self.expected.index(name)]
                for other in prior:
                    segment = self.segments[other]
                    # A following autoplay item confirms an immediately preceding
                    # near-end transition, but never a segment not observed playing.
                    previous = segment.previous
                    if (segment.started and not segment.done
                            and ((previous and previous.current is not None and previous.current >= previous.total-1)
                                 # Observation starts up to one polling interval
                                 # after playback. A visible successor confirms
                                 # that small timing uncertainty at the boundary.
                                 or segment.near_end(now+.75))
                            and readings.get(other, Reading('unknown')).state == 'idle'):
                        segment.done = True
                    if not segment.done:
                        raise RuntimeError('自动播放跳过了尚未确认完成的片段，停止本轮。')
            self.segments[name].observe(readings[name], now)
        return all(segment.done for segment in self.segments.values())

    def summary(self, readings):
        parts = []
        for name in self.expected:
            segment = self.segments[name]
            reading = readings.get(name, Reading('unknown'))
            timer = f'{reading.current}/{reading.total}秒' if reading.current is not None else '时间未读清'
            if reading.current is None and segment.total is not None:
                timer = f'总长 {segment.total} 秒，按已读时长跟踪并等待画面确认'
            state = '已结束' if segment.done else '播放中' if reading.state == 'active' else '等待'
            parts.append(f'{name} {state} {timer}')
        return '；'.join(parts)


