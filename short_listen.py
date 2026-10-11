"""Screen-only reset control matching and duration-aware opening previews."""
from pathlib import Path
import cv2


def choose_duration(total, minimum, maximum, rng):
    # Leave headroom for remote pause latency. If an entire clip is shorter
    # than the floor, a partial preview cannot be promised; skip it explicitly.
    upper = min(maximum, total*.5, total-.35)
    if upper < minimum: return None
    upper = max(minimum, upper)
    lower = max(minimum, min(upper*.6, upper))
    return rng.uniform(lower, upper)


def reset_control(image, target):
    template = cv2.imread(str(Path(__file__).with_name('reset_icon.png')), cv2.IMREAD_GRAYSCALE)
    if template is None: return None
    x, y, w, h = target.button
    left, right = round(x+w*1.12), round(x+w*1.98)
    top, bottom = max(0, y-2), min(image.shape[0], y+h+2)
    crop = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    best = None
    for factor in (.58, .68, .76):
        size = max(8, round(w*factor))
        ref = cv2.resize(template, (size, size))
        if crop.shape[0] < size or crop.shape[1] < size: continue
        scores = cv2.matchTemplate(crop, ref, cv2.TM_CCOEFF_NORMED)
        _, score, _, point = cv2.minMaxLoc(scores)
        if best is None or score > best[0]:
            best = (score, (left+point[0], top+point[1], size, size))
    return best[1] if best and best[0] >= .65 else None
