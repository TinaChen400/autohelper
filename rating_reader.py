"""Read rating dimensions and options from pixels; never choose or submit."""
import re
import cv2
import numpy as np


def option_name(text):
    normalized = re.sub(r'[^a-z]', '', text.lower())
    return {'responsea': 'Response A', 'responseb': 'Response B', 'bothgood': 'Both Good',
            'bothbad': 'Both Bad', 'na': 'N/A'}.get(normalized)


def dimensions_from_lines(lines, shape):
    items = []
    for quad, text, confidence in lines:
        if confidence < .75: continue
        q = np.asarray(quad)
        x, y = q.min(axis=0); right, bottom = q.max(axis=0)
        items.append({'text': text.strip(), 'box': (int(x), int(y), int(right-x), int(bottom-y)), 'option': option_name(text)})
    options = sorted([row for row in items if row['option']], key=lambda row: row['box'][1])
    groups = []
    for row in options:
        cy = row['box'][1]+row['box'][3]/2
        for group in groups:
            if abs(cy-group[0]['box'][1]-group[0]['box'][3]/2) <= max(6, row['box'][3]):
                group.append(row); break
        else: groups.append([row])
    dimensions = []
    for group in groups:
        names = {row['option'] for row in group}
        if len(names) < 2 or not names.intersection({'Both Good', 'Both Bad'}): continue
        left = min(row['box'][0] for row in group)
        cy = sum(row['box'][1]+row['box'][3]/2 for row in group)/len(group)
        labels = [row for row in items if not row['option'] and row['box'][0]+row['box'][2] < left
                  and abs(row['box'][1]+row['box'][3]/2-cy) < max(8, row['box'][3]*1.3)
                  and 'rate the' not in row['text'].lower()]
        if not labels: continue
        label = min(labels, key=lambda row: abs(row['box'][1]+row['box'][3]/2-cy))
        descriptions = [row['text'] for row in items if not row['option']
                        and row['box'][0] < left and row['box'][1] > label['box'][1]+label['box'][3]
                        and abs(row['box'][0]-label['box'][0]) < max(20, label['box'][3]*4)
                        and row['box'][1] < label['box'][1]+label['box'][3]*3]
        dimensions.append({'name': label['text'].rstrip(' *'), 'description': ' '.join(descriptions),
                           'label_box': label['box'], 'options': {row['option']: row['box'] for row in group}})
    return dimensions


def read_rating_page(detector, image):
    scale = min(1.5, 1600/image.shape[1])
    lines, _ = detector.ocr(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))
    lines = [(np.asarray(q)/scale, text, score) for q, text, score in (lines or [])]
    dimensions = dimensions_from_lines(lines, image.shape)
    submit_visible = any(re.sub(r'[^a-z]', '', text.lower()) in ('submit', 'submitrating', 'submitratings')
                         and score >= .85 for _, text, score in lines)
    return dimensions, submit_visible
