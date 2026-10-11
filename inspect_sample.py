"""Offline recognition, never controls the desktop."""
import json
import sys
from pathlib import Path
import cv2
from dataclasses import asdict
from vision import Detector, annotate

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    directory = Path(__file__).resolve().parent
    detector = Detector()
    results = []
    for path in sorted((directory/'artifacts').glob('frame_*.png')):
        image = cv2.imread(str(path))
        targets, issues, seconds = detector.detect(image)
        cv2.imwrite(str(path.with_name('detected_'+path.name)), annotate(image, targets))
        row = {'file': path.name, 'seconds': round(seconds, 3), 'targets': {k: asdict(v) for k, v in targets.items()}, 'issues': issues}
        print(json.dumps(row, ensure_ascii=False))
        results.append(row)
    (directory/'artifacts'/'recognition_report.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
