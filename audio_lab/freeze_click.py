"""Archive the click application without modifying it or including recordings."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
files = [p for p in root.iterdir() if p.is_file() and (p.suffix in ('.py','.bat','.txt','.md') or p.name in ('.gitignore','reset_icon.png'))]
files += [p for p in (root/'tests').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
folder = root/'backups'
folder.mkdir(exist_ok=True)
archive = folder/f'click_v0.14_{datetime.now():%Y%m%d_%H%M%S}.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as out:
    for p in files: out.write(p,p.relative_to(root).as_posix())
    out.writestr('SNAPSHOT_SHA256.json',json.dumps(manifest,indent=2,ensure_ascii=False))
with zipfile.ZipFile(archive) as saved:
    assert saved.testzip() is None
    for name,digest in manifest.items(): assert hashlib.sha256(saved.read(name)).hexdigest() == digest
print(archive)

