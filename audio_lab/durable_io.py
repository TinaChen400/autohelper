"""Flush checkpoints before publishing metadata or reporting saved audio."""
import json
import os
from pathlib import Path
import tempfile


def checkpoint(stream):
    stream.flush()
    os.fsync(stream.fileno())


def write_json(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            checkpoint(output)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
