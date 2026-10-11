"""Expose pip-installed Windows CUDA DLLs only inside this Python process."""
import os
from pathlib import Path
import sys


def cuda_paths():
    handles = []
    if os.name != 'nt': return handles
    for folder in (Path(sys.prefix)/'Lib'/'site-packages'/'nvidia').glob('*/bin'):
        os.environ['PATH'] = str(folder)+os.pathsep+os.environ.get('PATH','')
        handles.append(os.add_dll_directory(str(folder)))
    return handles
