"""Shared paths and small helpers for the child jamo fine-tuning run.

Manifests, catalogs, logs and checkpoints live under WORK and are excluded
from Git. The recorded workspace move preserves historical path identities
through explicit path translation instead of rewriting frozen training records.
"""
from pathlib import Path
import hashlib
import json
import os
import time

from workspace_paths import DATASETS, PROJECT, physical_path

ROOT = PROJECT
WORK = ROOT / '.venv/hubert-child-20260914'
RUN = WORK / 'run'
INDEX = DATASETS / 'AIHUB정리/index.csv'
SPLITS = DATASETS / 'AIHUB정리/splits.json'
RAW = DATASETS / 'AIHUB'
OLD_DEV = ROOT / 'python/tools/child_finetune/data/dev.csv'

#: The verified pretrained encoder. `CHILD_JAMO_ENCODER` points this at a
#: tiny stand-in so the state-machine tests can drive the real training
#: entry point - same phases, cursor, checkpointing and refusal paths - in
#: seconds instead of writing multi-gigabyte checkpoints. It is never set
#: for a real run; every report records which encoder was actually used.
ENCODER = physical_path(os.environ.get(
    'CHILD_JAMO_ENCODER',
    ROOT / '.venv/encoder-screen-0914/models/hubertko1b'))

for d in (WORK, RUN):
    d.mkdir(parents=True, exist_ok=True)


def digest(path):
    with physical_path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    """Write JSON atomically; a crash mid-write never leaves a torn file."""
    path = physical_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    # Windows readers may briefly hold a handle without FILE_SHARE_DELETE.
    # Keep atomic replacement, but do not abort hours of training merely
    # because a status reader overlapped this small rename. Persistent
    # permission errors still surface; the previous JSON stays intact.
    for attempt in range(40):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def load(path):
    return json.loads(physical_path(path).read_text(encoding='utf-8'))


def log(name, message):
    """Append one UTF-8 line to a run log and echo it."""
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {message}'
    path = RUN / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(message, flush=True)
