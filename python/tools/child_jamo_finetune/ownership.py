"""One live writer per run directory, enforced by the OS.

Reading a PID file and then writing it is not exclusion: two launchers
can both read "no owner" and both write. On Windows `msvcrt.locking`
takes a real mandatory byte-range lock on an open handle, and the handle
is held for the life of the process, so the second process is refused by
the kernel rather than by a race. If the owner dies the handle closes and
the lock is released, so a crashed run does not leave the directory
permanently blocked.

The PID file next to it is descriptive only - it says who holds the lock,
for anyone looking - and is removed only by the process that wrote it.
"""
from pathlib import Path
import atexit
import json
import msvcrt
import os
import time


class Owned:
    def __init__(self, directory, root):
        self.directory = Path(directory).resolve()
        self.root = Path(root).resolve()
        # A run may only ever write inside the work root.
        if self.root not in self.directory.parents:
            raise SystemExit(f'run directory {self.directory} is outside '
                             f'{self.root}')
        self.lock = self.directory / 'run.lock'
        self.owner = self.directory / 'run.owner.json'
        self.handle = None

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.lock, 'a+b')
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            self.handle = None
            raise SystemExit(
                f'run "{self.directory.name}" is already being written by '
                f'another live process ({self._described()}); refusing to '
                f'start a second writer')
        self.owner.write_text(json.dumps(
            {'pid': os.getpid(), 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
             'directory': str(self.directory)}), encoding='utf-8')
        atexit.register(self.release)
        return self

    def _described(self):
        try:
            held = json.loads(self.owner.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return 'owner unknown'
        return f'pid {held.get("pid")} since {held.get("started")}'

    def release(self):
        if self.handle is None:
            return
        # Remove our own metadata while still holding the lock. Unlocking
        # first would open a window where the next writer has acquired the
        # lock and written its own owner file, and this process would then
        # read that file, see a foreign pid, and - worse, if the pids ever
        # matched - delete someone else's metadata.
        try:
            held = json.loads(self.owner.read_text(encoding='utf-8'))
            if held.get('pid') == os.getpid():
                self.owner.unlink()
        except (OSError, ValueError):
            pass
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        self.handle.close()
        self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_exc):
        self.release()
        return False
