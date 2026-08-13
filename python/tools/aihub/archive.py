"""Read AI Hub's archives without unpacking them first.

The corpus arrives as tars split into 1 GB parts. Joining them back
together needs as much free space again as the download itself, and the
audio runs to hundreds of gigabytes, so instead the parts are chained
into one stream and handed to tarfile directly. Nothing is written to
disk that we did not ask for.

Reading labels straight out of the stream also sidesteps the slowest
step of the old approach: 274k tiny JSON files took 22 minutes to open
one by one on Windows, and a sequential read of the same archive takes a
fraction of that.
"""

import io
import os
import re
import tarfile
from collections import defaultdict

PART = re.compile(r"^(?P<base>.+\.tar)\.part(?P<n>\d+)$", re.IGNORECASE)


class ChainedParts(io.RawIOBase):
    """The parts of one split tar, read end to end as a single file."""

    def __init__(self, paths):
        self.paths = list(paths)
        self.i = 0
        self.fh = open(self.paths[0], "rb")

    def readable(self):
        return True

    def readinto(self, buf):
        while True:
            n = self.fh.readinto(buf)
            if n:
                return n
            self.fh.close()
            self.i += 1
            if self.i >= len(self.paths):
                return 0
            self.fh = open(self.paths[self.i], "rb")

    def close(self):
        if not self.fh.closed:
            self.fh.close()
        super().close()


class Archive:
    """One logical tar - either a plain file or a set of parts."""

    def __init__(self, name, paths, kind, style, subset):
        self.name = name        # VS_kor_free_01.tar
        self.paths = paths      # the file, or its parts in order
        self.kind = kind        # "audio" | "label"
        self.style = style      # "free" | "formatted"
        self.subset = subset    # "train" | "valid"

    @property
    def bytes(self):
        return sum(os.path.getsize(p) for p in self.paths)

    def open(self):
        """A streaming tarfile. Sequential access only - do not seek."""
        if len(self.paths) == 1:
            return tarfile.open(self.paths[0], "r|")
        stream = io.BufferedReader(ChainedParts(self.paths), 1 << 22)
        return tarfile.open(fileobj=stream, mode="r|")

    def __repr__(self):
        return (f"<{self.name} {self.kind}/{self.style}/{self.subset} "
                f"{self.bytes / 2**30:.1f}GB {len(self.paths)}개>")


def _classify(name, path):
    low = (name + " " + path).lower()
    if re.search(r"[tv]l_", low) or "라벨링데이터" in path:
        kind = "label"
    elif re.search(r"[tv]s_", low) or "원천데이터" in path:
        kind = "audio"
    else:
        kind = "unknown"

    style = "formatted" if "formatted" in low else (
        "free" if "free" in low else "unknown")

    if "training" in low or re.search(r"\bt[sl]_", low):
        subset = "train"
    elif "validation" in low or re.search(r"\bv[sl]_", low):
        subset = "valid"
    else:
        subset = "unknown"
    return kind, style, subset


def discover(root):
    """Every archive under root, parts already grouped and ordered."""
    groups = defaultdict(list)
    plain = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            full = os.path.join(dirpath, f)
            m = PART.match(f)
            if m:
                groups[(dirpath, m.group("base"))].append(
                    (int(m.group("n")), full))
            elif f.lower().endswith(".tar"):
                plain[(dirpath, f)] = full

    out = []
    for (dirpath, base), parts in groups.items():
        # a joined tar sitting next to its parts wins; the parts are leftovers
        if (dirpath, base) in plain:
            continue
        paths = [p for _n, p in sorted(parts)]
        out.append(Archive(base, paths, *_classify(base, dirpath)))
    for (dirpath, name), full in plain.items():
        out.append(Archive(name, [full], *_classify(name, dirpath)))

    out.sort(key=lambda a: (a.kind, a.subset, a.style, a.name))
    return out


def members(archive, suffix):
    """Yield (member, extracted bytes) for members ending in suffix."""
    with archive.open() as tar:
        for m in tar:
            if not m.isfile() or not m.name.lower().endswith(suffix):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            yield m, f.read()


if __name__ == "__main__":
    import sys

    from config import RAW

    root = sys.argv[1] if len(sys.argv) > 1 else RAW
    found = discover(root)
    if not found:
        print(f"아카이브 없음: {root}")
        raise SystemExit(1)
    total = 0
    for a in found:
        print(f"  {a}")
        total += a.bytes
    print(f"\n{len(found)}개 · {total / 2**30:.1f}GB")
