"""Pull the wanted ages out of the audio archives into age folders.

Each archive is streamed straight through exactly once. Seeking into a
hundred-gigabyte tar to fetch scattered files would be far slower than
reading it start to finish and keeping what goes past.

Safe to re-run: files already on disk are skipped, so an interrupted run
picks up where it stopped.

    python python/tools/aihub/organize.py
"""

import csv
import os
import sys
import time

from archive import discover
from config import AGES, AUDIO, INDEX, RAW


def wanted_files():
    if not os.path.exists(INDEX):
        print(f"색인이 없습니다: {INDEX}\n먼저 index.py 를 돌리세요.")
        return None
    wanted = {}
    with open(INDEX, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                age = int(r["age"])
            except ValueError:
                continue
            if age in AGES and r["wav"]:
                wanted[r["wav"]] = os.path.join(
                    AUDIO, f"{age:02d}세", r["style"] or "unknown")
    return wanted


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else RAW
    wanted = wanted_files()
    if wanted is None:
        return 1
    print(f"대상 {len(wanted):,}개 (나이 {sorted(AGES)})")

    for folder in set(wanted.values()):
        os.makedirs(folder, exist_ok=True)

    archives = [a for a in discover(root) if a.kind == "audio"]
    if not archives:
        print(f"오디오 아카이브를 못 찾았습니다: {root}")
        return 1

    t0 = time.time()
    written = skipped = 0
    for a in archives:
        print(f"\n{a}", flush=True)
        seen = 0
        with a.open() as tar:
            for m in tar:
                if not m.isfile():
                    continue
                base = os.path.basename(m.name)
                folder = wanted.get(base)
                if folder is None:
                    continue
                seen += 1
                dest = os.path.join(folder, base)
                if os.path.exists(dest):
                    skipped += 1
                    continue
                src = tar.extractfile(m)
                if src is None:
                    continue
                tmp = dest + ".part"
                with open(tmp, "wb") as out:
                    out.write(src.read())
                os.replace(tmp, dest)
                written += 1
                if (written + skipped) % 2000 == 0:
                    print(f"  {written + skipped:,}/{len(wanted):,}"
                          f"  {time.time()-t0:.0f}s", flush=True)
        print(f"  이 아카이브에서 {seen:,}개 해당", flush=True)

    missing = sum(1 for f, d in wanted.items()
                  if not os.path.exists(os.path.join(d, f)))
    print(f"\n새로 꺼냄 {written:,} · 이미 있음 {skipped:,} · "
          f"못 찾음 {missing:,}  {time.time()-t0:.0f}초")
    if missing:
        print("못 찾은 파일이 있으면 해당 원천데이터 아카이브가 빠진 것입니다.")
    print(AUDIO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
