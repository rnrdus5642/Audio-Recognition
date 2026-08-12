"""Sort the child audio into age folders.

Ages 5-7 only: that is the range the VR users fall in, and the corpus is
dominated by 9-12 year olds. Each tar is streamed once - random access
into a 46 GB archive would be far slower than reading it straight
through.
"""

import csv
import os
import tarfile
import time

BASE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE, "aihub_index.csv")
ROOT = r"C:\Users\user\Desktop\AIHUB아동오디오데이터"
DEST = os.path.join(ROOT, "나이별")

VALIDATION = os.path.join(
    ROOT, "011.한국어 아동 음성 데이터", "01.데이터", "2.Validation")
TARS = {
    "free": os.path.join(
        VALIDATION, "VS_kor_free_01", "011.한국어_아동_음성_데이터",
        "01.데이터", "2.Validation", "원천데이터", "VS_kor_free_01.tar"),
    "formatted": os.path.join(
        VALIDATION, "VS_kor_formatted_01", "011.한국어_아동_음성_데이터",
        "01.데이터", "2.Validation", "원천데이터", "VS_kor_formatted_01.tar"),
}
AGES = {"5", "6", "7"}

rows = list(csv.DictReader(open(INDEX, encoding="utf-8-sig")))
wanted = {}
per_folder = {}
for r in rows:
    if r["age"] not in AGES:
        continue
    folder = os.path.join(DEST, f"{int(r['age']):02d}세", r["set"])
    wanted[r["wav"]] = folder
    per_folder.setdefault(folder, []).append(r)

print(f"대상 {len(wanted):,}개", flush=True)
for folder, items in sorted(per_folder.items()):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "_목록.csv"), "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(items)
    secs = sum(float(i["seconds"] or 0) for i in items)
    print(f"  {os.path.relpath(folder, DEST):22} {len(items):6,}개 "
          f"{secs/3600:5.1f}시간 화자 {len({i['speaker'] for i in items})}명",
          flush=True)

t0 = time.time()
done = 0
for name, path in TARS.items():
    if not os.path.exists(path):
        print(f"{name}: tar 없음 - 건너뜀", flush=True)
        continue
    print(f"\n{name} 스트리밍…", flush=True)
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            base = os.path.basename(member.name)
            folder = wanted.get(base)
            if folder is None:
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(os.path.join(folder, base), "wb") as out:
                out.write(src.read())
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(wanted):,}  {time.time()-t0:.0f}s",
                      flush=True)

print(f"\n꺼낸 파일 {done:,}/{len(wanted):,}  {time.time()-t0:.0f}초")
print(DEST)
