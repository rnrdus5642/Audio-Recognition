"""Pull the old split definition out of the frame caches.

The detection figures on record - 48.7% child, 28.2% at age five - belong
to particular children, and the only surviving record of which children
those were is the cached frames sitting in a session scratchpad. Written
out here so the numbers stay reproducible after the audio is re-fetched
and re-sorted.

Speaker ids are hashed. The corpus labels children by name or initials -
CHOIYEJUN, KDH - and this file is committed, so the raw ids must not be.
Hashing the ids of freshly indexed audio the same way reproduces the
split without ever storing who the children are.

Run once, pointing at wherever the caches live:

    python python/tools/aihub/rescue_splits.py <scratchpad-dir>
"""

import hashlib
import json
import os
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "splits_v1.json")
FILES = ("tune_frames.json", "valid_frames.json", "more_frames.json")


def speaker_hash(name):
    return hashlib.sha256(name.strip().upper().encode()).hexdigest()[:12]


src = sys.argv[1] if len(sys.argv) > 1 else "."
splits = defaultdict(lambda: {"speakers": {}, "utterances": 0})

for f in FILES:
    path = os.path.join(src, f)
    if not os.path.exists(path):
        print(f"없음: {f}", flush=True)
        continue
    data = json.load(open(path, encoding="utf-8"))
    for name, items in data["sets"].items():
        split = name.rsplit("_", 1)[0]
        d = splits[split]
        for it in items:
            key = speaker_hash(it["speaker"])
            d["speakers"].setdefault(key, {"age": int(it["age"]),
                                           "n": 0})["n"] += 1
            d["utterances"] += 1
    print(f"{f}: {', '.join(sorted(data['sets']))}", flush=True)

out = {
    "note": "2026-08-11 아동 튜닝에 쓴 화자 분리. 기록용 - 새 작업은 "
            "splits.py 가 만드는 분할을 쓴다.",
    "speaker_id": "sha256(대문자 화자ID)[:12]",
    "window_s": 2.5,
    "hop_s": 0.5,
    "splits": {},
}
for name in sorted(splits):
    d = splits[name]
    ages = defaultdict(int)
    for v in d["speakers"].values():
        ages[v["age"]] += 1
    out["splits"][name] = {
        "utterances": d["utterances"],
        "speakers": dict(sorted(d["speakers"].items())),
        "speakers_by_age": dict(sorted(ages.items())),
    }
    print(f"  {name:6} 화자 {len(d['speakers']):4}명  발화 {d['utterances']:5,}"
          f"  나이 {dict(sorted(ages.items()))}", flush=True)

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n{OUT}", flush=True)
