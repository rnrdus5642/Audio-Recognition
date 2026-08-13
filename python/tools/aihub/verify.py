"""Check the tidied corpus before anything is trained on it.

Catches the mistakes that are expensive to find later: a child on both
sides of a split, audio the index promises but cannot be found, an eval
split too small for its numbers to mean anything.

    python python/tools/aihub/verify.py
"""

import collections
import csv
import json
import os
import random

from config import AGES, AUDIO, INDEX, SPLITS, TARGET_AGES

# Detection varies about 16 points from child to child, so the error on
# a split's mean is roughly 16/sqrt(n): 25 children gives +-6 points at
# 95%, which is comfortably inside the swing a fine-tune should produce.
# Fitting is the part that needs more - 15 children once pushed a
# threshold to 0.925 - and fitting happens on train, which is far larger.
MIN_EVAL_SPEAKERS = 25

problems = []
notes = []


def check(ok, message):
    (notes if ok else problems).append(message)


def main():
    if not os.path.exists(INDEX):
        print(f"색인이 없습니다: {INDEX}")
        return 1

    rows = []
    with open(INDEX, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                r["_age"] = int(r["age"])
            except ValueError:
                continue
            rows.append(r)
    print(f"색인 {len(rows):,}행")

    # one child, one age
    ages = collections.defaultdict(set)
    for r in rows:
        ages[r["speaker"]].add(r["_age"])
    mixed = [s for s, a in ages.items() if len(a) > 1]
    check(not mixed, f"화자당 나이 하나 (어긋남 {len(mixed)}명)")

    # duplicate filenames would silently overwrite during organize
    names = collections.Counter(r["wav"] for r in rows)
    dupes = [w for w, n in names.items() if n > 1]
    check(not dupes, f"파일명 중복 없음 (중복 {len(dupes)}개)")

    # transcripts present
    empty = sum(1 for r in rows if not r["text"].strip())
    check(not empty, f"전사 비어있음 {empty}건")

    # audio on disk
    want = [r for r in rows if r["_age"] in AGES]
    missing = []
    for r in want:
        p = os.path.join(AUDIO, f"{r['_age']:02d}세",
                         r["style"] or "unknown", r["wav"])
        if not os.path.exists(p):
            missing.append(p)
    check(not missing,
          f"오디오 {len(want) - len(missing):,}/{len(want):,}개 존재"
          + (f" · 없음 {len(missing):,}" if missing else ""))
    if missing:
        for p in missing[:3]:
            notes.append(f"    예: {os.path.basename(p)}")

    # splits
    if os.path.exists(SPLITS):
        sp = json.load(open(SPLITS, encoding="utf-8"))["speakers"]
        by_split = collections.defaultdict(set)
        for s, v in sp.items():
            by_split[v["split"]].add(s)
        overlap = []
        seen = {}
        for name, members in by_split.items():
            for s in members:
                if s in seen:
                    overlap.append(s)
                seen[s] = name
        check(not overlap, f"화자 중복 분할 없음 (중복 {len(overlap)}명)")

        for name in ("valid", "test"):
            for age in sorted(TARGET_AGES):
                n = sum(1 for s, v in sp.items()
                        if v["split"] == name and v["age"] == age)
                check(n >= MIN_EVAL_SPEAKERS,
                      f"{name} {age}세 화자 {n}명"
                      + ("" if n >= MIN_EVAL_SPEAKERS
                         else f" (권장 {MIN_EVAL_SPEAKERS}명 이상, "
                              "적으면 결과 불안정)"))
    else:
        problems.append(f"분할 파일 없음: {SPLITS}")

    # a few files should actually open
    if want and not missing:
        try:
            import soundfile as sf
            for r in random.Random(0).sample(want, min(20, len(want))):
                p = os.path.join(AUDIO, f"{r['_age']:02d}세",
                                 r["style"] or "unknown", r["wav"])
                info = sf.info(p)
                if info.samplerate != 16000 or info.channels != 1:
                    problems.append(
                        f"{r['wav']}: {info.samplerate}Hz {info.channels}ch")
            check(True, "표본 20개 16kHz 모노로 열림")
        except ImportError:
            notes.append("soundfile 없음 - 오디오 열기 검사 건너뜀")

    for m in notes:
        print(f"  OK  {m}" if not m.startswith("    ") else m)
    for m in problems:
        print(f"  !!  {m}")
    print(f"\n{'문제 없음' if not problems else f'문제 {len(problems)}건'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
