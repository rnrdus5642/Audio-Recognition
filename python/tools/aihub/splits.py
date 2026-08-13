"""Divide the children into train, valid and test.

Split by speaker, never by utterance. A child who appears on both sides
lets the model recognise the voice rather than the speech, and every
number measured after that is flattered.

Assignment is by hash of the speaker id, so it is stable: re-running
gives the same answer, and adding more children later leaves the
existing ones where they are instead of reshuffling every split.

Ages 5-7 are the product's users and get all three splits. Ages 8-9 go
to train only - they are there to teach the model what a child sounds
like, and there is no reason to spend eval budget on them.

    python python/tools/aihub/splits.py
"""

import collections
import csv
import hashlib
import json
import os

from config import AGES, INDEX, SPLITS, TARGET_AGES

# percentage points out of 100, by hash bucket
SHARE = [("train", 70), ("valid", 15), ("test", 15)]


def bucket(speaker):
    h = hashlib.sha256(("split:" + speaker).encode()).hexdigest()
    return int(h[:8], 16) % 100


def assign(speaker):
    b = bucket(speaker)
    edge = 0
    for name, share in SHARE:
        edge += share
        if b < edge:
            return name
    return SHARE[-1][0]


def main():
    if not os.path.exists(INDEX):
        print(f"색인이 없습니다: {INDEX}\n먼저 index.py 를 돌리세요.")
        return 1

    speakers = {}
    stats = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0.0]))
    with open(INDEX, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                age = int(r["age"])
            except ValueError:
                continue
            if age not in AGES or not r["speaker"]:
                continue
            sp = r["speaker"]
            if sp not in speakers:
                speakers[sp] = {
                    "age": age,
                    "split": assign(sp) if age in TARGET_AGES else "train",
                }
            split = speakers[sp]["split"]
            cell = stats[age][split]
            cell[0] += 1
            try:
                cell[1] += float(r["seconds"])
            except ValueError:
                pass

    if not speakers:
        print("색인에 해당 나이 화자가 없습니다.")
        return 1

    by_split = collections.Counter(v["split"] for v in speakers.values())
    print(f"화자 {len(speakers):,}명 → "
          + " · ".join(f"{k} {by_split[k]:,}" for k, _ in SHARE))

    print(f"\n{'나이':>4}{'분할':>8}{'화자':>7}{'발화':>9}{'시간':>9}")
    for age in sorted(stats):
        n_spk = collections.Counter(
            v["split"] for v in speakers.values() if v["age"] == age)
        for name, _ in SHARE:
            if not n_spk[name]:
                continue
            n, secs = stats[age][name]
            print(f"{age:>4}{name:>8}{n_spk[name]:>7,}{n:>9,}"
                  f"{secs/3600:>8.1f}h")

    total = collections.defaultdict(lambda: [0, 0.0])
    for age in stats:
        for name, cell in stats[age].items():
            total[name][0] += cell[0]
            total[name][1] += cell[1]
    print(f"\n{'합계':>4}")
    for name, _ in SHARE:
        n, secs = total[name]
        print(f"{'':>4}{name:>8}{by_split[name]:>7,}{n:>9,}{secs/3600:>8.1f}h")

    out = {
        "note": "화자 단위 분할. 해시 기반이라 데이터를 더 받아도 "
                "기존 화자의 소속은 바뀌지 않는다.",
        "speaker_id": "sha256(대문자 화자ID)[:12]",
        "share": dict(SHARE),
        "target_ages": sorted(TARGET_AGES),
        "train_only_ages": sorted(AGES - TARGET_AGES),
        "speakers": dict(sorted(speakers.items())),
    }
    os.makedirs(os.path.dirname(SPLITS), exist_ok=True)
    json.dump(out, open(SPLITS, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n{SPLITS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
