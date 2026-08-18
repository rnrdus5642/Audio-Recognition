"""Choose which utterances the fine-tune trains on.

The corpus holds 676 hours of 5-7 year olds across 942 speakers, and we
deliberately use a fraction of it. Past 150-200 hours the character error
rate stops moving and only the epoch gets longer; the reason the whole
download was worth doing is speaker count, not duration - a child's own
idiosyncrasy accounted for more of the variance than their age did.

So the sampling caps per speaker rather than taking a flat random slice.
The distribution is lopsided (p10 is 4 utterances, p90 is 1016) and a
flat sample would hand most of the budget to a few talkative children.

Two things get filtered out:

  * syllables the recogniser has no token for. Its vocabulary is
    syllable-level and fixed at 1205, and 0.75% of the corpus falls
    outside it - 11% of utterances carry at least one. Keeping the
    vocabulary as it is keeps the ONNX output shape, the C# decoder and
    the parity vectors all valid, and 600 hours still remain.
  * utterances too long to fit in VRAM alongside a 315M-parameter model.

Dev speakers are held out of training here rather than borrowed from the
valid split, which stays untouched so the final numbers mean something.

    python -m python.tools.child_finetune.manifest
    python -m python.tools.child_finetune.manifest --hours 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python.tools.aihub import config  # noqa: E402

HANGUL = re.compile(r"[^가-힣 ]")
OUT_DIR = Path(__file__).resolve().parent / "data"

# Below a second there is not enough audio to align a transcript to;
# above fifteen a single utterance starts crowding the batch.
MIN_SECONDS = 1.0
MAX_SECONDS = 15.0


def vocabulary() -> set[str]:
    """The syllables the acoustic model is able to emit."""
    root = Path(__file__).resolve().parents[3]
    path = (root / "unity" / "Assets" / "StreamingAssets"
            / "wav2vec2_ko_vocab.json")
    return set(json.loads(path.read_text(encoding="utf-8"))["tokens"])


def normalise(text: str) -> str:
    """Transcript as the CTC target: Hangul and spaces, nothing else.

    The transcripts are orthographic - 먹어요, not 머거요 - and stay that
    way. The device applies the phonological rules to whatever the model
    transcribes, so a model trained on surface forms would have them
    applied twice.
    """
    return " ".join(HANGUL.sub("", text).split())


def is_dev(speaker: str, share: float) -> bool:
    """Hash-based, so the same children land in dev on every re-run."""
    digest = hashlib.sha256(("dev:" + speaker).encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < share


def load(vocab: set[str]) -> tuple[list[dict], dict[str, int]]:
    """Every usable train-split utterance, with why the rest were not."""
    splits = json.loads(
        Path(config.SPLITS).read_text(encoding="utf-8"))["speakers"]
    rows: list[dict] = []
    dropped = {"vocab": 0, "short": 0, "long": 0, "empty": 0, "missing": 0}

    with open(config.INDEX, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            speaker = row["speaker"]
            meta = splits.get(speaker)
            if not meta or meta["split"] != "train":
                continue

            text = normalise(row["text"])
            if not text:
                dropped["empty"] += 1
                continue
            if any(c not in vocab for c in text if c != " "):
                dropped["vocab"] += 1
                continue

            seconds = float(row["seconds"] or 0)
            if seconds < MIN_SECONDS:
                dropped["short"] += 1
                continue
            if seconds > MAX_SECONDS:
                dropped["long"] += 1
                continue

            wav = os.path.join(
                config.AUDIO, f"{int(row['age']):02d}세", row["style"],
                row["wav"])
            if not os.path.exists(wav):
                dropped["missing"] += 1
                continue

            rows.append({
                "wav": wav,
                "text": text,
                "seconds": f"{seconds:.3f}",
                "start": row["speech_start"] or "",
                "end": row["speech_end"] or "",
                "speaker": speaker,
                "age": row["age"],
                "style": row["style"],
            })

    return rows, dropped


def cap_for(by_speaker: dict[str, list], hours: float) -> int:
    """Largest per-speaker cap that stays inside the hour budget.

    Utterances are ordered by the index, which is stable, so the same cap
    selects the same utterances every time.
    """
    target = hours * 3600

    def total(cap: int) -> float:
        return sum(
            sum(float(r["seconds"]) for r in rows[:cap])
            for rows in by_speaker.values())

    if total(10_000) <= target:
        return 10_000

    low, high = 1, 10_000
    while low < high:
        mid = (low + high + 1) // 2
        if total(mid) <= target:
            low = mid
        else:
            high = mid - 1
    return low


def readable(rows: list[dict]) -> list[dict]:
    """Drop the handful of wavs libsndfile cannot open.

    About 30 in 125,000 arrive with a broken header - they have a
    plausible size, so existence is not enough to catch them. Reading
    the header is cheap and doing it here means the training loop never
    has to survive a bad file mid-epoch.
    """
    def ok(row: dict) -> bool:
        try:
            sf.info(row["wav"])
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(16) as pool:
        keep = list(pool.map(ok, rows, chunksize=256))
    bad = len(rows) - sum(keep)
    if bad:
        print(f"  열리지 않는 wav {bad}개 제외")
    return [row for row, good in zip(rows, keep) if good]


def write(path: Path, rows: list[dict]) -> None:
    fields = ["wav", "text", "seconds", "start", "end",
              "speaker", "age", "style"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarise(name: str, rows: list[dict]) -> None:
    hours = sum(float(r["seconds"]) for r in rows) / 3600
    speakers = len({r["speaker"] for r in rows})
    print(f"  {name:5} {len(rows):8,} 발화  {hours:7.1f}h  화자 {speakers:4}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=200.0,
                        help="training budget in audio hours")
    parser.add_argument("--dev-share", type=float, default=0.055,
                        help="fraction of train speakers held out for "
                             "early stopping (0.055 = about 52)")
    parser.add_argument("--dev-hours", type=float, default=1.5,
                        help="dev is scored repeatedly during training, "
                             "so it is capped far below its speakers' "
                             "full output")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    vocab = vocabulary()
    print(f"어휘 {len(vocab)} 음절")
    rows, dropped = load(vocab)
    print(f"train split 사용 가능 {len(rows):,} 발화")
    print("  제외  " + "  ".join(f"{k} {v:,}" for k, v in dropped.items()))

    dev_pool, train_pool = {}, {}
    for row in rows:
        pool = (dev_pool if is_dev(row["speaker"], args.dev_share)
                else train_pool)
        pool.setdefault(row["speaker"], []).append(row)

    dev_cap = cap_for(dev_pool, args.dev_hours)
    dev_rows = [r for rows_ in dev_pool.values() for r in rows_[:dev_cap]]
    cap = cap_for(train_pool, args.hours)
    train_rows = [r for rows_ in train_pool.values() for r in rows_[:cap]]

    train_rows = readable(train_rows)
    dev_rows = readable(dev_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "train.csv", train_rows)
    write(args.out / "dev.csv", dev_rows)

    print(f"화자당 상한  train {cap} 발화 · dev {dev_cap} 발화")
    summarise("train", train_rows)
    summarise("dev", dev_rows)
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
