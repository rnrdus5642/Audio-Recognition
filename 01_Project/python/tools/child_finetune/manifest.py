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
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python.tools.aihub import config  # noqa: E402
from python.tools import vocab_coverage  # noqa: E402

HANGUL = re.compile(r"[^가-힣 ]")
OUT_DIR = Path(__file__).resolve().parent / "data"
SHIPPED_VOCAB = vocab_coverage.DEFAULT_VOCAB

# Below a second there is not enough audio to align a transcript to;
# above fifteen a single utterance starts crowding the batch.
MIN_SECONDS = 1.0
MAX_SECONDS = 15.0


def vocabulary(path: Path | None = None) -> set[str]:
    """The syllables the acoustic model is able to emit.

    Defaults to the vocabulary that ships. Point it at a candidate
    checkpoint to build a manifest for a wider head - the loader bounds
    what it returns by that checkpoint's own `config.vocab_size`, so a
    tokenizer's sentence markers can never be mistaken for syllables the
    model can produce.
    """
    return set(vocab_coverage.load_vocabulary(path or SHIPPED_VOCAB).ids)


def wants(text: str, keep: set[str]) -> bool:
    """Whether this utterance carries a syllable we are short of."""
    return any(syllable in text for syllable in keep)


def normalise(text: str) -> str:
    """Transcript as the CTC target: Hangul and spaces, nothing else.

    The transcripts are orthographic - 먹어요, not 머거요 - and stay that
    way. The device applies the phonological rules to whatever the model
    transcribes, so a model trained on surface forms would have them
    applied twice.

    Composed to NFC first. A decomposed syllable is a run of jamo, none
    of which are in the 가-힣 range, so filtering before composing
    deletes it - silently, and worst of all in the mixed case, where
    엄마 놔 would arrive as 엄마 and train that target as a shorter word.
    """
    return " ".join(HANGUL.sub("", unicodedata.normalize("NFC", text)).split())


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


def pools(rows: list[dict], dev_share: float) -> tuple[dict, dict]:
    """Split by speaker, and keep the held-out speakers to themselves."""
    train_pool: dict[str, list[dict]] = {}
    dev_pool: dict[str, list[dict]] = {}
    for row in rows:
        pool = (dev_pool if is_dev(row["speaker"], dev_share) else train_pool)
        pool.setdefault(row["speaker"], []).append(row)
    return train_pool, dev_pool


def choose(by_speaker: dict[str, list[dict]], hours: float,
           keep: set[str] | None = None,
           repeat: int = 1) -> tuple[list[dict], int, set[int]]:
    """Cap each speaker, then put back what carries a wanted syllable.

    The cap keeps a speaker's first N utterances in index order, which is
    what stops a few talkative children owning the budget. It is also
    what quietly loses a rare syllable: a few hundred rows in four
    hundred thousand are almost never inside the slice. Widening the
    vocabulary alone would therefore change the manifest by nothing at
    all, so those rows are added back explicitly.

    `repeat` copies them. The counts are not computed here: rows still
    have to survive `readable`, and a report that counted a rare
    utterance the CSV never received would claim evidence that is not
    there. This returns the cap and which rows came back past it, and
    `tally` does the arithmetic once the survivors are known.
    """
    keep = keep or set()
    cap = cap_for(by_speaker, hours)
    chosen: list[dict] = []
    retained: set[int] = set()
    for rows in by_speaker.values():
        chosen.extend(rows[:cap])
        past = [row for row in rows[cap:] if wants(row["text"], keep)]
        retained.update(id(row) for row in past)
        chosen.extend(past)

    carrying = [row for row in chosen if wants(row["text"], keep)]
    return chosen + carrying * max(repeat - 1, 0), cap, retained


def tally(rows: list[dict], keep: set[str] | None, cap: int,
          retained: set[int]) -> dict:
    """Counts describing exactly the rows that will be written.

    Called after unreadable files are dropped, so every number here is a
    number of lines in the CSV. Unique and repeated seconds are kept
    apart: an oversampled hour is not an hour of speech.
    """
    keep = keep or set()
    seen: set[int] = set()
    unique: list[dict] = []
    copies = 0
    for row in rows:
        if id(row) in seen:
            copies += 1
            continue
        seen.add(id(row))
        unique.append(row)

    carrying = [row for row in unique if wants(row["text"], keep)]
    seconds = sum(float(row["seconds"]) for row in unique)
    return {
        "cap": cap,
        "unique": len(unique),
        "retained_past_cap": sum(1 for row in unique if id(row) in retained),
        "carrying": len(carrying),
        "extra_copies": copies,
        "unique_seconds": seconds,
        "total_seconds": sum(float(row["seconds"]) for row in rows),
        "speakers_carrying": len({row["speaker"] for row in carrying}),
    }


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


def report(name: str, stats: dict, keep: set[str]) -> None:
    """Say what retention and repetition actually did, in both currencies."""
    if not keep:
        return
    print(f"  {name:5} 상한 밖에서 되살린 발화 {stats['retained_past_cap']:,} · "
          f"해당 음절을 가진 발화 {stats['carrying']:,} "
          f"(화자 {stats['speakers_carrying']})")
    if stats["extra_copies"]:
        print(f"        복제 {stats['extra_copies']:,} · "
              f"고유 {stats['unique_seconds']/3600:.2f}h → "
              f"학습 노출 {stats['total_seconds']/3600:.2f}h "
              "(복제는 새로운 음성이 아닙니다)")


def check_out(out: Path, force: bool) -> str | None:
    """Never land a different manifest on top of the one in use."""
    existing = [name for name in ("train.csv", "dev.csv")
                if (out / name).exists()]
    if existing and not force:
        return (f"{out} 에 이미 {', '.join(existing)} 이 있습니다. "
                "새 --out 을 주거나 --force 를 붙이세요")
    return None


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
    parser.add_argument("--vocab", type=Path, default=None,
                        help="어휘 출처: 체크포인트 디렉터리나 어휘 파일. "
                             "기본은 배포 중인 Unity 어휘")
    parser.add_argument("--keep-syllables", action="append", default=[],
                        help="화자당 상한을 넘겨도 남길 음절 (여러 번 가능)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="위 음절을 가진 train 발화를 몇 번 노출할지. "
                             "dev 는 복제하지 않습니다")
    parser.add_argument("--force", action="store_true",
                        help="이미 manifest 가 있는 --out 에 덮어쓰기")
    args = parser.parse_args()

    keep = {s for entry in args.keep_syllables
            for s in unicodedata.normalize("NFC", entry)}
    # Resolved, because `data`, `./data` and an absolute path are the
    # same directory and only one of them loses a string comparison.
    if args.vocab is not None and args.out.resolve() == OUT_DIR.resolve():
        print("ERROR: --vocab 를 바꾸면 --out 도 새로 지정하세요. 배포 "
              "어휘로 만든 manifest 를 덮어쓰지 않습니다", file=sys.stderr)
        return 1

    problem = check_out(args.out, args.force)
    if problem:
        print(f"ERROR: {problem}", file=sys.stderr)
        return 1

    try:
        vocab = vocabulary(args.vocab)
    except (vocab_coverage.VocabularyError, OSError, KeyError) as e:
        print(f"ERROR: 어휘를 읽지 못했습니다: {e}", file=sys.stderr)
        return 1
    print(f"어휘 {len(vocab)} 음절")
    outside = sorted(s for s in keep if s not in vocab)
    if outside:
        print(f"ERROR: {outside} 이 어휘에 없습니다. 그 음절이 든 발화는 "
              "되살리기 전에 이미 걸러집니다 - --vocab 을 확인하세요",
              file=sys.stderr)
        return 1

    rows, dropped = load(vocab)
    print(f"train split 사용 가능 {len(rows):,} 발화")
    print("  제외  " + "  ".join(f"{k} {v:,}" for k, v in dropped.items()))

    train_pool, dev_pool = pools(rows, args.dev_share)
    # Retention runs in both pools so a held-out speaker's rare rows stay
    # in dev rather than crossing into train. Only train is oversampled:
    # a repeated dev row would score the same audio twice.
    dev_rows, dev_cap, dev_kept = choose(dev_pool, args.dev_hours, keep)
    train_rows, cap, train_kept = choose(
        train_pool, args.hours, keep, args.repeat)

    # Counted after this, never before: a broken wav that never reaches
    # the CSV must not be reported as evidence that did.
    train_rows = readable(train_rows)
    dev_rows = readable(dev_rows)
    train_stats = tally(train_rows, keep, cap, train_kept)
    dev_stats = tally(dev_rows, keep, dev_cap, dev_kept)

    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "train.csv", train_rows)
    write(args.out / "dev.csv", dev_rows)

    print(f"화자당 상한  train {cap} 발화 · dev {dev_cap} 발화")
    summarise("train", train_rows)
    summarise("dev", dev_rows)
    report("train", train_stats, keep)
    report("dev", dev_stats, keep)
    if keep and dev_stats["retained_past_cap"]:
        print("  dev 에도 해당 음절 발화가 들어갔습니다. dev CER 은 이제 "
              "이전 실행과 같은 집합이 아니니 전체 CER 비교에 쓰지 마세요")
    print(f"→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
