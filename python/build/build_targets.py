"""Build targets.json from words.csv.

Reads the authoring file (answer_id, text, language) and produces a
runtime-ready JSON containing IPA phoneme sequences and per-word
thresholds.

The output is a flat list of answers. Words used to be grouped into
segments that competed with each other, on the theory that "which of
these did the child say" is a safer question than "did they say this
one". Measured on the golden set, it made no difference at all -
positives 69.4% and negative rejection 91.7% either way - because the
per-word threshold is what actually decides. The grouping only forced
the application to know which other words were in play, so it is gone.

This script runs OFFLINE only - no model, no audio. The runtime never
calls G2P; it just loads the resulting targets.json.

Usage (from the project root, with venv activated):
    python -m python.build.build_targets

Optional args:
    --input   path to words.csv  (default: shared/words.csv)
    --output  path to targets.json (default: shared/targets.json)
    --matrix-id  confusion matrix id to embed (default: ko_child_v1)
    --strict     fail on validation warnings instead of just printing them
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Iterable, NamedTuple

from .g2p import get_g2p
from .g2p.ko.jamo_ipa import hangul_to_ipa_phonemes


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "shared" / "words.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "shared" / "targets.json"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


class WordRow(NamedTuple):
    answer_id: str
    text: str
    language: str


# ---------------------------------------------------------------------------
# Auto-threshold heuristic
# ---------------------------------------------------------------------------


def auto_threshold(n_phonemes: int) -> float:
    """Per-word default threshold based on phoneme count.

    Rationale:
      Shorter words have less information AND are more error-prone for
      the ASR (partial truncation, single-phoneme noise). They must be
      MORE strict to avoid false-accepts where a near-empty user output
      matches some word in a different segment by chance.

      Longer words tolerate more error per phoneme (False Reject is
      worse than False Accept for child UX).

    Returns a value in [0, 1] where higher = stricter match required.
    Values tuned against the v1 golden set (see eval_results.json).
    """
    if n_phonemes <= 2:
        return 0.85
    if n_phonemes == 3:
        return 0.73  # tuned against golden set: lets 아빠/앞 (0.733) pass
    if n_phonemes == 4:
        return 0.70
    if n_phonemes <= 6:
        return 0.65
    return 0.60


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_words_csv(path: Path) -> list[WordRow]:
    """Read and validate words.csv.

    Required columns: answer_id, text, language.

    The file is UTF-8 with a BOM, which is what makes Excel on a Korean
    Windows open it as UTF-8 rather than CP949 and show 엄마 instead of
    ?꾨쭏. Excel keeps the BOM when it saves, but other editors may not,
    so a CP949 file is read rather than rejected.
    """
    required = {"answer_id", "text", "language"}
    rows: list[WordRow] = []
    seen_answer_ids: set[str] = set()

    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = path.read_text(encoding="cp949")
        print(
            f"WARNING: {path.name} is not UTF-8; read as CP949. Save it as "
            "UTF-8 (with BOM) to keep Excel and this tool agreeing.",
            file=sys.stderr,
        )

    with io.StringIO(text, newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"words.csv missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for lineno, raw in enumerate(reader, start=2):
            row = {k: (v or "").strip() for k, v in raw.items()}
            for col in required:
                if not row[col]:
                    raise ValueError(
                        f"words.csv line {lineno}: empty value in column '{col}'"
                    )
            if row["answer_id"] in seen_answer_ids:
                raise ValueError(
                    f"words.csv line {lineno}: duplicate answer_id "
                    f"'{row['answer_id']}'"
                )
            seen_answer_ids.add(row["answer_id"])
            rows.append(
                WordRow(
                    answer_id=row["answer_id"],
                    text=row["text"],
                    language=row["language"],
                )
            )

    if not rows:
        raise ValueError(f"{path} contained no data rows")
    return rows


# ---------------------------------------------------------------------------
# Validation: phoneme collision detection
# ---------------------------------------------------------------------------


def find_collisions(answers: list[dict]) -> list[tuple[str, str]]:
    """Find pairs of answers that come out as the same phoneme sequence.

    Nothing downstream can tell them apart, so asking for one and hearing
    the other is indistinguishable from success.

    Returns list of (answer_a, answer_b) tuples.
    """
    by_phonemes: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for a in answers:
        by_phonemes[tuple(a["phonemes"])].append(a)

    collisions: list[tuple[str, str]] = []
    for group in by_phonemes.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                collisions.append((a["id"], b["id"]))
    return collisions


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def build_answer_entries(
    rows: Iterable[WordRow],
    apply_rules: bool = True,
) -> list[dict]:
    """Run each row through its language's G2P and return enriched dicts.

    With `apply_rules=False` the phonological rules are skipped and the
    text is mapped jamo by jamo - the same path the device runtime takes
    for the user's speech. That makes both sides identical and removes
    the Python dependency from adding words, at an accuracy cost that has
    to be measured rather than assumed (see `python.tools.evaluate`).
    """
    g2p_cache: dict[str, object] = {}
    out: list[dict] = []

    for row in rows:
        if row.language not in g2p_cache:
            g2p_cache[row.language] = get_g2p(row.language)
        g2p = g2p_cache[row.language]

        phonemes = (g2p.to_ipa(row.text) if apply_rules
                    else hangul_to_ipa_phonemes(row.text))
        if not phonemes:
            raise ValueError(
                f"G2P produced empty phoneme list for "
                f"'{row.text}' ({row.answer_id})"
            )

        out.append(
            {
                "id": row.answer_id,
                "text": row.text,
                "language": row.language,
                "phonemes": phonemes,
                "min_phonemes": len(phonemes),
                "threshold": round(auto_threshold(len(phonemes)), 4),
            }
        )
    return out


def build(
    input_path: Path,
    output_path: Path,
    matrix_id: str,
    *,
    strict: bool = False,
    apply_rules: bool = True,
) -> dict:
    rows = read_words_csv(input_path)
    answers = build_answer_entries(rows, apply_rules=apply_rules)

    collisions = find_collisions(answers)
    if collisions:
        msg = "Answers with identical phonemes (indistinguishable):"
        for a, b in collisions:
            msg += f"\n  - {a} == {b}"
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)

    languages = sorted({row.language for row in rows})

    targets = {
        "version": "2.0.0",
        "phoneme_set": "ipa",
        "languages": languages,
        "confusion_matrix_id": matrix_id,
        "build_date": date.today().isoformat(),
        "answers": answers,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return targets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT, help="Path to words.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write targets.json",
    )
    parser.add_argument(
        "--matrix-id",
        default="ko_child_v1",
        help="Confusion matrix id to embed in targets.json",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat validation warnings as errors",
    )
    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Skip phonological rules; map jamo directly, as the device "
             "runtime does for user speech",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 1

    targets = build(
        input_path=args.input,
        output_path=args.output,
        matrix_id=args.matrix_id,
        strict=args.strict,
        apply_rules=not args.no_rules,
    )

    print(
        f"OK: built {args.output} "
        f"({len(targets['answers'])} answers, "
        f"languages={targets['languages']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
