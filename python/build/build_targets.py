"""Build targets.json from words.csv.

Reads the authoring file (answer_id, text, language, and the optional
action_group) and produces a runtime-ready JSON containing IPA phoneme
sequences, per-word thresholds, and - for words that name an action
group - the competing actions the runtime must be able to tell them
apart from.

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
import math
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
    #: Which action this word asks for, or "" for the words that are not
    #: part of a contrast set. Two rows sharing a group are two ways of
    #: naming the same action, so they never compete with each other.
    action_group: str = ""


# ---------------------------------------------------------------------------
# Auto-threshold heuristic
# ---------------------------------------------------------------------------


def auto_threshold(n_phonemes: int) -> float:
    """Fallback threshold, from phoneme count alone.

    Only used for words that `--thresholds` does not cover. Length is a
    poor stand-in for how much unrelated speech a word attracts - 빵 and
    책 are both three phonemes and need 0.925 and 0.725 - so prefer a
    measured map when one exists.

    Short words (<= 6 phonemes) keep the golden-set values: a single
    wrong phoneme costs a quarter of a 4-phoneme word, so real speech
    already scores near the line and there is no room to raise it.

    Long words were the opposite of what the original rationale assumed.
    Measured on 142 words over 100 real adult utterances (2.5 s window,
    0.5 s hop, streak 2), a 7-phoneme target at 0.60 fired on unrelated
    speech in 14.6% of sessions - more than three times the rate of a
    4-phoneme target - because more phonemes means more ways for a long
    sentence to contain a passable window, and the loose threshold let
    them through. Raising 7+ to 0.75 cut that to 1.2% while detection
    went 95% -> 93% (7 phonemes) and 100% -> 100% (9 phonemes).

    Returns a value in [0, 1] where higher = stricter match required.

    Evidence is adult read speech. Child speech scores lower, so the
    short-word values stay put until there are child recordings to
    measure against.
    """
    if n_phonemes <= 2:
        return 0.85
    if n_phonemes == 3:
        return 0.73  # tuned against golden set: lets 아빠/앞 (0.733) pass
    if n_phonemes == 4:
        return 0.70
    if n_phonemes <= 6:
        return 0.65
    return 0.75


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def read_words_csv(path: Path) -> list[WordRow]:
    """Read and validate words.csv.

    Required columns: answer_id, text, language.

    Optional column: action_group. Empty or absent is what every row
    meant before the column existed - no contrast set, nothing embedded,
    nothing to veto. A value with whitespace in it is refused with its
    line number rather than trimmed into a different group id, because a
    group that nothing else joins silently protects nothing.

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
            action_group = row.get("action_group") or ""
            if any(ch.isspace() for ch in action_group):
                raise ValueError(
                    f"words.csv line {lineno}: action_group "
                    f"'{action_group}' has whitespace in it"
                )
            rows.append(
                WordRow(
                    answer_id=row["answer_id"],
                    text=row["text"],
                    language=row["language"],
                    action_group=action_group,
                )
            )

    if not rows:
        raise ValueError(f"{path} contained no data rows")
    return rows


# ---------------------------------------------------------------------------
# Validation: phoneme collision detection
# ---------------------------------------------------------------------------


def syllable_spans(
    text: str,
    groups: list[tuple[str, list[str]]] | None,
    phonemes: list[str],
) -> list[dict] | None:
    """Pair each syllable with the slice of `phonemes` it produced.

    Feedback has to name something the child can see - "사과의 과" - and
    a flat phoneme list cannot say which character a phoneme came from.
    It is not a fixed ratio: ㅘ is one character and two phonemes
    (사과 = 4 characters, 5 phonemes).

    Each entry carries both spellings, because they are not always the
    same syllable. `text` is what is printed on screen and `spoken` is
    how it comes out once the rules run. Under 연음 the two disagree
    about which syllable owns a consonant: 먹어요 is spoken 머거요, so
    the ㄱ written at the end of 먹 is spoken at the start of 거 and
    lands in the second entry. Six of the 29 curriculum words do this.
    Point at `text` to tell a reader where to look, but say `spoken` to
    describe the sound, or the two will contradict each other.

    `groups` is the same mapping run per spoken syllable, so the grouping
    is correct by construction as long as it still concatenates back to
    `phonemes`; when it does not, the two have drifted apart and guessing
    a boundary would point at the wrong syllable.

    Korean phonological rules rewrite syllables without adding or
    removing any, so written and spoken line up position by position.
    All 29 curriculum words hold to that; a word that ever broke it
    returns None rather than mislabelling every syllable after the
    break.
    """
    if groups is None:
        return None
    if [p for _, group in groups for p in group] != phonemes:
        return None

    written = [ch for ch in text if not ch.isspace()]
    voiced = [(ch, group) for ch, group in groups if group]
    if len(written) != len(voiced):
        return None

    spans: list[dict] = []
    start = 0
    for ch, (spoken, group) in zip(written, voiced):
        end = start + len(group)
        spans.append({"text": ch, "spoken": spoken, "span": [start, end]})
        start = end
    return spans


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


def add_contrastive_alternatives(answers: list[dict]) -> None:
    """Give each grouped answer every grouped answer, its own included.

    A word only needs telling apart from words that mean something else.
    두드리기 and 두드려 are the same instruction twice, so one winning
    over the other changes nothing; 들기 winning when the child said
    흔들기 changes which action the game performs, and 들기's phonemes
    are the tail of 흔들기's, so a matcher that may start its window
    anywhere cannot see the difference.

    Each grouped answer therefore ships EVERY grouped answer as
    `contrastive_alternatives` - text, phonemes and group - and the
    runtime splits them by group at scoring time. Own-group aliases are
    in there for a reason: a question may ask for 두드려 alone, and what
    the child said has to be measured against the whole action, 두드리기
    included, before another action is allowed to outscore it. Leaving
    them out let a rival beat an anchor that was missing the alias the
    child actually said. Nothing here is an answer: an alias that was not
    asked for can defend the action, never win it.

    A catalog with one group therefore still embeds its own entries -
    there is simply no other action to be outscored by.
    """
    grouped = [a for a in answers if a.get("action_group")]
    for answer in grouped:
        alternatives = [
            {
                "text": other["text"],
                "phonemes": list(other["phonemes"]),
                "action_group": other["action_group"],
            }
            for other in grouped
        ]
        if alternatives:
            answer["contrastive_alternatives"] = alternatives


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def load_thresholds(path: Path | None) -> dict[str, float]:
    """Measured per-word thresholds, keyed by word.

    Produced by tools/child_tuning/derive_thresholds.py, which scores
    each target against children saying other things and takes the
    lowest threshold that stays inside the false-accept budget. Words
    missing from the map fall back to `auto_threshold`.
    """
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["thresholds"]


def _finite(value: object) -> bool:
    """A real, finite number. `True` is not one, and neither is "0.01"."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _count(value: object) -> bool:
    """A non-negative whole count. `True` is not one."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def load_policy(path: Path | None, matrix_id: str | None = None) -> dict:
    """Per-word calibration policy, keyed by word.

    Carries the measured false-accept count behind each threshold and
    whether the one-exact-frame escape was calibrated for that word. Older
    threshold files have no policy block; then no word gets the escape and
    the strict two-frame rule stands, which is the safe direction.

    Returns the whole calibration, not just the per-word records: what
    licensed a word is the matrix and profile it was measured under, and
    the runtime has to be able to check that it is still running them.
    """
    if path is None:
        return {"policy": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    policy = data.get("policy") or {}
    if not policy:
        return {"policy": {}}
    for field in ("matrix_id", "consecutive", "budget", "streaming_profile",
                  "reference"):
        if field not in data:
            raise ValueError(
                f"threshold file has a policy block but no {field}; "
                "recalibrate so the policy is tied to what produced it")
    if not isinstance(data["matrix_id"], str) or not data["matrix_id"].strip():
        raise ValueError(
            "the policy does not name the matrix it was measured under")
    if not _count(data["consecutive"]) or data["consecutive"] < 1:
        raise ValueError(
            f"consecutive {data['consecutive']!r} is not a streak length")
    # Only the escape was calibrated at 2. A file that switches nothing on
    # is the ordinary strict path, and deriving those thresholds at a
    # different streak length is a normal thing to do.
    enabled = [word for word, record in policy.items()
               if record.get("allow_exact_single_frame") is True]
    if enabled and data["consecutive"] != 2:
        raise ValueError(
            "the exact-single-frame policy is only calibrated for "
            f"consecutive 2, not {data['consecutive']}")
    if matrix_id is not None and data["matrix_id"] != matrix_id:
        raise ValueError(
            f"policy was calibrated for {data['matrix_id']} but this build "
            f"targets {matrix_id}; recalibrate")
    budget = data["budget"]
    if not _finite(budget) or budget < 0:
        raise ValueError(f"the false-accept budget {budget!r} is not usable")
    profile = data["streaming_profile"]
    for key in ("skip_cost", "coverage", "context_mult"):
        value = profile.get(key)
        if not _finite(value):
            raise ValueError(f"streaming_profile.{key} is not a finite number")
    seconds = data["reference"].get("seconds_exact") or data["reference"].get("seconds")
    if not _finite(seconds) or seconds <= 0:
        raise ValueError("reference seconds are missing or not usable")
    for word, record in policy.items():
        if record.get("feasible") is not True:
            raise ValueError(
                f"'{word}' has no threshold that meets the false-accept "
                "budget; recalibrate rather than shipping the fallback")
        if record.get("allow_exact_single_frame") is not True and \
                record.get("allow_exact_single_frame") is not False:
            raise ValueError(f"'{word}' policy flag is not a boolean")
        if not _finite(record.get("threshold")):
            raise ValueError(f"'{word}' policy threshold is not a number")
        if record.get("threshold") != data["thresholds"].get(word):
            raise ValueError(
                f"'{word}' policy threshold disagrees with its threshold")
        rate = record.get("rate_per_10s")
        fired = record.get("false_accepts")
        if not _finite(rate) or rate < 0 or rate > budget:
            raise ValueError(
                f"'{word}' policy rate {rate} is missing or outside the "
                f"budget {budget}")
        if not _count(fired):
            raise ValueError(f"'{word}' false_accepts is not a count")
        recomputed = fired / seconds * 10.0
        if abs(recomputed - rate) > 1e-6:
            raise ValueError(
                f"'{word}' rate {rate} does not match {fired} firings over "
                f"{seconds:.1f}s")
        # The written rate is rounded, so checking only that one lets a
        # measurement of 0.0100001 ship as 0.010000 against a budget of
        # 0.01. The count and the seconds are what was measured.
        if recomputed > budget:
            raise ValueError(
                f"'{word}' fired {fired} times in {seconds:.1f}s - "
                f"{recomputed} per 10s, over the budget {budget}")
    return {
        "policy": policy,
        "matrix_id": data["matrix_id"],
        "streaming_profile": {k: float(profile[k])
                              for k in ("skip_cost", "coverage", "context_mult")},
        "consecutive": data["consecutive"],
    }


def build_answer_entries(
    rows: Iterable[WordRow],
    apply_rules: bool = True,
    thresholds: dict[str, float] | None = None,
    strict: bool = False,
    calibration: dict | None = None,
) -> list[dict]:
    """Run each row through its language's G2P and return enriched dicts.

    With `apply_rules=False` the phonological rules are skipped and the
    text is mapped jamo by jamo - the same path the device runtime takes
    for the user's speech. That makes both sides identical and removes
    the Python dependency from adding words, at an accuracy cost that has
    to be measured rather than assumed (see `python.tools.evaluate`).
    """
    thresholds = thresholds or {}
    calibration = calibration or {"policy": {}}
    policy = calibration.get("policy") or {}
    g2p_cache: dict[str, object] = {}
    out: list[dict] = []
    guessed: list[str] = []
    ungrouped: list[str] = []

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

        groups = (g2p.to_ipa_syllables(row.text) if apply_rules
                  else [(ch, hangul_to_ipa_phonemes(ch)) for ch in row.text])
        spans = syllable_spans(row.text, groups, phonemes)
        if spans is None:
            ungrouped.append(row.text)

        if thresholds and row.text not in thresholds:
            guessed.append(row.text)

        out.append(
            {
                "id": row.answer_id,
                "text": row.text,
                "language": row.language,
                "phonemes": phonemes,
                **({"syllables": spans} if spans else {}),
                "min_phonemes": len(phonemes),
                "threshold": round(
                    thresholds.get(row.text, auto_threshold(len(phonemes))), 4
                ),
                # Only for rows that named one, so a catalog built from a
                # file without the column is what it always was.
                **({"action_group": row.action_group}
                   if row.action_group else {}),
            }
        )
        record = policy.get(row.text)
        if record and record.get("allow_exact_single_frame") is True:
            # What was calibrated has to match what is being built, or the
            # approval belongs to a different word than the one shipping.
            if record.get("phonemes") != phonemes:
                raise ValueError(
                    f"'{row.text}' was calibrated for different phonemes; "
                    "recalibrate (did the G2P or --no-rules change?)")
            # Not `round(record["threshold"], 4)`: both sides would come
            # from the same number and always agree. What has to hold is
            # that the value shipped IS the value the budget was measured
            # at - 0.82594 serialised as 0.8259 is a looser pass mark that
            # nobody counted false accepts for.
            if record["threshold"] != out[-1]["threshold"]:
                raise ValueError(
                    f"'{row.text}' was calibrated at {record['threshold']} "
                    f"but the target file would ship "
                    f"{out[-1]['threshold']}; recalibrate on a grid that "
                    "survives 4-decimal rounding")
            out[-1]["allow_exact_single_frame"] = True
            # The snapshot the runtime checks itself against. Only written
            # when calibration switched it on, so older data and
            # uncalibrated words stay on the strict two-frame rule.
            out[-1]["exact_single_frame_calibration"] = {
                "matrix_id": calibration["matrix_id"],
                "streaming_profile": dict(calibration["streaming_profile"]),
                "consecutive": calibration["consecutive"],
                "text": row.text,
                "phonemes": list(phonemes),
                "threshold": out[-1]["threshold"],
            }
    if guessed:
        # The phoneme-count table lands within one step of the measured
        # value 19% of the time. 북 got 0.73 where measuring says 0.85,
        # and nothing said so.
        msg = ("측정된 임계값이 없어 음소 개수 기본값을 씁니다 "
               f"({len(guessed)}단어): {' '.join(guessed)}. "
               "python python/tools/child_tuning/derive_thresholds.py "
               "를 다시 돌리세요.")
        if strict:
            raise ValueError(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
    if ungrouped:
        # Not fatal: matching never reads the grouping. Only the
        # per-syllable feedback does, and it stays silent without it.
        print(f"WARNING: 음절 구분을 만들지 못했습니다 "
              f"({len(ungrouped)}단어): {' '.join(ungrouped)}. "
              "이 단어들은 발음 피드백에서 글자를 짚지 못합니다.",
              file=sys.stderr)

    add_contrastive_alternatives(out)
    return out


def build(
    input_path: Path,
    output_path: Path,
    matrix_id: str,
    *,
    strict: bool = False,
    apply_rules: bool = True,
    thresholds_path: Path | None = None,
) -> dict:
    rows = read_words_csv(input_path)
    answers = build_answer_entries(
        rows, apply_rules=apply_rules,
        thresholds=load_thresholds(thresholds_path), strict=strict,
        calibration=load_policy(thresholds_path, matrix_id))

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
        "--thresholds",
        type=Path,
        default=None,
        help="Measured per-word thresholds from "
             "tools/child_tuning/derive_thresholds.py; words it does not "
             "cover fall back to the phoneme-count default",
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
        thresholds_path=args.thresholds,
    )

    print(
        f"OK: built {args.output} "
        f"({len(targets['answers'])} answers, "
        f"languages={targets['languages']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
