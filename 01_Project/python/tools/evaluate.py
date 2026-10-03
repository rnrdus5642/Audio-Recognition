"""Evaluation harness for the Korean phoneme-matching pipeline.

For every clip in the golden set:
  * Load audio (16 kHz mono).
  * Run the recognizer (audio -> IPA).
  * Score against the clip's OWN segment (positive case).
  * Score against EVERY other segment (negative cases).
  * Compare actual pass/fail with expected.

Outputs:
  * Per-case results (printed + JSON).
  * Aggregate metrics: positive accuracy, negative accuracy, mean score
    gap between positives and negatives.
  * Failure breakdown so we know exactly which words / matrix entries
    need tuning.

The recognizer's IPA output is cached per (audio_path, model_name) so
that iterative tuning of the confusion matrix / thresholds does not
re-run the (slow on CPU) ASR.

Usage:
    python -m python.tools.evaluate
    python -m python.tools.evaluate --refresh-cache
    python -m python.tools.evaluate --model kresnik/wav2vec2-large-xlsr-korean
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows so IPA characters print without
# triggering cp949 encoding errors.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:  # pragma: no cover - very old Python
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

from python.runtime.audio import load_audio_16k_mono
from python.runtime.matching import ConfusionMatrix, Matcher
from python.runtime.recognizer.ko import (
    DEFAULT_MODEL,
    KoreanASRRecognizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = PROJECT_ROOT / "shared" / "targets.json"
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v1.json"
)
GOLDEN_SET_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "golden_set.json"
)
CACHE_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "recognizer_cache.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "eval_results.json"
)


# ---------------------------------------------------------------------------
# Recognizer cache (so we don't re-run the ASR every iteration)
# ---------------------------------------------------------------------------


class RecognizerCache:
    def __init__(self, path: Path, model_name: str) -> None:
        self.path = path
        self.model_name = model_name
        self._data: dict[str, dict[str, list[str]]] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, audio_path: str) -> list[str] | None:
        rec = self._data.get(audio_path)
        if not rec:
            return None
        # Also store hangul transcripts under "{model}__hangul"
        return rec.get(self.model_name)

    def get_hangul(self, audio_path: str) -> str | None:
        rec = self._data.get(audio_path)
        if not rec:
            return None
        val = rec.get(f"{self.model_name}__hangul")
        return val if isinstance(val, str) else None

    def put(
        self,
        audio_path: str,
        ipa: list[str],
        hangul: str,
    ) -> None:
        self._data.setdefault(audio_path, {})[self.model_name] = ipa
        self._data[audio_path][f"{self.model_name}__hangul"] = hangul

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Targets helpers
# ---------------------------------------------------------------------------


def load_targets(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def answers_by_id(targets: dict) -> dict[str, dict]:
    return {a["id"]: a for a in targets["answers"]}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    targets: dict,
    matrix: ConfusionMatrix,
    golden_cases: list[dict],
    recognizer: KoreanASRRecognizer,
    cache: RecognizerCache,
    refresh_cache: bool,
    audio_root: Path,
    verbose: bool,
) -> dict:
    matcher = Matcher(matrix)
    by_answer = answers_by_id(targets)
    per_case: list[dict] = []

    n_pos = 0
    n_pos_pass = 0
    n_neg = 0
    n_neg_pass = 0  # negatives that incorrectly pass (false accepts)

    # Score collectors for analysis
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    confusion_pairs: dict[tuple[str, str], int] = defaultdict(int)

    t0 = time.time()
    for i, case in enumerate(golden_cases):
        rel_path = case["audio_path"]
        audio_path = audio_root / rel_path
        if not audio_path.exists():
            print(f"  missing audio: {rel_path}", file=sys.stderr)
            continue

        # Recognize (cache)
        ipa = None if refresh_cache else cache.get(rel_path)
        hangul = (
            None if refresh_cache else cache.get_hangul(rel_path)
        )
        if ipa is None:
            t_rec = time.time()
            audio = load_audio_16k_mono(audio_path)
            # One acoustic pass for both outputs - calling
            # transcribe_hangul() and then recognize() runs the model twice.
            hangul, ipa = recognizer.recognize_with_text(audio)
            cache.put(rel_path, ipa, hangul or "")
            rec_dt = time.time() - t_rec
            if verbose:
                print(
                    f"  [{i+1:>3}/{len(golden_cases)}] "
                    f"{rel_path}: {rec_dt:5.2f}s  "
                    f"hangul={hangul!r}  ipa={ipa}"
                )

        own_ans_id = case["target_answer_id"]
        own = by_answer[own_ans_id]

        # Positive: the word that was asked for, scored on its own.
        # Nothing competes: a lesson asks for one word and the caller
        # should not have to know which others exist.
        result = matcher.best_match(ipa or [], [own])
        pos_pass = result.passed
        n_pos += 1
        if pos_pass:
            n_pos_pass += 1
        pos_scores.append(result.score)

        # Negatives: every OTHER word. A false accept here means the
        # child said one thing and a different question would have
        # accepted it.
        neg_results = []
        for ans_id, answer in by_answer.items():
            if ans_id == own_ans_id:
                continue
            neg = matcher.best_match(ipa or [], [answer])
            n_neg += 1
            if neg.passed:
                n_neg_pass += 1
                confusion_pairs[(own_ans_id, ans_id)] += 1
            neg_scores.append(neg.score)
            neg_results.append(
                {
                    "answer_id": ans_id,
                    "score": round(neg.score, 4),
                    "passed": neg.passed,
                }
            )

        per_case.append(
            {
                "case_id": case["case_id"],
                "target_answer_id": own_ans_id,
                "target_text": case["target_text"],
                "voice": case["voice"],
                "hangul": hangul,
                "user_ipa": ipa,
                "positive": {
                    "score": round(result.score, 4),
                    "passed": pos_pass,
                },
                "negatives_summary": {
                    "total": len(neg_results),
                    "false_accepts": sum(
                        1 for r in neg_results if r["passed"]
                    ),
                    "max_score": (
                        max((r["score"] for r in neg_results), default=0.0)
                    ),
                },
            }
        )

    total_dt = time.time() - t0

    pos_acc = n_pos_pass / max(n_pos, 1)
    neg_acc = 1 - (n_neg_pass / max(n_neg, 1))  # higher = better rejection
    score_gap = (
        (sum(pos_scores) / len(pos_scores))
        - (sum(neg_scores) / len(neg_scores))
        if pos_scores and neg_scores
        else 0.0
    )

    summary = {
        "model": recognizer.name,
        "matrix_id": matrix.matrix_id,
        "n_positive_cases": n_pos,
        "n_positive_pass": n_pos_pass,
        "positive_accuracy": round(pos_acc, 4),
        "n_negative_cases": n_neg,
        "n_negative_false_accepts": n_neg_pass,
        "negative_rejection_rate": round(neg_acc, 4),
        "positive_mean_score": (
            round(sum(pos_scores) / len(pos_scores), 4)
            if pos_scores
            else None
        ),
        "negative_mean_score": (
            round(sum(neg_scores) / len(neg_scores), 4)
            if neg_scores
            else None
        ),
        "score_gap_pos_minus_neg": round(score_gap, 4),
        "elapsed_seconds": round(total_dt, 1),
        "top_confusions": sorted(
            (
                {"expected": e, "got": g, "count": c}
                for (e, g), c in confusion_pairs.items()
            ),
            key=lambda r: -r["count"],
        )[:5],
        "per_case": per_case,
    }
    return summary


def _print_summary(s: dict) -> None:
    print("\n" + "=" * 72)
    print(f"MODEL: {s['model']}")
    print(f"MATRIX: {s['matrix_id']}")
    print(f"Elapsed: {s['elapsed_seconds']}s")
    print("-" * 72)
    print(
        f"Positives:  {s['n_positive_pass']:>3}/{s['n_positive_cases']:<3} "
        f"= {s['positive_accuracy']:.1%}   "
        f"(mean score {s['positive_mean_score']})"
    )
    print(
        f"Negatives:  rejected "
        f"{s['n_negative_cases'] - s['n_negative_false_accepts']:>3}"
        f"/{s['n_negative_cases']:<3} "
        f"= {s['negative_rejection_rate']:.1%}   "
        f"(false-accepts {s['n_negative_false_accepts']}, "
        f"mean neg score {s['negative_mean_score']})"
    )
    print(
        f"Score gap (pos-neg): {s['score_gap_pos_minus_neg']:+.3f}  "
        "[higher = matcher separates target vs non-target better]"
    )
    if s["top_confusions"]:
        print("\nFalse accepts (spoken -> also accepted as):")
        for c in s["top_confusions"]:
            print(f"  {c['expected']:<10} -> {c['got']:<10} x{c['count']}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--targets", type=Path, default=TARGETS_PATH)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--golden", type=Path, default=GOLDEN_SET_PATH)
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore the recognizer cache and re-run all inference",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't write eval_results.json (still prints summary)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.targets.exists():
        print(
            f"ERROR: targets.json not found. Run "
            "`python -m python.build.build_targets` first.",
            file=sys.stderr,
        )
        return 1
    if not args.golden.exists():
        print(
            f"ERROR: golden_set.json not found. Run "
            "`python -m python.tools.generate_golden_audio` first.",
            file=sys.stderr,
        )
        return 1

    targets = load_targets(args.targets)
    matrix = ConfusionMatrix.from_json(args.matrix)
    golden = json.loads(args.golden.read_text(encoding="utf-8"))

    recognizer = KoreanASRRecognizer(model_name=args.model)
    cache = RecognizerCache(args.cache, args.model)

    print(f"Loading recognizer: {args.model}")
    print(f"  (first run downloads the model to ~/.cache/huggingface)")
    summary = evaluate(
        targets=targets,
        matrix=matrix,
        golden_cases=golden["cases"],
        recognizer=recognizer,
        cache=cache,
        refresh_cache=args.refresh_cache,
        audio_root=PROJECT_ROOT,
        verbose=args.verbose,
    )

    cache.save()
    if not args.no_save:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
