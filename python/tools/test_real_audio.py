"""Test the recognition pipeline on real audio.

Designed for ad-hoc testing with arbitrary user-provided audio: self
recordings, child speech samples, downloaded clips, etc.

Five modes:

  1. SINGLE FILE against a known target (looked up in targets.json):
       python -m python.tools.test_real_audio path/to/clip.wav \
              --target 사과 --segment lesson_03_food

     If --segment is omitted, the script searches all segments and
     uses the first one containing --target.

  2. CUSTOM TARGET (word NOT in targets.json):
       python -m python.tools.test_real_audio path/to/clip.wav \
              --custom-target 맥주

     g2pkk runs on the fly; auto-threshold based on phoneme count.

  3. PROBE mode (just show ASR / IPA, no matching):
       python -m python.tools.test_real_audio path/to/clip.wav --probe

  4. SCAN-ALL (try every segment, find best matching one):
       python -m python.tools.test_real_audio path/to/clip.wav --scan-all

  5. BATCH via manifest CSV:
       python -m python.tools.test_real_audio --manifest my_tests.csv

     CSV columns: audio_path, target_text, [target_segment_id]
     Relative paths are resolved against the manifest file's directory.

Output:
  - Pretty console table per case (ASR / IPA / candidate scores)
  - Aggregate pass/fail summary (batch mode)
  - Optional JSON dump via --json-out

Common options:
  --pad <ms>          Pre/post-pad audio with silence (helps short clips)
  --model <name>      Use a different HF ASR model than the default
  --hangul-only       Skip IPA matching, show only ASR output
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout on Windows so IPA chars print cleanly.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:  # pragma: no cover
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

import numpy as np

from python.build.build_targets import auto_threshold
from python.runtime.audio import load_audio_16k_mono
from python.runtime.matching import ConfusionMatrix, Matcher
# NOTE: scoring now goes through Matcher.score_against, which dispatches
# to substring or exact edit distance based on mode. The lower-level
# functions are not needed here anymore.
from python.runtime.recognizer.ko import (
    DEFAULT_MODEL,
    KoreanASRRecognizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = PROJECT_ROOT / "shared" / "targets.json"
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v1.json"
)


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class CandidateScore:
    answer_id: str
    text: str
    phonemes: list[str]
    distance: float
    score: float
    threshold: float
    is_best: bool = False
    # Substring matching: which slice of user_ipa actually matched
    window_start: int = 0
    window_end: int = 0
    alignment: list[tuple[str, str, str]] = field(default_factory=list)

    @property
    def would_pass(self) -> bool:
        return self.score >= self.threshold


@dataclass
class TestResult:
    audio_path: str
    audio_duration_s: float
    hangul: str
    user_ipa: list[str]
    segment_id: Optional[str]
    target_text: Optional[str]
    candidates: list[CandidateScore] = field(default_factory=list)
    cross_segment_best: Optional["CandidateScore"] = None

    @property
    def best(self) -> Optional[CandidateScore]:
        return next((c for c in self.candidates if c.is_best), None)

    @property
    def overall_pass(self) -> Optional[bool]:
        b = self.best
        if b is None or self.target_text is None:
            return None
        return b.would_pass and b.text == self.target_text


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------


def _fmt_ipa(ipa: list[str]) -> str:
    if not ipa:
        return "[]"
    return "[" + ", ".join(ipa) + "]"


def _fmt_passed(passed: bool) -> str:
    return "PASS" if passed else "fail"


def _print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_audio_info(path: Path, audio: np.ndarray) -> None:
    duration = len(audio) / 16000
    print(f"Audio: {path}")
    print(
        f"  Duration: {duration:.2f}s @ 16kHz mono "
        f"({len(audio)} samples)"
    )
    if duration < 0.3:
        print("  ⚠ Very short (< 0.3s) - ASR may struggle")
    if duration > 5.0:
        print("  ⚠ Long (> 5.0s) - matcher expects single words/phrases")
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"  Level: RMS={rms:.4f}  peak={peak:.4f}")
    if peak < 0.05:
        print("  ⚠ Very quiet - consider boosting input level")


def _print_candidates(
    candidates: list[CandidateScore],
    target_text: Optional[str],
) -> None:
    """Print a table of candidates with their scores."""
    # Compute column widths
    id_w = max((len(c.answer_id) for c in candidates), default=4)
    text_w = max((len(c.text) for c in candidates), default=4)
    ipa_w = max((len(_fmt_ipa(c.phonemes)) for c in candidates), default=4)
    id_w = max(id_w, 6)
    ipa_w = min(max(ipa_w, 20), 40)

    header = (
        f"  {'answer':<{id_w}}  {'text':<{text_w + 2}}  "
        f"{'IPA':<{ipa_w}}  {'dist':>6}  {'score':>6}  "
        f"{'thr':>5}  result"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in candidates:
        marker = " ← BEST" if c.is_best else ""
        target_marker = "*" if (target_text and c.text == target_text) else " "
        ipa_str = _fmt_ipa(c.phonemes)
        if len(ipa_str) > ipa_w:
            ipa_str = ipa_str[: ipa_w - 1] + "…"
        print(
            f"  {target_marker}{c.answer_id:<{id_w - 1}}  "
            f"{c.text:<{text_w + 2}}  "
            f"{ipa_str:<{ipa_w}}  "
            f"{c.distance:>6.2f}  {c.score:>6.3f}  "
            f"{c.threshold:>5.2f}  "
            f"{_fmt_passed(c.would_pass)}{marker}"
        )
    if target_text:
        print(f"  ('*' marks the intended target: '{target_text}')")


def _print_test_result(result: TestResult) -> None:
    print()
    print(f"ASR output:")
    print(f"  Hangul: '{result.hangul}'")
    print(f"  IPA:    {_fmt_ipa(result.user_ipa)}")
    print()
    if result.segment_id:
        print(f"Match analysis (segment: {result.segment_id})")
        _print_candidates(result.candidates, result.target_text)
    elif result.candidates:
        print("Match analysis (cross-segment scan)")
        _print_candidates(result.candidates, result.target_text)

    if result.cross_segment_best:
        print()
        print(
            f"⚠ Strongest cross-segment match: "
            f"{result.cross_segment_best.text} "
            f"({result.cross_segment_best.answer_id}, "
            f"score {result.cross_segment_best.score:.3f}, "
            f"{'would PASS' if result.cross_segment_best.would_pass else 'rejected'})"
        )

    if result.target_text is not None:
        print()
        overall = result.overall_pass
        if overall is True:
            print(f"✓ PASS — matched '{result.target_text}' as intended.")
        else:
            best = result.best
            if best is None:
                print(f"✗ FAIL — no candidates to match against.")
            elif best.text != result.target_text:
                print(
                    f"✗ FAIL — matcher picked '{best.text}' "
                    f"(score {best.score:.3f}) instead of "
                    f"'{result.target_text}'."
                )
            else:
                print(
                    f"✗ FAIL — correct word picked but score "
                    f"{best.score:.3f} below threshold {best.threshold:.2f}."
                )


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------


class AudioTester:
    """Multi-language audio recognition + matching.

    Lazy-loads a per-language recognizer (and confusion matrix) the
    first time each language is used, then caches it for subsequent
    calls. The Korean recognizer alone uses ~1.5 GB of RAM, so we never
    pre-load more than the languages actually in use.
    """

    def __init__(
        self,
        targets_path: Path = TARGETS_PATH,
        matrix_path: Path = MATRIX_PATH,
        model_name: str | None = None,
    ) -> None:
        self.targets = json.loads(targets_path.read_text(encoding="utf-8"))
        # Korean's matrix is the default for backwards compatibility.
        # Other languages can ship their own ko_child_v1.json siblings.
        self._default_matrix_path = matrix_path
        self._matrix_paths: dict[str, Path] = {"ko": matrix_path}
        self._matrices: dict[str, ConfusionMatrix] = {}
        self._matchers: dict[str, Matcher] = {}
        self._recognizers: dict[str, object] = {}
        self._custom_model_name = model_name  # only used for Korean today

        # Build segment / answer indexes (language-aware)
        self._segments_by_id = {s["id"]: s for s in self.targets["segments"]}
        self._answer_index: dict[str, tuple[str, dict]] = {}
        for seg in self.targets["segments"]:
            for a in seg["answers"]:
                self._answer_index[a["text"]] = (seg["id"], a)
                self._answer_index[a["id"]] = (seg["id"], a)

    # ----- language-aware accessors -----

    def available_target_languages(self) -> list[str]:
        """Languages that have at least one entry in targets.json."""
        return sorted({s["language"] for s in self.targets["segments"]})

    def _matrix_for(self, language: str) -> ConfusionMatrix:
        if language in self._matrices:
            return self._matrices[language]
        path = self._matrix_paths.get(language)
        if path is None:
            # Try a sibling matrix `<lang>_child_v1.json`
            candidate = (
                self._default_matrix_path.parent
                / f"{language}_child_v1.json"
            )
            if candidate.exists():
                path = candidate
            else:
                raise FileNotFoundError(
                    f"No confusion matrix for language {language!r}. "
                    f"Expected {candidate}."
                )
        m = ConfusionMatrix.from_json(path)
        self._matrices[language] = m
        self._matchers[language] = Matcher(m)
        return m

    def _matcher_for(self, language: str) -> Matcher:
        self._matrix_for(language)  # ensures _matchers populated
        return self._matchers[language]

    def _recognizer_for(self, language: str):
        if language in self._recognizers:
            return self._recognizers[language]
        from python.runtime.recognizer import get_recognizer

        kwargs = {}
        if language == "ko" and self._custom_model_name:
            kwargs["model_name"] = self._custom_model_name
        rec = get_recognizer(language, **kwargs)
        self._recognizers[language] = rec
        return rec

    # Backwards-compatible attribute used by older test code.
    @property
    def matrix(self) -> ConfusionMatrix:
        return self._matrix_for("ko")

    @property
    def matcher(self) -> Matcher:
        return self._matcher_for("ko")

    @property
    def recognizer(self):
        return self._recognizer_for("ko")

    # ----- helpers -----

    def _make_custom_target(
        self, text: str, language: str = "ko"
    ) -> tuple[str, dict]:
        """Build an on-the-fly candidate for a word not in targets.json."""
        from python.build.g2p import get_g2p

        g2p = get_g2p(language)
        phonemes = g2p.to_ipa(text)
        if not phonemes:
            raise ValueError(
                f"G2P produced empty phoneme list for {text!r}"
            )
        return (
            "__custom__",
            {
                "id": "custom_target",
                "text": text,
                "phonemes": phonemes,
                "min_phonemes": len(phonemes),
                "threshold": round(auto_threshold(len(phonemes)), 4),
            },
        )

    def _score_against(
        self,
        user_ipa: list[str],
        candidates: list[dict],
        language: str = "ko",
    ) -> list[CandidateScore]:
        """Score user IPA against every candidate; mark the best.

        Uses the language's Matcher (substring by default) so noise
        outside the matched window does not count against the score.
        """
        matcher = self._matcher_for(language)
        rows: list[CandidateScore] = []
        for c in candidates:
            d, s, ws, we, ops = matcher.score_against(
                user_ipa, c["phonemes"]
            )
            rows.append(
                CandidateScore(
                    answer_id=c["id"],
                    text=c["text"],
                    phonemes=c["phonemes"],
                    distance=d,
                    score=s,
                    threshold=float(c.get("threshold", 0.6)),
                    window_start=ws,
                    window_end=we,
                    alignment=ops,
                )
            )
        if rows:
            best_idx = max(range(len(rows)), key=lambda i: rows[i].score)
            rows[best_idx].is_best = True
        return rows

    # ----- main entry points -----

    def load_audio(
        self, audio_path: Path, pad_ms: int = 0
    ) -> np.ndarray:
        audio = load_audio_16k_mono(audio_path)
        if pad_ms > 0:
            pad_samples = int(16000 * pad_ms / 1000)
            audio = np.concatenate(
                [
                    np.zeros(pad_samples, dtype=np.float32),
                    audio,
                    np.zeros(pad_samples, dtype=np.float32),
                ]
            )
        return audio

    def probe(
        self, audio_path: Path, pad_ms: int = 0
    ) -> TestResult:
        """Just run ASR, no matching."""
        audio = self.load_audio(audio_path, pad_ms=pad_ms)
        _print_audio_info(audio_path, audio)
        hangul = self.recognizer.transcribe_hangul(audio)
        ipa = self.recognizer.recognize(audio) if hangul else []
        return TestResult(
            audio_path=str(audio_path),
            audio_duration_s=len(audio) / 16000,
            hangul=hangul,
            user_ipa=ipa,
            segment_id=None,
            target_text=None,
        )

    # ----- in-memory variants (used by record_live and any future runtime) -----

    def test_array(
        self,
        audio: np.ndarray,
        target_text: Optional[str] = None,
        segment_id: Optional[str] = None,
        custom_target: bool = False,
        scan_all: bool = False,
        audio_label: str = "<live>",
        language: str = "ko",
    ) -> TestResult:
        """Recognise `audio` and score against the appropriate candidates.

        The `language` parameter is used:
          * for the recogniser + matcher in scan_all / probe modes
          * for the G2P of a custom target
        Otherwise the language is inferred from the target's segment
        (segments in targets.json carry a language tag).
        """
        # Resolve target
        target_seg_id = segment_id
        custom_cand: Optional[dict] = None
        if custom_target:
            if not target_text:
                raise ValueError(
                    "custom_target=True requires target_text"
                )
            target_seg_id, custom_cand = self._make_custom_target(
                target_text, language=language
            )
        elif target_text and segment_id is None:
            if target_text not in self._answer_index:
                raise ValueError(
                    f"'{target_text}' not in targets.json. "
                    "Pass custom_target=True or supply segment_id."
                )
            target_seg_id, _ = self._answer_index[target_text]

        # Determine the language to use for recognition + matching
        if target_seg_id and target_seg_id in self._segments_by_id:
            inferred_lang = self._segments_by_id[target_seg_id].get(
                "language", language
            )
        else:
            inferred_lang = language

        # Recognise (lazy-loads the language's model on first call)
        recognizer = self._recognizer_for(inferred_lang)
        hangul = recognizer.transcribe_hangul(audio)
        ipa = recognizer.recognize(audio) if hangul else []

        # Build candidate list
        if custom_cand is not None:
            candidates = [custom_cand]
        elif scan_all:
            # Scan only segments in the matching language to avoid
            # cross-language IPA comparison noise.
            candidates = []
            for seg in self.targets["segments"]:
                if seg.get("language", "ko") == inferred_lang:
                    candidates.extend(seg["answers"])
        else:
            if target_seg_id is None:
                # No target, no scan: just probe (caller wanted ASR only)
                return TestResult(
                    audio_path=audio_label,
                    audio_duration_s=len(audio) / 16000,
                    hangul=hangul,
                    user_ipa=ipa,
                    segment_id=None,
                    target_text=target_text,
                )
            seg = self._segments_by_id.get(target_seg_id)
            if seg is None:
                raise ValueError(
                    f"Segment '{target_seg_id}' not found in targets.json"
                )
            candidates = seg["answers"]

        rows = self._score_against(ipa, candidates, language=inferred_lang)

        # Cross-segment risk check (within the same language only)
        cross_best: Optional[CandidateScore] = None
        if target_seg_id is not None and not scan_all and not custom_target:
            for seg in self.targets["segments"]:
                if seg["id"] == target_seg_id:
                    continue
                if seg.get("language", "ko") != inferred_lang:
                    continue
                others = self._score_against(
                    ipa, seg["answers"], language=inferred_lang
                )
                for o in others:
                    if cross_best is None or o.score > cross_best.score:
                        cross_best = o
            if cross_best is not None:
                cross_best.is_best = False

        return TestResult(
            audio_path=audio_label,
            audio_duration_s=len(audio) / 16000,
            hangul=hangul,
            user_ipa=ipa,
            segment_id=target_seg_id,
            target_text=target_text,
            candidates=rows,
            cross_segment_best=cross_best,
        )

    def test_single(
        self,
        audio_path: Path,
        target_text: Optional[str],
        segment_id: Optional[str],
        custom_target: bool = False,
        pad_ms: int = 0,
        scan_all: bool = False,
    ) -> TestResult:
        # Resolve target
        target_seg_id = segment_id
        custom_cand: Optional[dict] = None
        if custom_target:
            if not target_text:
                raise ValueError(
                    "--custom-target requires a target text"
                )
            target_seg_id, custom_cand = self._make_custom_target(target_text)
        elif target_text and segment_id is None:
            # Auto-locate the target in targets.json
            if target_text not in self._answer_index:
                raise ValueError(
                    f"'{target_text}' not in targets.json. Use "
                    "--custom-target, or specify --segment."
                )
            target_seg_id, _ = self._answer_index[target_text]

        # Recognize
        audio = self.load_audio(audio_path, pad_ms=pad_ms)
        _print_audio_info(audio_path, audio)
        hangul = self.recognizer.transcribe_hangul(audio)
        ipa = self.recognizer.recognize(audio) if hangul else []

        # Build candidate list
        if custom_cand is not None:
            candidates = [custom_cand]
        elif scan_all:
            candidates = []
            for seg in self.targets["segments"]:
                candidates.extend(seg["answers"])
        else:
            if target_seg_id is None:
                raise ValueError(
                    "No segment specified and no target to look up."
                )
            seg = self._segments_by_id.get(target_seg_id)
            if seg is None:
                raise ValueError(
                    f"Segment '{target_seg_id}' not found in targets.json"
                )
            candidates = seg["answers"]

        rows = self._score_against(ipa, candidates)

        # Cross-segment risk check (only when matching within a single segment)
        cross_best: Optional[CandidateScore] = None
        if target_seg_id is not None and not scan_all and not custom_target:
            for seg in self.targets["segments"]:
                if seg["id"] == target_seg_id:
                    continue
                others = self._score_against(ipa, seg["answers"])
                for o in others:
                    if cross_best is None or o.score > cross_best.score:
                        cross_best = o
            # The cross_best should NOT be marked as the in-segment best
            if cross_best is not None:
                cross_best.is_best = False

        return TestResult(
            audio_path=str(audio_path),
            audio_duration_s=len(audio) / 16000,
            hangul=hangul,
            user_ipa=ipa,
            segment_id=target_seg_id,
            target_text=target_text,
            candidates=rows,
            cross_segment_best=cross_best,
        )


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def _read_manifest(path: Path) -> list[dict]:
    base = path.parent
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"audio_path", "target_text"} - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}. "
                "Required: audio_path, target_text. "
                "Optional: target_segment_id."
            )
        for raw in reader:
            row = {k: (v or "").strip() for k, v in raw.items()}
            audio_path = Path(row["audio_path"])
            if not audio_path.is_absolute():
                audio_path = base / audio_path
            rows.append(
                {
                    "audio_path": audio_path,
                    "target_text": row["target_text"],
                    "target_segment_id": (
                        row.get("target_segment_id") or None
                    ),
                }
            )
    return rows


def run_batch(
    tester: AudioTester,
    rows: list[dict],
    pad_ms: int = 0,
) -> list[TestResult]:
    results: list[TestResult] = []
    for i, row in enumerate(rows, start=1):
        print()
        _print_header(
            f"[{i}/{len(rows)}] {row['audio_path'].name}  "
            f"→ target '{row['target_text']}'"
        )
        try:
            r = tester.test_single(
                audio_path=row["audio_path"],
                target_text=row["target_text"],
                segment_id=row["target_segment_id"],
                pad_ms=pad_ms,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        _print_test_result(r)
        results.append(r)
    return results


def _print_batch_summary(results: list[TestResult]) -> None:
    print()
    _print_header(f"BATCH SUMMARY ({len(results)} tests)")
    n_pass = sum(1 for r in results if r.overall_pass)
    print(
        f"Passed: {n_pass}/{len(results)} "
        f"= {n_pass / max(len(results), 1):.1%}"
    )
    print()
    print(f"{'#':<4} {'audio':<32} {'target':<10} {'ASR':<14} {'score':>6}  result")
    print("  " + "-" * 78)
    for i, r in enumerate(results, start=1):
        b = r.best
        score = b.score if b else 0.0
        status = "PASS" if r.overall_pass else "FAIL"
        audio_name = Path(r.audio_path).name
        if len(audio_name) > 30:
            audio_name = audio_name[:29] + "…"
        hangul = r.hangul or "(empty)"
        if len(hangul) > 12:
            hangul = hangul[:11] + "…"
        target = r.target_text or ""
        if len(target) > 8:
            target = target[:7] + "…"
        print(
            f"{i:<4} {audio_name:<32} {target:<10} {hangul:<14} "
            f"{score:>6.3f}  {status}"
        )


# ---------------------------------------------------------------------------
# JSON dump
# ---------------------------------------------------------------------------


def _result_to_dict(r: TestResult) -> dict:
    return {
        "audio_path": r.audio_path,
        "audio_duration_s": round(r.audio_duration_s, 3),
        "hangul": r.hangul,
        "user_ipa": r.user_ipa,
        "segment_id": r.segment_id,
        "target_text": r.target_text,
        "best": (
            {
                "answer_id": r.best.answer_id,
                "text": r.best.text,
                "score": round(r.best.score, 4),
                "passed": r.best.would_pass,
            }
            if r.best
            else None
        ),
        "overall_pass": r.overall_pass,
        "candidates": [
            {
                "answer_id": c.answer_id,
                "text": c.text,
                "distance": round(c.distance, 4),
                "score": round(c.score, 4),
                "threshold": c.threshold,
                "would_pass": c.would_pass,
                "is_best": c.is_best,
            }
            for c in r.candidates
        ],
        "cross_segment_best": (
            {
                "answer_id": r.cross_segment_best.answer_id,
                "text": r.cross_segment_best.text,
                "score": round(r.cross_segment_best.score, 4),
                "would_pass": r.cross_segment_best.would_pass,
            }
            if r.cross_segment_best
            else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  test_real_audio.py my.wav --target 사과\n"
            "  test_real_audio.py my.wav --target 사과 --segment lesson_03_food\n"
            "  test_real_audio.py my.wav --custom-target 맥주\n"
            "  test_real_audio.py my.wav --probe\n"
            "  test_real_audio.py my.wav --scan-all\n"
            "  test_real_audio.py --manifest recordings.csv\n"
        ),
    )
    parser.add_argument(
        "audio", nargs="?", type=Path, help="Audio file path"
    )
    parser.add_argument(
        "--target",
        help="Target Korean word to match against (e.g., 사과)",
    )
    parser.add_argument(
        "--segment",
        help="Segment id in targets.json (e.g., lesson_03_food). "
        "Auto-detected from --target if omitted.",
    )
    parser.add_argument(
        "--custom-target",
        action="store_true",
        help="Treat --target as a brand new word; runs g2pkk on the fly. "
        "Skips targets.json lookup. Cross-segment scan disabled.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Just run ASR + G2P, no matching",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Match against EVERY answer in EVERY segment "
        "(useful to see which target this audio is closest to)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV with columns audio_path, target_text, "
        "[target_segment_id] for batch testing",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        metavar="MS",
        help="Pre/post-pad audio with N ms of silence "
        "(may help short clips - try 300)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF model id for ASR"
    )
    parser.add_argument(
        "--targets", type=Path, default=TARGETS_PATH
    )
    parser.add_argument(
        "--matrix", type=Path, default=MATRIX_PATH
    )
    parser.add_argument(
        "--json-out", type=Path, help="Write detailed results as JSON"
    )
    args = parser.parse_args()

    # Validate combinations
    if args.manifest:
        if args.audio or args.target or args.probe or args.scan_all:
            parser.error(
                "--manifest cannot be combined with single-file options"
            )
    else:
        if not args.audio:
            parser.error(
                "Provide an audio file (or use --manifest)"
            )
        if not args.audio.exists():
            parser.error(f"Audio file not found: {args.audio}")
        if not args.probe and not args.target and not args.scan_all:
            parser.error(
                "Provide --target, --probe, or --scan-all"
            )
        if args.custom_target and args.scan_all:
            parser.error(
                "--custom-target and --scan-all are mutually exclusive"
            )

    print("Loading targets and recognizer (model: {})".format(args.model))
    tester = AudioTester(
        targets_path=args.targets,
        matrix_path=args.matrix,
        model_name=args.model,
    )

    results: list[TestResult] = []

    if args.manifest:
        rows = _read_manifest(args.manifest)
        results = run_batch(tester, rows, pad_ms=args.pad)
        _print_batch_summary(results)
    else:
        print()
        _print_header(f"TEST: {args.audio.name}")
        if args.probe:
            r = tester.probe(args.audio, pad_ms=args.pad)
            _print_test_result(r)
        else:
            r = tester.test_single(
                audio_path=args.audio,
                target_text=args.target,
                segment_id=args.segment,
                custom_target=args.custom_target,
                pad_ms=args.pad,
                scan_all=args.scan_all,
            )
            _print_test_result(r)
        results.append(r)

    # JSON dump
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": args.model,
            "matrix_path": str(args.matrix),
            "pad_ms": args.pad,
            "results": [_result_to_dict(r) for r in results],
        }
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nDetailed results saved: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
