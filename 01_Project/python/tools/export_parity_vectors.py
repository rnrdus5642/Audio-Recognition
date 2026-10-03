"""Export reference vectors for the C# port.

The tuning lives in Python but the app ships C#. If the two drift apart
the measured numbers (thresholds, skip_cost, streaming profile) stop
describing what users actually get, and the drift is silent - both sides
still return a plausible score. These vectors pin every layer of the
pipeline so the C# tests fail loudly instead.

Covers, in dependency order:
  1. phonological rules     (hangul -> surface-form hangul)
  2. hangul -> IPA
  3. confusion matrix costs
  4. score_against          (batch and streaming profiles)
  5. best_match
  6. StreamingMatcher       (streak behaviour over real ASR frames)

Usage:
    python -m python.tools.export_parity_vectors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")

from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes
from python.build.g2p.ko.rules import apply_rules
from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v1.json"
)
TARGETS_PATH = PROJECT_ROOT / "shared" / "targets.json"
FRAMES_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "streaming_frames.json"
)
# Inputs chosen to fire every entry of the rule table at least once:
# the curriculum, corpus sentences, and synthesised syllable pairs for
# the coda/onset combinations that Korean children's speech never
# happens to contain.
RULE_SAMPLES_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "rule_samples.json"
)
DEFAULT_OUT = (
    PROJECT_ROOT / "unity" / "Packages" / "com.domicube.phoneme-matching"
    / "Tests" / "Runtime" / "parity_vectors.json"
)

# Words and phrases exercising every jamo class: plain/tense/aspirated
# onsets, every vowel shape, coda clusters, and the silent onset ㅇ.
HANGUL_SAMPLES = [
    "사과", "엄마", "아빠", "토끼", "빵", "책", "우유", "학교",
    "할아버지", "먹어요", "강아지", "고양이", "바나나", "친구",
    "나비", "가요", "와요", "할머니",
    "닭", "값", "앉다", "많다", "읽다", "여덟", "핥다", "없다",
    "의사", "왼쪽", "위험", "웨딩", "예의", "얘기",
    "쌀", "짜다", "꽃", "밖", "히읗", "excuse 한글아닌것 123",
]

PHONEME_PAIRS = [
    ("s", "s"), ("s", "t"), ("s", "h"), ("k", "k͈"), ("k", "kʰ"),
    ("k̚", "k͈"), ("ɾ", "l"), ("l", "n"), ("a", "ʌ"), ("ɛ", "e"),
    ("j", "i"), ("w", "u"), ("tɕ", "tɕʰ"), ("m", "ŋ"),
    ("s", "ŋ"),        # unknown pair -> default
    ("zzz", "qqq"),    # entirely unknown symbols
]
SINGLE_PHONEMES = [
    "ŋ", "n", "m", "l", "ɾ", "k̚", "t̚", "p̚", "h", "j", "w", "ɰ",
    "a", "s", "k", "unknown",
]


def _score_case(matcher: Matcher, user: list[str], target: list[str]) -> dict:
    d, s, ws, we, _ = matcher.score_against(user, target)
    scored, dropped = matcher.context_slice(user, target)
    return {
        "user": user,
        "target": target,
        "distance": round(d, 9),
        "score": round(s, 9),
        "window_start": ws,
        "window_end": we,
        "context_kept": len(scored),
        "context_dropped": dropped,
    }


def build(matrix_path: Path, targets_path: Path, frames_path: Path) -> dict:
    matrix = ConfusionMatrix.from_json(matrix_path)
    batch = Matcher(matrix)
    stream = Matcher.for_streaming(matrix)
    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    frames = json.loads(frames_path.read_text(encoding="utf-8"))

    answers = targets["answers"]
    by_text = {a["text"]: a for a in answers}

    # --- score_against: real ASR output vs real targets, plus edge cases ---
    score_cases: list[tuple[list[str], list[str]]] = []
    seen: set[tuple] = set()
    for case in frames["cases"]:
        tgt = by_text[case["target_text"]]["phonemes"]
        for fr in case["frames"]:
            k = (tuple(fr["ipa"]), tuple(tgt))
            if k not in seen:
                seen.add(k)
                score_cases.append((fr["ipa"], tgt))
    score_cases += [
        ([], ["s", "a"]),
        (["s", "a"], []),
        ([], []),
        (["s", "a", "k", "w", "a"], ["s", "a", "k", "w", "a"]),
        (["n", "n", "s", "a", "k", "w", "a", "m"], ["s", "a", "k", "w", "a"]),
        (["a"], ["h", "a", "ɾ", "a", "p", "ʌ", "tɕ", "i"]),
    ]

    # --- best_match: the asked word, and a multi-answer question ---
    #
    # A question normally has one accepted word, so most cases carry a
    # single candidate. Several answers still have to behave, for
    # questions that take a synonym, so every case is also run against
    # the whole catalogue - the widest list the port will ever see.
    best_cases = []
    for case in frames["cases"]:
        for cands in ([by_text[case["target_text"]]], answers):
            for fr in case["frames"][:4]:
                for name, m in (("batch", batch), ("streaming", stream)):
                    r = m.best_match(fr["ipa"], cands)
                    best_cases.append({
                        "profile": name,
                        "user": fr["ipa"],
                        "candidate_ids": [c["id"] for c in cands],
                        "target_id": r.target_id,
                        "score": round(r.score, 9),
                        "passed": r.passed,
                    })

    # --- StreamingMatcher over whole sessions ---
    streaming_cases = []
    for case in frames["cases"]:
        cands = [by_text[case["target_text"]]]
        for consecutive in (1, 3):
            sm = StreamingMatcher(
                Matcher.for_streaming(matrix), cands, consecutive=consecutive
            )
            fired_at, fired_id, streaks = None, None, []
            for i, fr in enumerate(case["frames"]):
                hit = sm.push(fr["ipa"])
                streaks.append(sm.streak)
                if hit and fired_at is None:
                    fired_at, fired_id = i, hit.result.target_id
            streaming_cases.append({
                "case_id": case["id"],
                "consecutive": consecutive,
                "candidate_ids": [c["id"] for c in cands],
                "frames": [fr["ipa"] for fr in case["frames"]],
                "streaks": streaks,
                "fired_at_frame": fired_at,
                "fired_target_id": fired_id,
            })

    return {
        "_comment": (
            "python.tools.export_parity_vectors 산출물. C# 포팅이 파이썬과 "
            "같은 값을 내는지 검증하는 기준. 매처나 matrix를 고치면 다시 "
            "생성하고, C# 테스트가 깨지면 포팅 쪽을 맞출 것."
        ),
        "matrix_id": matrix.matrix_id,
        "matrix_version": matrix.version,
        "profiles": {
            "batch": {
                "skip_cost": batch.skip_cost,
                "coverage": batch.coverage,
                "context_mult": batch.context_mult,
            },
            "streaming": {
                "skip_cost": stream.skip_cost,
                "coverage": stream.coverage,
                "context_mult": stream.context_mult,
            },
        },
        "phonological_rules": [
            {"text": t, "surface": apply_rules(t)}
            for t in json.loads(
                RULE_SAMPLES_PATH.read_text(encoding="utf-8"))
        ],
        "hangul_to_ipa": [
            {"text": t, "phonemes": hangul_to_ipa_phonemes(t)}
            for t in HANGUL_SAMPLES
        ],
        "matrix_costs": {
            "substitution": [
                {"a": a, "b": b, "cost": round(matrix.sub_cost(a, b), 9)}
                for a, b in PHONEME_PAIRS
            ],
            "deletion": [
                {"phoneme": p, "cost": round(matrix.del_cost(p), 9)}
                for p in SINGLE_PHONEMES
            ],
            "insertion": [
                {"phoneme": p, "cost": round(matrix.ins_cost(p), 9)}
                for p in SINGLE_PHONEMES
            ],
        },
        "score_against": {
            "batch": [_score_case(batch, u, t) for u, t in score_cases],
            "streaming": [_score_case(stream, u, t) for u, t in score_cases],
        },
        "best_match": best_cases,
        "streaming_sessions": streaming_cases,
        "targets": [
            {"id": a["id"], "text": a["text"], "phonemes": a["phonemes"],
             "threshold": a["threshold"]}
            for a in answers
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    p.add_argument("--targets", type=Path, default=TARGETS_PATH)
    p.add_argument("--frames", type=Path, default=FRAMES_PATH)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    data = build(args.matrix, args.targets, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    n = (len(data["phonological_rules"])
         + len(data["hangul_to_ipa"])
         + sum(len(v) for v in data["matrix_costs"].values())
         + sum(len(v) for v in data["score_against"].values())
         + len(data["best_match"]) + len(data["streaming_sessions"]))
    print(f"OK: {args.output}  ({n} vectors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
