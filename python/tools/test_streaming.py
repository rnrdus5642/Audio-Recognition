"""Replay a WAV through the continuous-listening pipeline.

Simulates what the VR app will do: re-recognise the trailing audio
window every hop, score it against the candidate answers, and stop as
soon as one answer wins `--consecutive` frames in a row.

Usage:
    python -m python.tools.test_streaming my.wav --target 사과
    python -m python.tools.test_streaming my.wav --segment lesson_03_food
    python -m python.tools.test_streaming my.wav --target 사과 --verbose
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

from python.runtime.audio import load_audio_16k_mono, rolling_windows
from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher
from python.runtime.recognizer import get_recognizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_PATH = PROJECT_ROOT / "shared" / "targets.json"
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v1.json"
)


def pick_candidates(targets: dict, target: str | None, segment: str | None):
    """Answers to listen for: one word, one segment, or everything."""
    segs = targets["segments"]
    if segment:
        for s in segs:
            if s["id"] == segment:
                return s["answers"]
        raise SystemExit(f"segment not found: {segment}")
    if target:
        for s in segs:
            for a in s["answers"]:
                if target in (a["text"], a["id"]):
                    # The whole segment competes, as it will in the app.
                    return s["answers"]
        raise SystemExit(f"target not found in targets.json: {target}")
    return [a for s in segs for a in s["answers"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("audio", type=Path)
    p.add_argument("--target", help="정답 단어 (해당 세그먼트 전체가 후보)")
    p.add_argument("--segment", help="세그먼트 id 전체를 후보로")
    p.add_argument("--window", type=float, default=2.5,
                   help="ASR에 넣는 오디오 창 (초)")
    p.add_argument("--hop", type=float, default=0.4,
                   help="채점 주기 (초)")
    p.add_argument("--consecutive", type=int, default=3,
                   help="확정에 필요한 연속 프레임 수")
    p.add_argument("--verbose", action="store_true",
                   help="프레임마다 인식 결과와 점수 출력")
    args = p.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"audio not found: {args.audio}")

    targets = json.loads(TARGETS_PATH.read_text(encoding="utf-8"))
    matrix = ConfusionMatrix.from_json(MATRIX_PATH)
    cands = pick_candidates(targets, args.target, args.segment)

    matcher = Matcher.for_streaming(matrix)
    sm = StreamingMatcher(matcher, cands, consecutive=args.consecutive)
    rec = get_recognizer("ko")
    audio = load_audio_16k_mono(args.audio)

    print(f"오디오 {len(audio) / 16000:.2f}초 | 후보 "
          f"{', '.join(a['text'] for a in cands)}")
    print(f"창 {args.window}s / hop {args.hop}s / 연속 {args.consecutive}회 "
          f"| skip {matcher.skip_cost} coverage {matcher.coverage} "
          f"context x{matcher.context_mult}")
    print("-" * 68)

    hit = None
    for t, window in rolling_windows(audio, args.window, args.hop):
        ipa = rec.recognize(window)
        hit = sm.push(ipa)
        if args.verbose or hit:
            best = matcher.best_match(ipa, cands)
            print(f"  t={t:5.1f}s  streak={sm.streak}  "
                  f"{best.target_text or '-':6s} {best.score:.3f}  "
                  f"{' '.join(ipa) if ipa else '(무음)'}")
        if hit:
            break

    print("-" * 68)
    if hit:
        print(f"확정: {hit.result.target_text}  "
              f"(점수 {hit.result.score:.3f}, {hit.frames}프레임째)")
        return 0
    if sm.streak:
        print(f"확정 없음 — 마지막까지 연속 {sm.streak}회에서 끊겼습니다 "
              f"(확정에 {args.consecutive}회 필요).")
        print("  정답 직후 오디오가 끝나면 연속 확인이 완성되지 않습니다. "
              f"발화 후 최소 {args.consecutive * args.hop:.1f}초는 더 들어야 합니다.")
    else:
        print("확정 없음 — 어느 프레임에서도 임계값을 넘지 못했습니다.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
