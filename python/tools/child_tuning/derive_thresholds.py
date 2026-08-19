"""Set each word's threshold from what its IPA collides with.

A threshold exists to stop unrelated speech from confirming a word, and
how much unrelated speech a word attracts is a property of its phoneme
sequence - not of how many phonemes it has. Grouping by length forces
words with opposite behaviour onto one value:

    빵  [p͈ a ŋ]     tense/plain substitution is cheap and a, ŋ are
                     everywhere in Korean, so it fires on 3.4% of
                     sessions even at 0.925
    책  [tɕʰ ɛ k̚]   ɛ and a coda k̚ are rarer; 0.725 already keeps it
                     under a tenth of that

Both are three phonemes. Holding them at one value spent 34 points of
책's detection buying nothing, because 빵 needed the strictness and 책
never did.

So the threshold is measured per word instead: score the target against
a cache of children saying other things, and take the lowest threshold
whose false-accept rate stays inside the budget. Detection is then as
high as the budget allows, which is the whole objective.

Nothing here needs a recording of the word itself. That matters more
than the accuracy: only 10 of 29 curriculum words have enough spoken
examples to fit a threshold from, while this works for all 29 - and for
whatever word gets added next, including 실로폰, which appears nowhere
in 2.76M utterances.

The budget is spent against train speakers and lands higher on unseen
ones, so --budget sits below the real target: 0.5% here measured 0.70%
over 234 held-out children at the shipped profile.

The profile it reads from the matrix has to be the one that ships, and
so does --consecutive: a threshold is the lowest value that stays quiet
for that many frames in a row, and the answer changes if the app asks
for a different number. `fit_streaming.py` searches for the profile;
this sets thresholds once the profile is settled.

    python python/tools/child_tuning/derive_thresholds.py
    python python/tools/child_tuning/derive_thresholds.py --budget 0.003
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from python.runtime.matching.confusion_matrix import ConfusionMatrix  # noqa: E402
from python.runtime.matching.matcher import Matcher  # noqa: E402

from frames import load  # noqa: E402

SESSION_S = 10.0
CANDIDATES = [round(0.50 + 0.025 * i, 3) for i in range(21)]


def fires(scores, threshold, consecutive):
    """Whether the score clears the threshold for long enough to confirm."""
    streak = 0
    for score in scores:
        streak = streak + 1 if score >= threshold else 0
        if streak >= consecutive:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="child", help="어느 모델의 프레임 캐시")
    ap.add_argument("--split", default="train", help="임계값을 맞출 분할")
    ap.add_argument("--matrix",
                    default="shared/confusion_matrices/ko_child_v2.json")
    ap.add_argument("--targets", default="shared/targets.json",
                    help="IPA 를 읽어올 곳 - 임계값은 여기서 안 씁니다")
    ap.add_argument("--budget", type=float, default=0.005,
                    help="분할 내 오확정 상한. held-out 은 더 높게 나옵니다")
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--out", default="shared/thresholds_child.json")
    args = ap.parse_args()

    matrix = ConfusionMatrix.from_json(args.matrix)
    profile = matrix.streaming_profile
    matcher = Matcher(matrix,
                      skip_cost=profile["skip_cost"],
                      coverage=profile["coverage"],
                      context_mult=profile["context_mult"])
    answers = json.load(open(args.targets, encoding="utf-8"))["answers"]

    items = load(args.split, args.tag)
    if not items:
        raise SystemExit(f"'{args.tag}' 캐시에 {args.split} 이 없습니다.")
    negatives = [i for i in items if not i["positive"]]
    seconds = sum(i["seconds"] for i in negatives)

    print(f"캐시 {args.tag}/{args.split} · 부정 발화 {len(negatives):,}건 "
          f"({seconds/3600:.1f}시간) · 예산 {args.budget*100:.2f}%")
    print(f"matrix {matrix.matrix_id} · skip {profile['skip_cost']} · "
          f"커버리지 {profile['coverage']} · 문맥 {profile['context_mult']}배 · "
          f"연속 {args.consecutive}회\n")
    print(f"{'단어':10}{'음소':>4}{'임계값':>8}{'오확정':>9}")

    thresholds = {}
    for answer in answers:
        word = answer["text"]
        # Utterances containing the word are excluded: they would be
        # detections, not false accepts.
        scored = [
            [matcher.score_against(f, answer["phonemes"])[1] if f else 0.0
             for f in item["frames"]]
            for item in negatives if word not in item["text"]]

        chosen, rate = CANDIDATES[-1], None
        for candidate in CANDIDATES:
            fired = sum(fires(s, candidate, args.consecutive) for s in scored)
            hit_rate = fired / max(seconds, 1e-9) * SESSION_S
            if hit_rate <= args.budget:
                chosen, rate = candidate, hit_rate
                break
        if rate is None:
            # Nothing in range is quiet enough; the strictest value is the
            # best available and the word is worth reconsidering.
            fired = sum(fires(s, chosen, args.consecutive) for s in scored)
            rate = fired / max(seconds, 1e-9) * SESSION_S

        thresholds[word] = chosen
        mark = "  예산 초과" if rate > args.budget else ""
        print(f"{word:12}{len(answer['phonemes']):>4}{chosen:>8}"
              f"{rate*100:8.2f}%{mark}")

    payload = {
        "_comment": ("python/tools/child_tuning/derive_thresholds.py 산출물. "
                     "단어별 임계값 - build_targets.py --thresholds 로 씁니다."),
        "matrix_id": matrix.matrix_id,
        "cache": f"{args.tag}/{args.split}",
        "budget": args.budget,
        "consecutive": args.consecutive,
        "streaming_profile": dict(profile),
        "thresholds": thresholds,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\n{len(thresholds)}단어 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
