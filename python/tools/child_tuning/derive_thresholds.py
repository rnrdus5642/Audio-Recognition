"""Set each word's threshold, and say when no threshold works.

A threshold is squeezed from two sides.

Below it: unrelated speech must not confirm. That side is countable -
score the target against children saying other things and count what
gets through - and it is what this tool always did.

Above it: a good attempt has to be able to clear it. That side was never
checked, and it is where 쳐 failed. Its threshold of 0.850 was perfectly
reasonable against false accepts, but a correctly spoken 쳐 scored 0.825,
so the word could not be confirmed by anyone. A person found that a week
later. Here it is a line of output.

The upper side is estimated from what this recogniser actually mishears,
measured on single-word utterances with known transcripts: ㅃ comes out
as ㅂ 12% of the time, the glide in 놔 vanishes in 3.5%. Applying the
most likely error to a word and scoring the result says what a good
attempt is worth. If that is below the lower bound, no threshold
satisfies both and the word needs replacing rather than tuning.

Grouping by phoneme count - what this replaced - forces words with
opposite behaviour onto one value. 빵 [p͈ a ŋ] fires on unrelated speech
at 0.925 because tense and plain substitute cheaply and a, ŋ are
everywhere; 책 [tɕʰ ɛ k̚] is quiet at 0.725. Both are three phonemes,
and holding them together cost 책 34 points of detection.

Neither side needs a recording of the word itself, so a new word gets a
measured threshold at build time.

    python python/tools/child_tuning/derive_thresholds.py
    python python/tools/child_tuning/derive_thresholds.py --budget 0.003
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from python.runtime.matching.confusion_matrix import ConfusionMatrix  # noqa: E402
from python.runtime.matching.matcher import Matcher  # noqa: E402

SESSION_S = 10.0
CANDIDATES = [round(0.50 + 0.025 * i, 3) for i in range(21)]

# An error seen once in 700 utterances is not evidence of anything. Below
# this it is ignored and the hand-written matrix decides instead.
MIN_OBSERVATIONS = 2


def fires(scores, threshold, consecutive):
    streak = 0
    for score in scores:
        streak = streak + 1 if score >= threshold else 0
        if streak >= consecutive:
            return True
    return False


def lower_bound(matcher, phonemes, word, reference, seconds, budget,
                consecutive):
    """Lowest threshold whose false accepts stay inside the budget."""
    scored = [
        [matcher.score_against(f, phonemes)[1] if f else 0.0
         for f in item["frames"]]
        for item in reference if word not in item["text"]]

    for candidate in CANDIDATES:
        fired = sum(fires(s, candidate, consecutive) for s in scored)
        rate = fired / max(seconds, 1e-9) * SESSION_S
        if rate <= budget:
            return candidate, rate

    fired = sum(fires(s, CANDIDATES[-1], consecutive) for s in scored)
    return CANDIDATES[-1], fired / max(seconds, 1e-9) * SESSION_S


def likely_errors(errors):
    """Observed error rates, as {phoneme: [(kind, other, rate), ...]}."""
    out = {}
    for key, (count, total) in errors.get("substitutions", {}).items():
        if count < MIN_OBSERVATIONS or not total:
            continue
        wanted, said = key.split("|")
        out.setdefault(wanted, []).append(("sub", said, count / total))
    for wanted, (count, total) in errors.get("deletions", {}).items():
        if count < MIN_OBSERVATIONS or not total:
            continue
        out.setdefault(wanted, []).append(("del", None, count / total))
    return out


def upper_bound(matcher, phonemes, observed):
    """What a good attempt is worth once the likeliest error lands.

    The threshold has to sit at or below this, or the word only confirms
    on a flawless recognition - which is what 빵 and 쳐 do today.
    """
    best = None
    for i, phoneme in enumerate(phonemes):
        for kind, other, rate in observed.get(phoneme, []):
            variant = (phonemes[:i] + ([other] if kind == "sub" else [])
                       + phonemes[i + 1:])
            if not variant:
                continue
            score = matcher.score_against(variant, phonemes)[1]
            label = (f"{phoneme}→{other}" if kind == "sub"
                     else f"{phoneme} 탈락")
            if best is None or rate > best[0]:
                best = (rate, score, label)
    if best is None:
        return None, None, ""
    return best[1], best[0], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="shared/reference")
    ap.add_argument("--matrix",
                    default="shared/confusion_matrices/ko_child_v2.json")
    ap.add_argument("--targets", default="shared/targets.json",
                    help="IPA 를 읽어올 곳 - 임계값은 여기서 안 씁니다")
    ap.add_argument("--budget", type=float, default=0.01,
                    help="10초 세션당 오확정 상한. 기준 발화가 9시간이라 "
                         "1건이 0.031%%이고, 처음 보는 아이에서 0.40%%로 "
                         "재현됩니다 - 600건이던 시절의 안전 마진은 "
                         "필요 없습니다")
    ap.add_argument("--consecutive", type=int, default=2,
                    help="확정에 필요한 연속 프레임. 앱과 같아야 합니다")
    ap.add_argument("--out", default="shared/thresholds_child.json")
    args = ap.parse_args()

    matrix = ConfusionMatrix.from_json(args.matrix)
    profile = matrix.streaming_profile
    matcher = Matcher(matrix,
                      skip_cost=profile["skip_cost"],
                      coverage=profile["coverage"],
                      context_mult=profile["context_mult"])

    frames_path = os.path.join(args.reference, "reference_frames.json.gz")
    errors_path = os.path.join(args.reference, "reference_errors.json")
    if not os.path.exists(frames_path):
        raise SystemExit(
            f"기준 데이터가 없습니다: {frames_path}\n"
            "python python/tools/child_tuning/build_reference.py 를 "
            "먼저 돌리세요.")
    with gzip.open(frames_path, "rt", encoding="utf-8") as handle:
        reference = json.load(handle)
    errors = (json.load(open(errors_path, encoding="utf-8"))
              if os.path.exists(errors_path) else {})
    observed = likely_errors(errors)

    if reference.get("streaming_profile") != dict(profile):
        print("경고: 기준 데이터가 다른 설정에서 만들어졌습니다.",
              file=sys.stderr)
        print(f"  기준 {reference.get('streaming_profile')}", file=sys.stderr)
        print(f"  지금 {dict(profile)}", file=sys.stderr)

    items = reference["items"]
    seconds = sum(i["seconds"] for i in items)
    answers = json.load(open(args.targets, encoding="utf-8"))["answers"]

    print(f"기준 발화 {len(items):,}건 · {seconds/3600:.1f}시간 · "
          f"해상도 {1/seconds*SESSION_S*100:.3f}%")
    print(f"오류 분포 {len(observed)}음소 "
          f"({errors.get('utterances', 0):,}발화에서 측정)")
    print(f"예산 {args.budget*100:.2f}% · 연속 {args.consecutive}회 · "
          f"matrix {matrix.matrix_id}\n")
    print(f"{'단어':10}{'음소':>4}{'하한':>8}{'상한':>8}{'창':>8}"
          f"  {'가장 흔한 오류':16}판정")

    thresholds = {}
    warnings = []
    for answer in answers:
        word = answer["text"]
        phonemes = answer["phonemes"]
        low, rate = lower_bound(matcher, phonemes, word, items, seconds,
                                args.budget, args.consecutive)
        high, error_rate, label = upper_bound(matcher, phonemes, observed)
        thresholds[word] = low

        if high is None:
            gap, verdict = None, "오류 미관측"
        elif low > high:
            gap, verdict = high - low, "완벽한 발음만 통과"
            warnings.append((word, low, high))
        elif high - low >= 0.15:
            gap, verdict = high - low, "넉넉"
        elif high - low >= 0.08:
            gap, verdict = high - low, "보통"
        else:
            gap, verdict = high - low, "빠듯"

        shown = f"{label} {error_rate*100:.0f}%" if label else "-"
        print(f"{word:10}{len(phonemes):>4}{low:>8.3f}"
              f"{(f'{high:.3f}' if high is not None else '-'):>8}"
              f"{(f'{gap:+.3f}' if gap is not None else '-'):>8}"
              f"  {shown:16}{verdict}")

    payload = {
        "_comment": ("python/tools/child_tuning/derive_thresholds.py 산출물. "
                     "단어별 임계값 - build_targets.py --thresholds 로 씁니다."),
        "matrix_id": matrix.matrix_id,
        "reference": {"utterances": len(items), "seconds": round(seconds, 1)},
        "budget": args.budget,
        "consecutive": args.consecutive,
        "streaming_profile": dict(profile),
        "thresholds": thresholds,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\n{len(thresholds)}단어 -> {args.out}")

    if warnings:
        print("\n다음 단어는 어떤 임계값으로도 해결되지 않습니다. "
              "바꾸는 것 말고 방법이 없습니다:")
        for word, low, high in warnings:
            print(f"   {word}  오확정을 막으려면 {low:.3f} 이상이어야 하는데, "
                  f"흔한 오류 하나가 나면 {high:.3f} 까지만 나옵니다")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
