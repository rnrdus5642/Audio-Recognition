"""Fit the streaming profile against the condition the product uses.

The game shows a word and the child says that word, alone. Everything
measured before this fitted on the corpus instead, where the same word
sits inside a sentence, and the two score differently: phonemes outside
the match window cost `skip_cost`, and an isolated word has almost none
outside while a sentence has many. Dropping skip_cost tenfold was worth
11 points on sentences and nothing on single words - it was paying to
solve a problem the product does not have.

So detections come from a cache of single-word utterances and false
accepts from a cache of continuous speech, which is what a child
actually produces when not answering. Thresholds are per word rather
than per phoneme count, for the reason derive_thresholds explains.

Selection runs on train speakers only; valid and test are printed so a
profile that suits the tuning children is visible rather than inferred.

    python python/tools/child_tuning/fit_streaming.py
    python python/tools/child_tuning/fit_streaming.py --budget 0.003
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

SKIP = [0.005, 0.02, 0.05, 0.08]
COVERAGE = [0.8, 0.9]
CONTEXT = [4.0, 6.0, 8.0]
CONSECUTIVE = [1, 2]


def fires(scores, threshold, consecutive):
    streak = 0
    for score in scores:
        streak = streak + 1 if score >= threshold else 0
        if streak >= consecutive:
            return True
    return False


def score_all(matcher, items, answers, positives):
    """Frame scores per word, for the utterances of one kind."""
    out = {a["text"]: [] for a in answers}
    for item in items:
        if positives:
            for word in item["hits"]:
                answer = next(a for a in answers if a["text"] == word)
                out[word].append(
                    [matcher.score_against(f, answer["phonemes"])[1] if f
                     else 0.0 for f in item["frames"]])
        else:
            for answer in answers:
                if answer["text"] in item["text"]:
                    continue
                out[answer["text"]].append(
                    [matcher.score_against(f, answer["phonemes"])[1] if f
                     else 0.0 for f in item["frames"]])
    return out


def thresholds_for(negatives, seconds, answers, budget, consecutive):
    """Lowest threshold per word that keeps false accepts inside budget."""
    out = {}
    for answer in answers:
        word = answer["text"]
        out[word] = CANDIDATES[-1]
        for candidate in CANDIDATES:
            fired = sum(fires(s, candidate, consecutive)
                        for s in negatives[word])
            if fired / max(seconds, 1e-9) * SESSION_S <= budget:
                out[word] = candidate
                break
    return out


def evaluate(positives, negatives, seconds, answers, thresholds, consecutive):
    hit = said = 0
    for answer in answers:
        word = answer["text"]
        for seq in positives[word]:
            hit += fires(seq, thresholds[word], consecutive)
            said += 1
    rate = sum(
        sum(fires(s, thresholds[a["text"]], consecutive)
            for s in negatives[a["text"]]) / max(seconds, 1e-9) * SESSION_S
        for a in answers) / len(answers)
    return hit / max(said, 1), rate, hit, said


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positives", default="child-iso",
                    help="단독 단어 캐시")
    ap.add_argument("--negatives", default="child",
                    help="연속 발화 캐시 - 오확정은 여기서 납니다")
    ap.add_argument("--matrix",
                    default="shared/confusion_matrices/ko_child_v1.json",
                    help="치환 비용만 씁니다. 프로필은 여기서 찾습니다")
    ap.add_argument("--targets", default="shared/targets.json")
    ap.add_argument("--budget", type=float, default=0.005,
                    help="train 내 오확정 상한. held-out 은 약 2배가 됩니다")
    ap.add_argument("--consecutive", type=int, default=None,
                    help="고정할 연속 확인 횟수. 생략하면 1·2 둘 다 탐색")
    ap.add_argument("--out", default="shared/thresholds_child.json")
    args = ap.parse_args()
    consecutives = [args.consecutive] if args.consecutive else CONSECUTIVE

    base = ConfusionMatrix.from_json(args.matrix)
    answers = json.load(open(args.targets, encoding="utf-8"))["answers"]

    # A third set: the same words spoken inside a sentence. A child
    # answering a prompt often adds something - 사과요, 음 사과 - and this
    # is the column that separates profiles the isolated set cannot.
    # skip_cost 0.005 and 0.08 both score 88.1% on single words; with
    # anything around them it is 89.8% against 80.2%.
    data = {}
    for split in ("train", "valid", "test"):
        pos = [i for i in load(split, args.positives) if i["positive"]]
        other = load(split, args.negatives)
        neg = [i for i in other if not i["positive"]]
        surround = [i for i in other if i["positive"]]
        data[split] = (pos, neg, sum(i["seconds"] for i in neg), surround)
        print(f"{split:6} 단독 {len(pos):4}건 · 연속 {len(neg):4}건 · "
              f"덧붙임 {len(surround):4}건", flush=True)

    print(f"\n{'skip':>7}{'cov':>6}{'ctx':>6}{'연속':>5}"
          f"{'train 검출':>11}{'덧붙임':>9}{'train 오확정':>13}", flush=True)
    best = None
    for skip in SKIP:
        for coverage in COVERAGE:
            for context in CONTEXT:
                matcher = Matcher(base, skip_cost=skip, coverage=coverage,
                                  context_mult=context)
                scored = {}
                for split, (pos, neg, secs, surround) in data.items():
                    scored[split] = (score_all(matcher, pos, answers, True),
                                     score_all(matcher, neg, answers, False),
                                     secs,
                                     score_all(matcher, surround, answers,
                                               True))
                for consecutive in consecutives:
                    p, n, secs, sur = scored["train"]
                    th = thresholds_for(n, secs, answers, args.budget,
                                        consecutive)
                    det, rate, _, _ = evaluate(p, n, secs, answers, th,
                                               consecutive)
                    add, _, _, _ = evaluate(sur, n, secs, answers, th,
                                            consecutive)
                    print(f"{skip:>7}{coverage:>6}{context:>6}{consecutive:>5}"
                          f"{det*100:10.1f}%{add*100:8.1f}%"
                          f"{rate*100:12.2f}%", flush=True)
                    if best is None or det > best["detection"]:
                        best = {"skip_cost": skip, "coverage": coverage,
                                "context_mult": context,
                                "consecutive": consecutive,
                                "detection": det, "thresholds": th,
                                "scored": scored}

    print(f"\ntrain 최적  skip {best['skip_cost']} · 커버리지 "
          f"{best['coverage']} · 문맥 {best['context_mult']}배 · "
          f"연속 {best['consecutive']}회")
    for split in ("train", "valid", "test"):
        p, n, secs, _ = best["scored"][split]
        det, rate, hit, said = evaluate(p, n, secs, answers,
                                        best["thresholds"],
                                        best["consecutive"])
        print(f"  {split:6} 검출 {det*100:5.1f}% ({hit}/{said})  "
              f"오확정 {rate*100:5.2f}%")

    payload = {
        "_comment": ("python/tools/child_tuning/fit_streaming.py 산출물. "
                     "단독 단어 조건에서 맞춘 프로필과 단어별 임계값."),
        "matrix_id": base.matrix_id,
        "positives": args.positives,
        "negatives": args.negatives,
        "budget": args.budget,
        "consecutive": best["consecutive"],
        "streaming_profile": {
            "skip_cost": best["skip_cost"],
            "coverage": best["coverage"],
            "context_mult": best["context_mult"],
        },
        "thresholds": best["thresholds"],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
