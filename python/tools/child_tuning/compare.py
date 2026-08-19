"""Confirmation rate for one cache, at the settings that ship.

Character error says how much of a transcript is right. The product
asks something narrower: did the child say the word, and did we say so.
The two move together but not one-for-one, and every decision left -
which checkpoint to keep, whether more training is worth the hours -
turns on the second rather than the first.

So the parameters are held at the deployed values and only the model
changes. Anything that differs between two runs is the acoustic model,
not the tuning; retuning per model comes after, once it is known which
model is worth tuning for.

    python python/tools/child_tuning/compare.py base child epoch-01
    python python/tools/child_tuning/compare.py child --split test
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from python.runtime.matching.confusion_matrix import ConfusionMatrix  # noqa: E402
from python.runtime.matching.matcher import Matcher  # noqa: E402

from frames import load  # noqa: E402

# A confirmation needs this many windows in a row over threshold, and a
# false accept is counted per ten seconds of speech - the length of one
# turn in the game.
CONSECUTIVE = 2
SESSION_S = 10.0


def fires(sequence, threshold):
    """Whether a run of CONSECUTIVE windows ever clears the threshold."""
    streak = 0
    for score in sequence:
        streak = streak + 1 if score >= threshold else 0
        if streak >= CONSECUTIVE:
            return True
    return False


def measure(items, matcher, by_text, targets):
    """Detection over spoken words, false accepts over unspoken ones."""
    hit = said = 0
    per_word = {}
    for item in items:
        if not item["positive"]:
            continue
        for word in item["hits"]:
            answer = by_text[word]
            seq = [matcher.score_against(f, answer["phonemes"])[1] if f else 0.0
                   for f in item["frames"]]
            got = fires(seq, answer["threshold"])
            hit += got
            said += 1
            tally = per_word.setdefault(word, [0, 0])
            tally[0] += got
            tally[1] += 1

    trips = 0.0
    seconds = 0.0
    for item in items:
        if item["positive"]:
            continue
        seconds += item["seconds"]
        for answer in targets:
            if answer["text"] in item["text"]:
                continue
            seq = [matcher.score_against(f, answer["phonemes"])[1] if f else 0.0
                   for f in item["frames"]]
            trips += fires(seq, answer["threshold"])

    if not seconds:
        # An --isolated cache holds detections only. Reporting 0.00% here
        # would read as "never false-fires" rather than "not measured".
        return hit, said, None, per_word

    # Every utterance is scored against every answer, so the rate is per
    # word-second; one session listens for one word at a time.
    words = max(1, len(targets))
    false_accept = trips / max(seconds * words, 1e-9) * SESSION_S
    return hit, said, false_accept, per_word


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+", help="비교할 캐시 폴더 이름")
    ap.add_argument("--split", action="append",
                    help="기본: valid test (두 모델 모두 가진 것)")
    ap.add_argument("--words", action="store_true",
                    help="단어별 검출률도 출력")
    ap.add_argument("--matrix",
                    default="shared/confusion_matrices/ko_child_v1.json")
    ap.add_argument("--targets", default="shared/targets.json")
    args = ap.parse_args()
    splits = args.split or ["valid", "test"]

    matrix = ConfusionMatrix.from_json(args.matrix)
    profile = matrix.streaming_profile
    matcher = Matcher(matrix,
                      skip_cost=profile["skip_cost"],
                      coverage=profile["coverage"],
                      context_mult=profile["context_mult"])
    targets = json.load(open(args.targets, encoding="utf-8"))["answers"]
    by_text = {t["text"]: t for t in targets}

    print(f"설정  skip {profile['skip_cost']} · 커버리지 "
          f"{profile['coverage']} · 문맥 {profile['context_mult']}배 · "
          f"연속 {CONSECUTIVE}회 · matrix {matrix.matrix_id}")
    print(f"분할  {' '.join(splits)}\n")
    print(f"{'캐시':12} {'검출':>16} {'오확정':>10}   {'화자':>5}")

    results = {}
    for tag in args.tags:
        items = [i for s in splits for i in load(s, tag)]
        if not items:
            print(f"{tag:12}  캐시가 없습니다")
            continue
        hit, said, fa, per_word = measure(items, matcher, by_text, targets)
        speakers = len({i["speaker"] for i in items})
        rate = hit / max(said, 1) * 100
        shown = f"{fa*100:8.2f}%" if fa is not None else "  측정 안 함"
        print(f"{tag:12} {rate:6.1f}% ({hit:4}/{said:4}) "
              f"{shown:>10}   {speakers:5}")
        results[tag] = per_word

    if args.words and results:
        print(f"\n{'단어':10}" + "".join(f"{t:>12}" for t in results))
        words = sorted({w for r in results.values() for w in r})
        for word in words:
            row = f"{word:10}"
            for tag in results:
                got, n = results[tag].get(word, (0, 0))
                row += f"{got/n*100:9.0f}% ({n})" if n else f"{'-':>12}"
            print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
