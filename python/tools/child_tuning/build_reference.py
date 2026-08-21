"""Pack what a threshold needs to know into two small files.

A word's threshold has two sides, and both used to need something this
repository did not have.

The lower side is false accepts: the lowest threshold at which unrelated
speech stops confirming. That is countable rather than predictable - run
the target against children saying other things and count what gets
through - but the recordings lived only on one machine, so adding a word
fell back to a phoneme-count guess that lands within one step 19% of the
time. The counting needs no audio, only the phoneme sequences the
recogniser produced, which is 34 KB per 600 utterances.

The upper side is whether a correct pronunciation can clear it at all.
쳐 scored 0.825 against a threshold of 0.850 and could never confirm; it
took a week and a person to notice. Estimating it means knowing what this
recogniser actually mishears, and that is measurable from any child
speech with a known transcript - not from recordings of the word itself.
On 697 single-word utterances the answer is concrete: 빵's tense ㅃ is
heard as plain ㅂ 12% of the time, and 놔's glide disappears in 3.5%.
Those are exactly the two words that fail.

So this writes:

    reference_frames.json   what the recogniser heard on speech that is
                            not the target - for counting false accepts
    reference_errors.json   how often each phoneme comes out as another
                            - for estimating what a good attempt scores

Both are model-specific: refit the acoustic model and they are stale.

    python python/tools/child_tuning/build_reference.py
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sys.path.insert(0, HERE)

from python.build.g2p.ko import rules  # noqa: E402
from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes  # noqa: E402
from python.runtime.matching.confusion_matrix import ConfusionMatrix  # noqa: E402
from python.runtime.matching.matcher import Matcher  # noqa: E402

from frames import load  # noqa: E402

# An alignment is only evidence if the recogniser was in the right
# neighbourhood. Below this the two sequences share nothing and the
# operations it reports are noise.
MIN_ALIGN_SCORE = 0.4


def reference_frames(tag, split):
    """Per-window phonemes for speech that is not a curriculum word."""
    out = []
    for item in load(split, tag):
        if item["positive"]:
            continue
        out.append({
            "seconds": round(float(item["seconds"]), 3),
            "text": item["text"],
            "frames": item["frames"],
        })
    return out


def error_rates(matcher, tag):
    """How often each target phoneme comes out as something else.

    Measured on single-word utterances, where the whole recording is one
    known word and the alignment is unambiguous. Windows are scored
    against the truth and the best one per utterance is counted.
    """
    subs = collections.Counter()
    dels = collections.Counter()
    total = collections.Counter()
    used = seen = 0

    for split in ("train", "valid", "test"):
        for item in load(split, tag):
            if not item["positive"]:
                continue
            seen += 1
            target = hangul_to_ipa_phonemes(
                rules.apply_rules(item["text"].replace(".", "").strip()))
            if not target:
                continue

            best = None
            for frame in item["frames"]:
                if not frame:
                    continue
                _d, score, _ws, _we, ops = matcher.score_against(frame, target)
                if best is None or score > best[0]:
                    best = (score, ops)
            if best is None or best[0] < MIN_ALIGN_SCORE:
                continue

            used += 1
            for said, wanted, op in best[1]:
                if op in ("match", "sub"):
                    total[wanted] += 1
                    if op == "sub":
                        subs[(wanted, said)] += 1
                elif op == "del":
                    total[wanted] += 1
                    dels[wanted] += 1

    return subs, dels, total, used, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-tag", default="ref",
                    help="오확정 기준으로 쓸 캐시")
    ap.add_argument("--frames-split", default="train")
    ap.add_argument("--errors-tag", default="child-iso",
                    help="오류 분포를 잴 캐시 - 단독 단어여야 정렬이 명확")
    ap.add_argument("--matrix",
                    default="shared/confusion_matrices/ko_child_v2.json")
    ap.add_argument("--out-dir", default="shared/reference")
    args = ap.parse_args()

    matrix = ConfusionMatrix.from_json(args.matrix)
    profile = matrix.streaming_profile
    matcher = Matcher(matrix,
                      skip_cost=profile["skip_cost"],
                      coverage=profile["coverage"],
                      context_mult=profile["context_mult"])

    os.makedirs(args.out_dir, exist_ok=True)

    frames = reference_frames(args.frames_tag, args.frames_split)
    if not frames:
        raise SystemExit(f"'{args.frames_tag}' 캐시가 비어 있습니다.")
    seconds = sum(f["seconds"] for f in frames)
    payload = {
        "_comment": ("python/tools/child_tuning/build_reference.py 산출물. "
                     "아이들이 커리큘럼 단어가 아닌 말을 한 발화를 인식한 "
                     "결과. 임계값의 하한(오확정)을 여기서 셉니다."),
        "matrix_id": matrix.matrix_id,
        "streaming_profile": dict(profile),
        "utterances": len(frames),
        "seconds": round(seconds, 1),
        "items": frames,
    }
    # Phoneme strings repeat heavily, so this compresses about twelve
    # times over. 300 KB belongs in a repository; 3.8 MB is a nuisance.
    path = os.path.join(args.out_dir, "reference_frames.json.gz")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    print(f"기준 음소열  {len(frames):,}발화 · {seconds/3600:.1f}시간 · "
          f"{os.path.getsize(path)/1024:.0f} KB")
    print(f"   해상도    1건 = {1/seconds*10*100:.3f}%")

    subs, dels, total, used, seen = error_rates(matcher, args.errors_tag)
    rates = {
        "substitutions": {f"{a}|{b}": [n, total[a]] for (a, b), n in subs.items()},
        "deletions": {a: [n, total[a]] for a, n in dels.items()},
    }
    payload = {
        "_comment": ("이 인식기가 실제로 무엇을 무엇으로 틀리는지. "
                     "[관측횟수, 그 음소의 전체 등장횟수]. 임계값의 "
                     "상한(정확히 말해도 넘을 수 있나)을 여기서 추정합니다."),
        "matrix_id": matrix.matrix_id,
        "utterances": used,
        "of": seen,
        **rates,
    }
    path = os.path.join(args.out_dir, "reference_errors.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"오류 분포    {used:,}/{seen:,}발화 정렬 · "
          f"치환 {len(subs)}종 · 탈락 {len(dels)}종 · "
          f"{os.path.getsize(path)/1024:.0f} KB")
    print()
    print("가장 흔한 오류:")
    for (a, b), n in subs.most_common(6):
        print(f"   {a} → {b:<6} {n:3}회  {n/max(total[a],1)*100:5.1f}%")
    for a, n in dels.most_common(3):
        print(f"   {a} 탈락{'':<4} {n:3}회  {n/max(total[a],1)*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
