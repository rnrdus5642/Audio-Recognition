"""Estimate the confusion matrix from what children actually said.

The costs in ko_child_v1 were written by hand from the developmental
literature and adjusted against synthetic speech. This replaces them with
counts: how often does a child produce phoneme B when the word calls for
A? Common substitutions become cheap, unseen ones expensive.

Three things this has to get right, learned the hard way from a first
attempt that ran away:

  The alignment filter must not move. Judging alignments with the matrix
  being learned is a feedback loop - cheaper costs admit worse
  alignments, which make things cheaper still. The first run went from
  554 to 872 accepted alignments over four iterations and never settled.
  The filter here uses the hand matrix and is computed once.

  Rare pairs must not be trusted. tɕ→tɕʰ was seen twice and that was
  enough to overwrite a hand-set value. Pairs need MIN_PAIR observations
  before they are used at all.

  Everything else shrinks toward the hand value in proportion to how
  little was seen, so a pair observed six times barely moves and one
  observed sixty moves most of the way.

Learning only from utterances that passed would reinforce what already
works, so every utterance whose transcript contains the word is aligned,
pass or fail.
"""

import json
import math
import os
from collections import Counter, defaultdict

from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import Matcher

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ko_child_v2.json")

MIN_PAIR = 5          # observations of A→B before the pair is used
MIN_SOURCE = 30       # observations of A before its row is used
MAX_DIST_RATIO = 0.6  # alignment quality cutoff, judged by the hand matrix
P_FLOOR = 0.01        # probability that maps to full cost
SHRINK_K = 20.0       # pseudo-count pulling toward the hand value

SETS = json.load(open(os.path.join(BASE, "tune_frames.json"),
                      encoding="utf-8"))["sets"]
targets = json.load(open("shared/targets.json", encoding="utf-8"))["answers"]
by_text = {t["text"]: t for t in targets}
hand = ConfusionMatrix.from_json("shared/confusion_matrices/ko_child_v1.json")

# One fixed pass. The filter and the alignments both come from the hand
# matrix, so nothing here depends on what is being estimated.
m = Matcher(hand, coverage=0.5)
subs = defaultdict(Counter)
dels = Counter()
source_total = Counter()
used = dropped = 0

for item in SETS["tune_pos"]:
    for w in item["hits"]:
        target = by_text[w]["phonemes"]
        best = None
        for frame in item["frames"]:
            if not frame:
                continue
            dist, score, ws, we, ops = m.score_against(frame, target)
            if best is None or dist < best[0]:
                best = (dist, ops)
        if best is None:
            continue
        dist, ops = best
        if dist / max(len(target), 1) > MAX_DIST_RATIO:
            dropped += 1
            continue
        used += 1
        for user_p, target_p, op in ops:
            if op in ("match", "sub"):
                subs[target_p][user_p if op == "sub" else target_p] += 1
                source_total[target_p] += 1
            elif op == "del":
                dels[target_p] += 1
                source_total[target_p] += 1

print(f"정렬 {used}건 사용 / {dropped}건 버림", flush=True)

substitutions = {}
kept = skipped_rare = skipped_thin = 0
for a, counter in subs.items():
    total = source_total[a]
    if total < MIN_SOURCE:
        skipped_thin += sum(1 for b in counter if b != a)
        continue
    for b, n in counter.items():
        if a == b:
            continue
        if n < MIN_PAIR:
            skipped_rare += 1
            continue
        p = (n + 0.5) / (total + 1.0)
        learned = max(0.05, min(1.0, math.log(p) / math.log(P_FLOOR)))
        prior = hand.sub_cost(a, b)
        cost = (n * learned + SHRINK_K * prior) / (n + SHRINK_K)
        substitutions[(a, b)] = round(cost, 3)
        kept += 1

deletions = {}
for a, n in dels.items():
    total = source_total[a]
    if total < MIN_SOURCE or n < MIN_PAIR:
        continue
    p = (n + 0.5) / (total + 1.0)
    learned = max(0.05, min(1.0, math.log(p) / math.log(P_FLOOR)))
    prior = hand.del_cost(a)
    deletions[a] = round((n * learned + SHRINK_K * prior) / (n + SHRINK_K), 3)

print(f"치환 {kept}쌍 학습 · 관측 부족으로 제외 {skipped_rare}쌍 · "
      f"음소 자체가 드물어 제외 {skipped_thin}쌍", flush=True)
print(f"삭제 {len(deletions)}개 학습", flush=True)

merged = {f"{a}|{b}": c for (a, b), c in substitutions.items()}
for (a, b), c in hand.known_substitutions.items() if hasattr(
        hand, "known_substitutions") and isinstance(
        hand.known_substitutions, dict) else []:
    merged.setdefault(f"{a}|{b}", c)

payload = {
    "language": "ko",
    "version": "2.0.0",
    "phoneme_set": "ipa",
    "description": (
        "ko_child_v1 을 AI Hub 아동 자유 발화(6-7세, 화자 15명)의 정렬 "
        "빈도로 보정. cost = log P(관측|정답) / log 0.01 을 관측 수에 따라 "
        f"기존값 쪽으로 수축(k={SHRINK_K}). 쌍 {MIN_PAIR}회·음소 "
        f"{MIN_SOURCE}회 미만은 기존값 유지."),
    "default_substitution": hand.default_substitution,
    "default_insertion": hand.default_insertion,
    "default_deletion": hand.default_deletion,
    "skip_cost": hand.skip_cost,
    "streaming_profile": dict(hand.streaming_profile),
    "substitutions": merged,
    "deletions": deletions,
    "insertions": {},
}
json.dump(payload, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n{OUT}", flush=True)

print("\n=== 크게 바뀐 것 (관측 수 순) ===", flush=True)
rows = []
for (a, b), cost in substitutions.items():
    old = hand.sub_cost(a, b)
    rows.append((subs[a][b], a, b, cost, old, abs(cost - old)))
rows.sort(key=lambda r: -r[0])
print(f"{'관측':>5}  {'치환':12}{'학습':>7}{'기존':>7}", flush=True)
for n, a, b, cost, old, diff in rows[:18]:
    flag = "   ←" if diff >= 0.15 else ""
    print(f"{n:>5}  {a:4}→{b:6}{cost:>7.2f}{old:>7.2f}{flag}", flush=True)
