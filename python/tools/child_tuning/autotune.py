"""Alternate between learning the matrix and searching the parameters.

Neither can be settled alone: alignments depend on skip_cost and
coverage, and the best thresholds depend on the matrix. So one is held
while the other moves, and the pair is iterated until detection stops
improving on held-out children.

Splits are disjoint by speaker:
  tune   free 6-7 (15명) + formatted 6-7 (103명)
  valid  free 8-9 (47명)          same conditions, older
  check  formatted 5 (30명)       younger, and read rather than free

Selection uses tune only. valid and check are printed every round so a
choice that only suits the tuning children is visible rather than
inferred at the end.
"""

import json
import math
import os
import time
from collections import Counter, defaultdict

from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import Matcher

BASE = os.path.dirname(os.path.abspath(__file__))
SESSION_S = 10.0
BUDGET = 0.01
ROUNDS = 3

MIN_PAIR, MIN_SOURCE = 5, 30
MAX_DIST_RATIO, P_FLOOR, SHRINK_K = 0.6, 0.01, 20.0

SKIP = [0.005, 0.01, 0.02, 0.03]
COVERAGE = [0.8, 0.9]
CTX = [6.0, 8.0, 12.0]
CONSECUTIVE = [1, 2]
THRESHOLDS = [round(0.50 + 0.025 * i, 3) for i in range(19)]

raw = {}
for f in ("tune_frames.json", "valid_frames.json", "more_frames.json"):
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        raw.update(json.load(open(p, encoding="utf-8"))["sets"])

SPLITS = {
    "tune": (raw["tune_pos"] + raw["more_pos"], raw["tune_neg"] + raw["more_neg"]),
    "valid": (raw["valid_pos"], raw["valid_neg"]),
    "check": (raw["check_pos"], raw["check_neg"]),
}
for name, (pos, neg) in SPLITS.items():
    print(f"{name}: 검출 {len(pos)} · 오발동 {len(neg)} · 화자 "
          f"{len({i['speaker'] for i in pos + neg})}명", flush=True)

targets = json.load(open("shared/targets.json", encoding="utf-8"))["answers"]
by_text = {t["text"]: t for t in targets}
LENGTHS = sorted({len(t["phonemes"]) for t in targets})
HAND = ConfusionMatrix.from_json("shared/confusion_matrices/ko_child_v1.json")


def learn(prior, skip, coverage):
    """Count what children produced for each target phoneme."""
    m = Matcher(prior, skip_cost=skip, coverage=coverage)
    subs = defaultdict(Counter)
    dels = Counter()
    total = Counter()
    pos, _ = SPLITS["tune"]
    used = 0
    for item in pos:
        for w in item["hits"]:
            target = by_text[w]["phonemes"]
            best = None
            for frame in item["frames"]:
                if not frame:
                    continue
                dist, _s, _ws, _we, ops = m.score_against(frame, target)
                if best is None or dist < best[0]:
                    best = (dist, ops)
            if best is None or best[0] / max(len(target), 1) > MAX_DIST_RATIO:
                continue
            used += 1
            for user_p, target_p, op in best[1]:
                if op in ("match", "sub"):
                    subs[target_p][user_p if op == "sub" else target_p] += 1
                    total[target_p] += 1
                elif op == "del":
                    dels[target_p] += 1
                    total[target_p] += 1

    substitutions, deletions = {}, {}
    for a, counter in subs.items():
        if total[a] < MIN_SOURCE:
            continue
        for b, n in counter.items():
            if a == b or n < MIN_PAIR:
                continue
            p = (n + 0.5) / (total[a] + 1.0)
            learned = max(0.05, min(1.0, math.log(p) / math.log(P_FLOOR)))
            prior_cost = HAND.sub_cost(a, b)
            substitutions[(a, b)] = round(
                (n * learned + SHRINK_K * prior_cost) / (n + SHRINK_K), 3)
    for a, n in dels.items():
        if total[a] < MIN_SOURCE or n < MIN_PAIR:
            continue
        p = (n + 0.5) / (total[a] + 1.0)
        learned = max(0.05, min(1.0, math.log(p) / math.log(P_FLOOR)))
        deletions[a] = round(
            (n * learned + SHRINK_K * HAND.del_cost(a)) / (n + SHRINK_K), 3)

    merged = dict(HAND.known_substitutions) if isinstance(
        HAND.known_substitutions, dict) else {}
    merged.update(substitutions)
    return ConfusionMatrix(
        matrix_id="ko_child_v2", language="ko", version="2.0.0",
        substitutions=merged, deletions=deletions, insertions={},
        default_substitution=HAND.default_substitution,
        default_deletion=HAND.default_deletion,
        default_insertion=HAND.default_insertion,
        skip_cost=HAND.skip_cost,
        streaming_profile=dict(HAND.streaming_profile)), used, len(substitutions)


def scores(m, split):
    pos, neg = SPLITS[split]
    p = [(len(by_text[w]["phonemes"]),
          [m.score_against(f, by_text[w]["phonemes"])[1] if f else 0.0
           for f in item["frames"]])
         for item in pos for w in item["hits"]]
    g = [(len(t["phonemes"]),
          [m.score_against(f, t["phonemes"])[1] if f else 0.0
           for f in item["frames"]])
         for item in neg for t in targets if t["text"] not in item["text"]]
    seconds = sum(i["seconds"] for i in neg)
    return p, g, seconds, len(neg)


def fires(seq, th, con):
    streak = 0
    for s in seq:
        streak = streak + 1 if s >= th else 0
        if streak >= con:
            return True
    return False


def report(p, g, seconds, n_items, th, con):
    hit = sum(fires(s, th[n], con) for n, s in p)
    trips = sum(fires(s, th[n], con) for n, s in g)
    words = max(1, len(g) // max(1, n_items))
    return hit, len(p), trips / max(seconds * words, 1e-9) * SESSION_S


matrix = HAND
best_overall = None
t0 = time.time()

for rnd in range(1, ROUNDS + 1):
    print(f"\n{'='*62}\n반복 {rnd}\n{'='*62}", flush=True)

    if rnd > 1:
        skip = best_overall["skip"]
        cov = best_overall["coverage"]
        matrix, used, learned_n = learn(matrix, skip, cov)
        print(f"matrix 재학습: 정렬 {used}건 · 치환 {learned_n}쌍 갱신",
              flush=True)

    round_best = None
    for skip in SKIP:
        for cov in COVERAGE:
            for ctx in CTX:
                m = Matcher(matrix, skip_cost=skip, coverage=cov,
                            context_mult=ctx)
                p, g, secs, n_items = scores(m, "tune")
                for con in CONSECUTIVE:
                    th = {}
                    for n in LENGTHS:
                        pn = [x for x in p if x[0] == n]
                        gn = [x for x in g if x[0] == n]
                        words = max(1, len(gn) // max(1, n_items))
                        pick = None
                        for t in THRESHOLDS:
                            hit = sum(fires(s, t, con) for _n, s in pn)
                            trips = sum(fires(s, t, con) for _n, s in gn)
                            rate = trips / max(
                                secs * words, 1e-9) * SESSION_S
                            key = (rate <= BUDGET,
                                   hit / max(len(pn), 1), -rate)
                            if pick is None or key > pick[0]:
                                pick = (key, t)
                        th[n] = pick[1]
                    hit, det, rate = report(p, g, secs, n_items, th, con)
                    cand = {"skip": skip, "coverage": cov, "ctx": ctx,
                            "consecutive": con, "thresholds": th,
                            "detection": hit / max(det, 1), "rate": rate}
                    if rate <= BUDGET and (
                            round_best is None
                            or cand["detection"] > round_best["detection"]):
                        round_best = cand

    best_overall = round_best
    print(f"튜닝 최적: 검출 {round_best['detection']:.1%} · "
          f"오확정 {round_best['rate']:.2%} · skip {round_best['skip']} · "
          f"cov {round_best['coverage']} · ctx {round_best['ctx']} · "
          f"연속 {round_best['consecutive']}", flush=True)
    print(f"  임계값 {round_best['thresholds']}", flush=True)

    m = Matcher(matrix, skip_cost=round_best["skip"],
                coverage=round_best["coverage"],
                context_mult=round_best["ctx"])
    for split in ("tune", "valid", "check"):
        p, g, secs, n_items = scores(m, split)
        hit, det, rate = report(p, g, secs, n_items,
                                round_best["thresholds"],
                                round_best["consecutive"])
        print(f"  {split:6} 검출 {hit/max(det,1):5.1%} ({hit}/{det})  "
              f"오확정 {rate:5.2%}", flush=True)

json.dump(
    {"skip": best_overall["skip"], "coverage": best_overall["coverage"],
     "ctx": best_overall["ctx"], "consecutive": best_overall["consecutive"],
     "thresholds": best_overall["thresholds"],
     "substitutions": {f"{a}|{b}": c
                       for (a, b), c in matrix.known_substitutions.items()}
     if isinstance(matrix.known_substitutions, dict) else {}},
    open(os.path.join(BASE, "autotune_best.json"), "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)
print(f"\n{time.time()-t0:.0f}초", flush=True)
