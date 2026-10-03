"""What does relaxing the false-accept budget actually buy?

Runs the same search at several budgets so the trade can be read as a
curve instead of argued from one point. The matrix is the one learned by
autotune - fixed here, since only the budget is meant to vary.
"""

import json
import os

from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import Matcher

BASE = os.path.dirname(os.path.abspath(__file__))
SESSION_S = 10.0
BUDGETS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 1.0]

SKIP = [0.01, 0.02, 0.03]
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
    "tune": (raw["tune_pos"] + raw["more_pos"],
             raw["tune_neg"] + raw["more_neg"]),
    "valid": (raw["valid_pos"], raw["valid_neg"]),
    "check": (raw["check_pos"], raw["check_neg"]),
}
targets = json.load(open("shared/targets.json", encoding="utf-8"))["answers"]
by_text = {t["text"]: t for t in targets}
LENGTHS = sorted({len(t["phonemes"]) for t in targets})
matrix = ConfusionMatrix.from_json(os.path.join(BASE, "ko_child_v2.json"))


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
    return p, g, sum(i["seconds"] for i in neg), len(neg)


def fires(seq, th, con):
    streak = 0
    for s in seq:
        streak = streak + 1 if s >= th else 0
        if streak >= con:
            return True
    return False


cache = {}
for skip in SKIP:
    for cov in COVERAGE:
        for ctx in CTX:
            m = Matcher(matrix, skip_cost=skip, coverage=cov, context_mult=ctx)
            cache[(skip, cov, ctx)] = (m, scores(m, "tune"))
print("점수 계산 완료", flush=True)

print(f"\n{'예산':>8}{'검출':>9}{'실제 오확정':>13}   설정", flush=True)
picks = {}
for budget in BUDGETS:
    best = None
    for (skip, cov, ctx), (m, (p, g, secs, n_items)) in cache.items():
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
                    rate = trips / max(secs * words, 1e-9) * SESSION_S
                    key = (rate <= budget, hit / max(len(pn), 1), -rate)
                    if pick is None or key > pick[0]:
                        pick = (key, t)
                th[n] = pick[1]
            hit = sum(fires(s, th[n], con) for n, s in p)
            trips = sum(fires(s, th[n], con) for n, s in g)
            words = max(1, len(g) // max(1, n_items))
            rate = trips / max(secs * words, 1e-9) * SESSION_S
            if rate <= budget and (best is None or hit / len(p) > best[0]):
                best = (hit / len(p), rate, skip, cov, ctx, con, th)
    if best:
        picks[budget] = best
        d, rate, skip, cov, ctx, con, th = best
        label = "제한없음" if budget >= 1.0 else f"{budget:.1%}"
        print(f"{label:>8}{d:>9.1%}{rate:>13.2%}"
              f"   skip {skip} cov {cov} ctx {ctx} 연속 {con}", flush=True)

print(f"\n{'예산':>8}{'tune':>18}{'valid':>18}{'check':>18}", flush=True)
for budget, best in picks.items():
    _d, _r, skip, cov, ctx, con, th = best
    m = Matcher(matrix, skip_cost=skip, coverage=cov, context_mult=ctx)
    row = []
    for split in ("tune", "valid", "check"):
        p, g, secs, n_items = scores(m, split)
        hit = sum(fires(s, th[n], con) for n, s in p)
        trips = sum(fires(s, th[n], con) for n, s in g)
        words = max(1, len(g) // max(1, n_items))
        rate = trips / max(secs * words, 1e-9) * SESSION_S
        row.append(f"{hit/max(len(p),1):5.1%} / {rate:5.2%}")
    label = "제한없음" if budget >= 1.0 else f"{budget:.1%}"
    print(f"{label:>8}" + "".join(f"{c:>18}" for c in row), flush=True)

print(f"\n예산별 임계값", flush=True)
for budget, best in picks.items():
    label = "제한없음" if budget >= 1.0 else f"{budget:.1%}"
    print(f"  {label:>8}  {best[6]}", flush=True)
