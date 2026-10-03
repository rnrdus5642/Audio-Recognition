"""Which knob actually moves the numbers?

Holds the proposed setting fixed and varies one parameter at a time, so
the spread in each block is that parameter's influence and nothing else.
"""

import json
import os

from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import Matcher
from python.runtime.matching.streaming import StreamingMatcher

BASE = os.path.dirname(os.path.abspath(__file__))
SESSION_S = 10.0

SETS = {}
for f in ("tune_frames.json", "valid_frames.json"):
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        SETS.update(json.load(open(p, encoding="utf-8"))["sets"])

matrix = ConfusionMatrix.from_json("shared/confusion_matrices/ko_child_v1.json")
targets = json.load(open("shared/targets.json", encoding="utf-8"))["answers"]
by_text = {t["text"]: t for t in targets}

PROPOSED_TH = {3: 0.75, 4: 0.80, 5: 0.80, 6: 0.75, 7: 0.75, 8: 0.70}
PROPOSED = dict(skip=0.03, coverage=0.9, ctx=5.0, consecutive=2)


def confirms(m, frames, target, threshold, consecutive):
    t = dict(target)
    t["threshold"] = threshold
    sm = StreamingMatcher(m, [t], consecutive=consecutive)
    for f in frames:
        if sm.push(f):
            return True
    return False


def run(th, skip, coverage, ctx, consecutive):
    m = Matcher(matrix, skip_cost=skip, coverage=coverage, context_mult=ctx)
    det = hit = 0
    for item in SETS["tune_pos"]:
        for w in item["hits"]:
            t = by_text[w]
            det += 1
            hit += confirms(m, item["frames"], t,
                            th[len(t["phonemes"])], consecutive)
    trips = seconds = 0.0
    per_utt = 0
    for item in SETS["tune_neg"]:
        seconds += item["seconds"]
        count = 0
        for t in targets:
            if t["text"] in item["text"]:
                continue
            count += 1
            trips += confirms(m, item["frames"], t,
                              th[len(t["phonemes"])], consecutive)
        per_utt = max(per_utt, count)
    rate = trips / max(seconds * max(per_utt, 1), 1e-9) * SESSION_S
    return hit / max(det, 1), rate


base_det, base_fa = run(PROPOSED_TH, PROPOSED["skip"], PROPOSED["coverage"],
                        PROPOSED["ctx"], PROPOSED["consecutive"])
print(f"기준(제안 설정)  검출 {base_det:.1%}  오확정 {base_fa:.2%}\n", flush=True)

blocks = []


def block(name, values, apply):
    rows = []
    for v in values:
        args = dict(th=PROPOSED_TH, skip=PROPOSED["skip"],
                    coverage=PROPOSED["coverage"], ctx=PROPOSED["ctx"],
                    consecutive=PROPOSED["consecutive"])
        apply(args, v)
        d, f = run(args["th"], args["skip"], args["coverage"],
                   args["ctx"], args["consecutive"])
        rows.append((v, d, f))
    dets = [r[1] for r in rows]
    fas = [r[2] for r in rows]
    blocks.append((name, max(dets) - min(dets), max(fas) - min(fas)))
    print(f"=== {name} ===", flush=True)
    for v, d, f in rows:
        mark = "  ←" if str(v) == str(
            {"임계값 전체 이동": 0.0, "skip_cost": PROPOSED["skip"],
             "커버리지": PROPOSED["coverage"], "문맥 배수": PROPOSED["ctx"],
             "연속 확인": PROPOSED["consecutive"]}.get(name, "")) else ""
        print(f"  {str(v):>8}   검출 {d:>6.1%}   오확정 {f:>7.2%}{mark}",
              flush=True)
    print(flush=True)


block("임계값 전체 이동", [-0.10, -0.05, 0.0, 0.05, 0.10],
      lambda a, v: a.update(th={k: round(min(0.95, val + v), 2)
                                for k, val in PROPOSED_TH.items()}))
block("skip_cost", [0.01, 0.03, 0.05, 0.10, 0.15],
      lambda a, v: a.update(skip=v))
block("커버리지", [0.6, 0.7, 0.8, 0.9, 1.0],
      lambda a, v: a.update(coverage=v))
block("문맥 배수", [2.0, 3.0, 4.0, 5.0, 8.0],
      lambda a, v: a.update(ctx=v))
block("연속 확인", [1, 2, 3, 4],
      lambda a, v: a.update(consecutive=v))

print("=== 영향력 순위 (검출 변동폭) ===", flush=True)
for name, d, f in sorted(blocks, key=lambda x: -x[1]):
    print(f"  {name:14} 검출 ±{d:5.1%}   오확정 ±{f:6.2%}", flush=True)
