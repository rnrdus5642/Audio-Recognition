"""A tiny, fixed, smoke-only view of the training pool.

The production entry point has to be exercised end to end - head phase
into encoder phase, a short accumulation tail, an epoch boundary, the
stop decision, and the final test of the best checkpoint - and that last
step scores a test manifest. Pointing it at the real test split would
spend the one evaluation those 2,999 recordings exist for.

So this carves three small speaker-disjoint views out of the TRAIN pool
only. The real valid and test manifests are never read here and never
scored during preflight. These files are for control-flow testing; no
number produced from them means anything about accuracy.

    python python/tools/child_jamo_finetune/smoke.py
"""
from collections import defaultdict
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data
import labels
from common import WORK, digest, dump, log

SIZES = {'smoketrain': 80, 'smokevalid': 24, 'smoketest': 24}
PREFIX = 'manifest-'


def main():
    rows = data.rows('train')
    by_speaker = defaultdict(list)
    for r in rows:
        by_speaker[r['speaker']].append(r)
    # Deterministic and speaker-disjoint: walk speakers in a fixed order
    # and hand each one entirely to a single view.
    speakers = sorted(by_speaker, key=lambda s: (len(by_speaker[s]), s))
    speakers = [s for s in speakers if len(by_speaker[s]) >= 8]
    views, cursor = {}, 0
    for name, size in SIZES.items():
        picked = []
        while len(picked) < size and cursor < len(speakers):
            picked.extend(sorted(by_speaker[speakers[cursor]],
                                 key=lambda r: r['wav']))
            cursor += 1
        views[name] = picked[:size]

    summary = {'note': 'SMOKE ONLY. Control-flow testing of the training '
                       'entry point. Drawn entirely from the train pool; the '
                       'real valid and test manifests are untouched. No score '
                       'from these files is meaningful.',
               'drawn_from': 'manifest-train.jsonl',
               'source_sha256': digest(WORK / 'manifest-train.jsonl'),
               # The views carry their own binding so integrity.py can
               # vouch for them exactly as audit-summary.json does for the
               # real splits.
               'labels_version': labels.LABELS_VERSION,
               'labels_contract_sha256': labels.contract_sha256(),
               'views': {}}
    for name, picked in views.items():
        path = WORK / f'{PREFIX}{name}.jsonl'
        temp = path.with_suffix('.jsonl.tmp')
        with temp.open('w', encoding='utf-8') as f:
            for r in picked:
                f.write(json.dumps(dict(r, split=name), ensure_ascii=False) + '\n')
        temp.replace(path)
        summary['views'][name] = {
            'recordings': len(picked),
            'speakers': sorted({r['speaker'] for r in picked}),
            'hours': sum(r['seconds'] for r in picked) / 3600,
            'sha256': digest(path),
        }
    groups = [set(v['speakers']) for v in summary['views'].values()]
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not (a & b), 'smoke views must not share a speaker'
    dump(WORK / 'smoke-view.json', summary)
    log('smoke.log', 'SMOKE_VIEW ' + json.dumps(
        {k: v['recordings'] for k, v in summary['views'].items()}))
    return summary


if __name__ == '__main__':
    main()
