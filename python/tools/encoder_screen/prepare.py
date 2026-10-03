"""Freeze a speaker-disjoint Korean child pilot and extract only its audio."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import heapq
import io
import json
import re
import sys
import tarfile
import time
import unicodedata
import numpy as np
import soundfile as sf
from common import ROOT, WORK, dump, load, digest

sys.path.insert(0, str(ROOT))
from python.build.g2p.ko.g2p import KoreanG2P
from python.build.g2p.ko.jamo_ipa import INITIALS, MEDIALS, FINALS
from python.workspace import DATASETS

INDEX = DATASETS / 'AIHUB정리/index.csv'
SPLITS = DATASETS / 'AIHUB정리/splits.json'
RAW = DATASETS / 'AIHUB'
TARGET = {'train': 6000, 'valid': 600, 'test': 600}
TOKENS = ['<blank>', '|'] + list(INITIALS) + list(MEDIALS)
VOCAB = {s: i for i, s in enumerate(TOKENS)}
G2P = KoreanG2P()


def labels(text):
    surface = G2P.apply_rules(text)
    out = []
    for ch in surface:
        if ch == ' ':
            out.append('|')
        elif '가' <= ch <= '힣':
            n = ord(ch) - 0xac00
            out += [INITIALS[n // 588], MEDIALS[n // 28 % 21]]
            if n % 28: out.append(FINALS[n % 28])
    if any(c not in VOCAB for c in out): return None
    return surface, [VOCAB[c] for c in out]


def select():
    manifest_path = WORK / 'selection.json'
    if manifest_path.exists(): return load(manifest_path)
    splits = load(SPLITS)['speakers']
    # A cap per child avoids allowing a few prolific speakers to dominate.
    pools = defaultdict(list)
    count = Counter()
    seen_names, duplicate_names = set(), set()
    t0 = time.monotonic()
    with INDEX.open(encoding='utf-8') as f:
        for n, r in enumerate(csv.DictReader(f), 1):
            if r['wav'] in seen_names:
                duplicate_names.add(r['wav'])
                continue
            seen_names.add(r['wav'])
            sp = splits.get(r['speaker'])
            if not sp or sp['split'] not in TARGET or r['age'] not in {'5', '6', '7'}: continue
            if not (1.0 <= float(r['seconds']) <= 6.5): continue
            raw = unicodedata.normalize('NFC', r['text'])
            if not re.fullmatch(r'[가-힣\s.,?!~]+', raw): continue
            text = ' '.join(re.sub(r'[^가-힣 ]', '', raw).split())
            if not (2 <= len(text.replace(' ', '')) <= 45): continue
            score = int(hashlib.sha256(('encoder-pilot-2026-09-14:' + r['wav']).encode()).hexdigest(), 16)
            key = (sp['split'], r['speaker'])
            entry = (-score, r['wav'], dict(r, text=text, split=sp['split']))
            heap = pools[key]
            cap = 16 if sp['split'] == 'train' else 8
            if len(heap) < cap: heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]: heapq.heapreplace(heap, entry)
            count[sp['split']] += 1
    selected = []
    for split, limit in TARGET.items():
        ranked = sorted(((-e[0], e[1], e[2]) for k, heap in pools.items() if k[0] == split for e in heap if e[1] not in duplicate_names), key=lambda e: (e[0], e[1]))
        # Approximately equal age representation, fixed before any model scores.
        age_counts = Counter()
        for _, _, r in ranked:
            if age_counts[r['age']] >= limit // 3: continue
            lab = labels(r['text'])
            if lab is None: continue
            r['surface'], r['label_ids'] = lab
            r['id'] = Path(r['wav']).stem
            selected.append(r)
            age_counts[r['age']] += 1
        assert sum(age_counts.values()) == limit, (split, age_counts)
    for a in TARGET:
        for b in TARGET:
            if a != b:
                assert not ({r['speaker'] for r in selected if r['split'] == a} & {r['speaker'] for r in selected if r['split'] == b})
    result = {'selection_seed': 'encoder-pilot-2026-09-14', 'target': TARGET,
              'duplicate_archive_basenames_excluded': len(duplicate_names),
              'index_sha256': digest(INDEX), 'split_sha256': digest(SPLITS),
              'labels_note': 'Project surface-form rules -> compatibility jamo; derived transcript labels, not human pronunciation-error annotation.',
              'vocab': VOCAB, 'rows': selected}
    dump(manifest_path, result)
    print('SELECTED', len(selected), 'eligible', dict(count), 'seconds', round(time.monotonic()-t0, 1), flush=True)
    return result


def extract(manifest):
    dest = WORK / 'audio'
    dest.mkdir(exist_ok=True)
    pending = {r['wav']: r for r in manifest['rows'] if not (dest / r['wav']).exists()}
    rejected_path = WORK / 'rejected-audio.json'
    rejected = load(rejected_path) if rejected_path.exists() else {}
    arcs = sorted(p for p in RAW.rglob('*.tar') if re.match('[TV]S_', p.name, re.I))
    for arc in arcs:
        if not pending: break
        print('SCAN', arc.name, 'remaining', len(pending), flush=True)
        found = 0
        # Plain tar seek skips unselected audio payloads; no archive-wide extraction.
        with tarfile.open(arc, 'r:') as tar:
            for member in tar:
                name = Path(member.name).name
                if name not in pending or not member.isfile(): continue
                r = pending[name]
                blob = tar.extractfile(member).read()
                try:
                    x, sr = sf.read(io.BytesIO(blob), dtype='float32')
                except (sf.LibsndfileError, ValueError) as exc:
                    rejected[name] = {'archive': arc.name, 'error': str(exc), 'bytes':len(blob),
                                      'prefix_hex':blob[:16].hex(), 'sha256':hashlib.sha256(blob).hexdigest()}
                    dump(rejected_path, rejected)
                    del pending[name]
                    print('REJECT_AUDIO', name, type(exc).__name__, flush=True)
                    continue
                assert sr == 16000 and np.isfinite(x).all(), name
                if x.ndim == 2: x = x.mean(axis=1)
                lo = max(0, round((float(r['speech_start'] or 0) - .25) * sr))
                hi = min(len(x), round((float(r['speech_end'] or len(x)/sr) + .25) * sr))
                assert hi-lo >= 8000
                x = np.ascontiguousarray(x[lo:hi], dtype=np.float32)
                sf.write(dest / name, x, sr, subtype='FLOAT')
                r['archive'] = arc.name
                r['source_member'] = member.name
                r['source_sha256'] = hashlib.sha256(blob).hexdigest()
                r['pcm_sha256'] = hashlib.sha256(x.tobytes()).hexdigest()
                r['samples'] = len(x)
                dump(dest / (name + '.json'), {k: r[k] for k in ['archive','source_member','source_sha256','pcm_sha256','samples']})
                del pending[name]
                found += 1
                if found % 500 == 0: print('EXTRACT', arc.name, found, flush=True)
        print('SCANNED', arc.name, found, flush=True)
    dump(WORK/'missing-audio.json', sorted(pending))
    manifest['unavailable_selected_recordings'] = sorted(set(pending) | set(rejected))
    manifest['rows'] = [r for r in manifest['rows'] if (dest/r['wav']).exists()]
    counts = {}
    for r in manifest['rows']:
        r.update(load(dest / (r['wav'] + '.json')))
    for split in TARGET:
        rows = [r for r in manifest['rows'] if r['split'] == split]
        counts[split] = {'recordings': len(rows), 'speakers': len({r['speaker'] for r in rows}),
                         'hours': sum(r['samples'] for r in rows) / 16000 / 3600,
                         'age_recordings': dict(Counter(r['age'] for r in rows))}
        assert len(rows) >= TARGET[split] * .8, (split,counts[split])
    manifest['counts'] = counts
    dump(WORK/'manifest.json', manifest)
    dump(WORK/'vocab.json', VOCAB)
    print('DATA_READY', json.dumps(counts), flush=True)


if __name__ == '__main__':
    extract(select())
