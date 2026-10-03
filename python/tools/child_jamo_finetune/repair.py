"""Build a separately audited training split from verified listening decisions.

The original manifests and source archives are read-only. Every old source
and crop hash is checked; unreviewed rows keep their exact original bytes.
"""
from collections import Counter
from pathlib import Path
import argparse
import json
import multiprocessing as mp
import re
import sqlite3
import time

import audio
import integrity
import labels
import manifest
import quality
import reviewed
from common import ROOT, WORK, digest, dump, load

DEFAULT_SPLIT = 'train-repaired-20260927'


def verify_listening(ledger):
    path = ROOT / ledger['review_report']
    report = load(path)
    evidence = {r['source_id']: r for r in report['answered']}
    database = path.parent / 'followup-listening/listening-answers.sqlite3'
    connection = sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        for entry in ledger['decisions']:
            answer = evidence['train:' + entry['wav']]
            expected = entry['review']
            for key in ('item_id', 'revision', 'binding', 'choice'):
                if answer[key] != expected[key]:
                    raise RuntimeError(f'{entry["wav"]}: listening evidence changed')
            if (not answer['played_audio_and_binding_verified'] or
                    answer['source_pcm_sha256'] != entry['original_pcm_sha256']):
                raise RuntimeError(f'{entry["wav"]}: listening audio is not verified')
            latest = connection.execute(
                'SELECT revision, binding, choice FROM answers WHERE item_id=? '
                'ORDER BY revision DESC LIMIT 1', (expected['item_id'],)).fetchone()
            if latest != (expected['revision'], expected['binding'], expected['choice']):
                raise RuntimeError(f'{entry["wav"]}: a newer listening answer needs review')
    finally:
        connection.close()
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}


def repaired_row(row, result):
    """Preserve every unreviewed row; change only the reviewed crop fields."""
    if 'error' in result:
        raise RuntimeError(f'{row["wav"]}: {result["error"]}')
    decision = result['review_decision']
    if decision and decision['action'] == 'exclude':
        return None
    if not result['row_updates']:
        return dict(row)
    changed = dict(row, **result['row_updates'])
    changed.update({k: result[k] for k in ('samples', 'source_sha256', 'pcm_sha256')})
    changed['seconds'] = changed['samples'] / audio.SAMPLE_RATE
    if not manifest.MIN_SECONDS <= changed['seconds'] <= manifest.MAX_SECONDS:
        raise RuntimeError('Reviewed full source is outside training length bounds')
    if audio.frames(changed['samples']) < labels.ctc_width(changed['label_ids']):
        raise RuntimeError('Reviewed full source cannot fit its CTC target')
    return changed


def build(name=DEFAULT_SPLIT, workers=6):
    if not re.fullmatch(r'train-repaired-[a-zA-Z0-9_-]+', name):
        raise ValueError('Use a new split name beginning with train-repaired-')
    if workers < 1:
        raise ValueError('workers must be positive')
    target = WORK / f'manifest-{name}.jsonl'
    audit_path = WORK / f'audit-{name}.json'
    review_path = WORK / f'quality-review-{name}.json'
    for path in (target, audit_path, review_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite {path}; choose a new --name')
    started = time.monotonic()
    parent = integrity._provenance('train')
    baseline = {s: digest(WORK / f'manifest-{s}.jsonl') for s in ('train', 'valid', 'test')}
    base_audit = digest(WORK / 'audit-summary.json')
    repair_sha, quality_sha = reviewed.contract_sha256(), quality.contract_sha256()
    ledger = load(reviewed.LEDGER)
    if parent['manifest_sha256'] != ledger['source_manifest_sha256']:
        raise RuntimeError('Repair decisions belong to a different baseline manifest')
    evidence = verify_listening(ledger)
    lines = (WORK / 'manifest-train.jsonl').read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if len({r['wav'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate source recording in baseline')
    # A restored full source must not become a duplicate, including holdouts.
    pcm_owners, holdout_speakers = {}, set()
    for split in ('valid', 'test'):
        for line in (WORK / f'manifest-{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            pcm_owners[row['pcm_sha256']] = split + ':' + row['wav']
            holdout_speakers.add(row['speaker'])
    candidates, changes, kept, output = [], [], [], []
    flag_counts = Counter()
    with mp.Pool(workers) as pool:
        for i, (wav, result) in enumerate(pool.imap(manifest._check, rows, chunksize=64)):
            row = rows[i]
            if wav != row['wav']:
                raise RuntimeError('Audit result order changed')
            after = repaired_row(row, result)
            if result['review_decision']:
                changes.append({'wav': wav, 'before': row, 'after': after,
                                'decision': result['review_decision'],
                                'quality_before': result['quality_before_repair'],
                                'quality_after': result['quality'] if after else None})
            if after is not None:
                prior = pcm_owners.get(after['pcm_sha256'])
                if prior:
                    raise RuntimeError(f'Repair creates duplicate PCM: {wav}, {prior}')
                pcm_owners[after['pcm_sha256']] = 'train:' + wav
                if after['speaker'] in holdout_speakers:
                    raise RuntimeError('Training speaker overlaps a holdout')
                kept.append(after)
                output.append(lines[i] if after == row else
                              (json.dumps(after, ensure_ascii=False) + '\n').encode('utf-8'))
                if result['quality']['flags']:
                    candidates.append({'wav': wav, 'split': 'train', 'text': after['text'],
                                       'source_sha256': after['source_sha256'],
                                       'pcm_sha256': after['pcm_sha256'], **result['quality']})
                    flag_counts.update(result['quality']['flags'])
            if (i + 1) % 10000 == 0 or i + 1 == len(rows):
                print(f'REPAIR_AUDIT {i + 1}/{len(rows)} '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    if {r['wav'] for r in changes} != set(reviewed.decisions()):
        raise RuntimeError('Not every verified decision was applied exactly once')
    reviewed.require_repaired(kept)
    # No artifacts are published until the complete source audit succeeds.
    for split, expected in baseline.items():
        if digest(WORK / f'manifest-{split}.jsonl') != expected:
            raise RuntimeError(f'{split} changed during repair')
    if (digest(WORK / 'audit-summary.json') != base_audit or
            reviewed.contract_sha256() != repair_sha or quality.contract_sha256() != quality_sha or
            verify_listening(ledger) != evidence):
        raise RuntimeError('Repair evidence or policy changed during the audit')
    if integrity._provenance('train') != parent:
        raise RuntimeError('Baseline provenance changed during repair')
    dump(review_path, {'version': quality.VERSION, 'limitation': quality.LIMITATION,
                       'candidate_count': len(candidates), 'flag_counts': dict(flag_counts),
                       'candidates': candidates, 'unreviewed_candidates_automatically_changed': 0})
    temporary = target.with_suffix('.jsonl.tmp')
    temporary.write_bytes(b''.join(output))
    temporary.replace(target)
    summary = {
        'kind': 'reviewed_training_repair', 'complete': True, 'split': name,
        'source_recordings': len(rows), 'recordings': len(kept),
        'counts': manifest.summarize(kept, 'seconds'),
        'changes': changes, 'quality_review_candidates': len(candidates),
        'manifest_sha256': {name: digest(target)},
        'provenance': {
            'labels_version': labels.LABELS_VERSION,
            'labels_contract_sha256': labels.contract_sha256(),
            'base_split': 'train', 'base_manifest_sha256': baseline['train'],
            'base_audit_sha256': base_audit,
            'reviewed_repairs_sha256': repair_sha,
            'quality_contract_sha256': quality_sha,
            'quality_review_file': review_path.name,
            'quality_review_sha256': digest(review_path),
            'listening_evidence': evidence,
        },
        'validation': {'all_original_source_and_crop_hashes_verified': len(rows),
                       'unchanged_rows_preserved_byte_for_byte': len(kept) - sum(
                           c['after'] is not None for c in changes),
                       'baseline_manifests_unchanged': baseline,
                       'speaker_overlap_with_holdouts': 0, 'duplicate_pcm': 0},
        'elapsed_seconds': round(time.monotonic() - started, 1),
        'model_retrained': False,
    }
    dump(audit_path, summary)
    integrity._provenance(name)
    print(f'REPAIR_COMPLETE {name}: {len(rows)} -> {len(kept)}, '
          f'{len(changes)} human decisions, {len(candidates)} review candidates', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=DEFAULT_SPLIT)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    build(args.name, args.workers)
