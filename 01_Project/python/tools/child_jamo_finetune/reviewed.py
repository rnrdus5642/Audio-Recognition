"""Apply listening decisions only to the exact source audio and label reviewed."""
from functools import lru_cache
from pathlib import Path
import hashlib

import audio
from common import digest, load

LEDGER = Path(__file__).with_name('reviewed-repairs.json')


@lru_cache(maxsize=1)
def decisions():
    document = load(LEDGER)
    if document['version'] != 1:
        raise ValueError('Unknown listening repair format')
    entries = {r['wav']: r for r in document['decisions']}
    if len(entries) != len(document['decisions']):
        raise ValueError('Duplicate listening decision')
    for entry in entries.values():
        choice = {'whole_source': 'full_only', 'exclude': 'no_speech'}[entry['action']]
        review = entry['review']
        if (review['choice'] != choice or not review['binding'] or
                not review['reason'] or review['revision'] < 1):
            raise ValueError('Repair lacks a matching human listening decision')
    return entries


def contract_sha256():
    return digest(LEDGER)


def _check_label(row, entry):
    for key in ('text', 'surface', 'label_ids'):
        if row[key] != entry[key]:
            raise ValueError(f'{row["wav"]}: {key} differs from the reviewed label')


def apply(row, full, source_sha256):
    """Return (row updates, decision); an exclusion has no row updates."""
    entry = decisions().get(row['wav'])
    if entry is None:
        return {}, None
    if row['split'] != 'train':
        raise ValueError(f'{row["wav"]}: training review cannot modify a holdout')
    if source_sha256 != entry['source_sha256']:
        raise ValueError(f'{row["wav"]}: audio differs from the reviewed source')
    _check_label(row, entry)
    if entry['action'] == 'exclude':
        return {}, entry
    pcm_sha256 = hashlib.sha256(full.tobytes()).hexdigest()
    if len(full) != entry['full_samples'] or pcm_sha256 != entry['full_pcm_sha256']:
        raise ValueError(f'{row["wav"]}: whole-source PCM differs from the review')
    seconds = len(full) / audio.SAMPLE_RATE
    return {'speech_start': '0.000', 'speech_end': f'{seconds:.3f}',
            'seconds_planned': seconds, 'span_policy': 'annotated_span'}, entry


def require_repaired(rows):
    """Stop a fresh training run from reusing a confirmed defective pair.

    Historical manifests remain available for reproducing old checkpoints.
    This check runs before allocating a model for a new training job.
    """
    for row in rows:
        entry = decisions().get(row['wav'])
        if entry is None:
            continue
        _check_label(row, entry)
        corrected = (entry['action'] == 'whole_source' and
                     row.get('source_sha256') == entry['source_sha256'] and
                     row.get('pcm_sha256') == entry['full_pcm_sha256'] and
                     row.get('samples') == entry['full_samples'] and
                     float(row['speech_start']) == 0.0 and
                     float(row['speech_end']) == entry['full_samples'] / audio.SAMPLE_RATE)
        if not corrected:
            raise RuntimeError(
                f'{row["wav"]}: confirmed defective training pair. Build and use '
                'the repaired split with repair.py and --train-split; the original '
                'manifest is retained only for baseline reproduction.')
