"""Choose the training, validation and test recordings, then verify them.

Selection is deterministic and happens before anything is scored. Two
holdouts are preserved from the start: the corpus split's own valid and
test speakers, which never enter training, and the 50 speakers that
served as development data for the previous model. Those 50 sit inside
the corpus 'train' pool, so without the extra exclusion they would have
walked back into training.

Hours are spread by round-robin over speakers rather than sampled flat.
A flat sample of this corpus is dominated by the handful of children who
recorded the most; taking one recording per child per pass means ~200
hours covers nearly every eligible child instead of a few hundred.

    python python/tools/child_jamo_finetune/manifest.py select
    python python/tools/child_jamo_finetune/manifest.py audit
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import sys
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
import catalog
import labels
import quality
import reviewed
from common import INDEX, OLD_DEV, SPLITS, WORK, digest, dump, load, log

AGES = ('5', '6', '7')
TRAIN_HOURS = 208.0      # a few per cent of surplus; the audit drops a little
VALID_UTTERANCES = 1500
TEST_UTTERANCES = 3000
MIN_SECONDS = 0.5
MAX_SECONDS = 15.0
SEED = 'child-jamo-2026-09-14'

TARGET_EXPRESSIONS = ['실로폰채', '들어', '잡아', '쳐', '때려', '연주해',
                      '놓아', '놔', '내려놔', '흔들어', '북채', '두드려',
                      '작아', '와']


def _order_key(name):
    return hashlib.sha256((SEED + ':' + name).encode()).hexdigest()


def old_dev_speakers():
    with OLD_DEV.open(encoding='utf-8', newline='') as f:
        return {r['speaker'] for r in csv.DictReader(f)}


#: Index columns that must agree when one basename appears twice. Two rows
#: that disagree on any of these describe different recordings under one
#: name, and there is no basis for picking a winner.
IDENTITY_COLUMNS = ('text', 'speaker', 'age', 'seconds', 'speech_start',
                    'speech_end', 'style', 'gender')


def planned_span(row):
    """Trimmed duration in seconds, using the one shared crop policy."""
    lo, hi, policy = audio.span(row['speech_start'], row['speech_end'],
                                float(row['seconds']))
    return hi - lo, policy


def candidates():
    """Every eligible utterance, grouped by (split, age, speaker)."""
    speakers = load(SPLITS)['speakers']
    reserved = old_dev_speakers()
    entries, ambiguous = catalog.read()
    pools = defaultdict(list)
    counts = Counter()
    seen = {}
    conflicting = set()
    t0 = time.monotonic()
    with INDEX.open(encoding='utf-8-sig', newline='') as f:
        for n, r in enumerate(csv.DictReader(f), 1):
            counts['index_rows'] += 1
            info = speakers.get(r['speaker'])
            if not info or r['age'] not in AGES:
                continue
            counts['eligible_speaker_rows'] += 1
            split = info['split']
            if split == 'train' and r['speaker'] in reserved:
                counts['excluded_previous_dev_speaker'] += 1
                continue
            name = unicodedata.normalize('NFC', r['wav'])
            fingerprint = tuple(r[c] for c in IDENTITY_COLUMNS)
            if name in seen:
                # An exact repeat is a duplicated index row and is simply
                # counted once. A repeat that disagrees means two different
                # recordings share a basename, and keeping whichever came
                # first would attach one recording's transcript to the
                # other's audio, so the name is dropped entirely.
                if seen[name] == fingerprint:
                    counts['excluded_duplicate_index_row'] += 1
                else:
                    conflicting.add(name)
                    counts['excluded_conflicting_index_row'] += 1
                continue
            seen[name] = fingerprint
            if name in ambiguous:
                counts['excluded_ambiguous_archive_member'] += 1
                continue
            if name not in entries:
                counts['excluded_missing_from_archives'] += 1
                continue
            seconds, policy = planned_span(r)
            if policy != 'annotated_span':
                counts['span_policy_' + policy] += 1
            if seconds < MIN_SECONDS:
                counts['excluded_too_short'] += 1
                continue
            if seconds > MAX_SECONDS:
                counts['excluded_too_long'] += 1
                continue
            text, surface, ids = labels.targets(r['text'])
            if ids is None:
                counts['excluded_label_' + text] += 1
                continue
            if audio.frames(round(seconds * audio.SAMPLE_RATE)) < labels.ctc_width(ids):
                counts['excluded_ctc_infeasible'] += 1
                continue
            counts['candidate_' + split] += 1
            archive, offset, size = entries[name]
            pools[(split, r['age'], r['speaker'])].append({
                'wav': name, 'archive': archive, 'offset': offset, 'size': size,
                'speaker': r['speaker'], 'age': r['age'], 'style': r['style'],
                'split': split, 'seconds_index': float(r['seconds']),
                'speech_start': r['speech_start'], 'speech_end': r['speech_end'],
                'seconds_planned': seconds, 'span_policy': policy,
                'text': text, 'surface': surface,
                'label_ids': ids, 'order': _order_key(name),
            })
            if n % 500000 == 0:
                log('manifest.log', f'INDEX {n:,} {time.monotonic() - t0:.0f}s')
    # A conflict is only visible at the second occurrence, by which time
    # the first is already in a pool. Drop both sides of every conflict.
    if conflicting:
        for key, pool in pools.items():
            keep = [r for r in pool if r['wav'] not in conflicting]
            counts['excluded_conflicting_index_row'] += len(pool) - len(keep)
            pools[key] = keep
    for pool in pools.values():
        pool.sort(key=lambda r: r['order'])
    counts['conflicting_basenames'] = len(conflicting)
    log('manifest.log', f'INDEX_DONE {counts["index_rows"]:,} rows '
        f'{time.monotonic() - t0:.0f}s')
    return pools, counts


def round_robin(pool_by_speaker, hours=None, utterances=None):
    """One recording per speaker per pass until the budget is met."""
    speakers = sorted(pool_by_speaker)
    cursors = {s: 0 for s in speakers}
    chosen, seconds = [], 0.0
    while speakers:
        progressed = False
        for speaker in list(speakers):
            rows = pool_by_speaker[speaker]
            i = cursors[speaker]
            if i >= len(rows):
                speakers.remove(speaker)
                continue
            chosen.append(rows[i])
            seconds += rows[i]['seconds_planned']
            cursors[speaker] = i + 1
            progressed = True
            if hours is not None and seconds >= hours * 3600:
                return chosen
            if utterances is not None and len(chosen) >= utterances:
                return chosen
        if not progressed:
            break
    return chosen


def allocate(available, total):
    """Split a budget evenly over ages, spilling what an age cannot supply."""
    share = {a: 0.0 for a in available}
    remaining, ages = total, sorted(available, key=lambda a: available[a])
    for i, age in enumerate(ages):
        even = remaining / (len(ages) - i)
        share[age] = min(available[age], even)
        remaining -= share[age]
    return share


def select():
    pools, counts = candidates()
    selected = {}
    allocation = {}
    for split, hours, utts in [('train', TRAIN_HOURS, None),
                               ('valid', None, VALID_UTTERANCES),
                               ('test', None, TEST_UTTERANCES)]:
        by_age = defaultdict(dict)
        for (s, age, speaker), rows in pools.items():
            if s == split:
                by_age[age][speaker] = rows
        if hours is not None:
            available = {a: sum(r['seconds_planned'] for rows in by_age[a].values()
                                for r in rows) / 3600 for a in by_age}
            share = allocate(available, hours)
            picked = [r for a in sorted(by_age)
                      for r in round_robin(by_age[a], hours=share[a])]
        else:
            per_age = {a: utts // len(by_age) + (i < utts % len(by_age))
                       for i, a in enumerate(sorted(by_age))}
            share = {a: float(per_age[a]) for a in by_age}
            picked = [r for a in sorted(by_age)
                      for r in round_robin(by_age[a], utterances=per_age[a])]
        allocation[split] = share
        picked.sort(key=lambda r: (r['archive'], r['offset']))
        selected[split] = picked
        log('manifest.log', f'SELECTED {split} {len(picked):,} recordings '
            f'{sum(r["seconds_planned"] for r in picked) / 3600:.2f}h '
            f'{len({r["speaker"] for r in picked})} speakers')

    groups = {s: {r['speaker'] for r in rows} for s, rows in selected.items()}
    for a in groups:
        for b in groups:
            assert a == b or not (groups[a] & groups[b]), (a, b)
    assert not (groups['train'] & old_dev_speakers())

    path = WORK / 'selection.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for split in ['train', 'valid', 'test']:
            for r in selected[split]:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dump(WORK / 'selection-summary.json', {
        'seed': SEED, 'labels_version': labels.LABELS_VERSION,
        'labels_contract_sha256': labels.contract_sha256(),
        'index_sha256': digest(INDEX), 'splits_sha256': digest(SPLITS),
        'old_dev_sha256': digest(OLD_DEV),
        'bounds_seconds': [MIN_SECONDS, MAX_SECONDS],
        # hours per age for train, recordings per age for valid and test
        'age_allocation': allocation,
        'exclusions': dict(sorted(counts.items())),
        'counts': {s: summarize(rows, 'seconds_planned')
                   for s, rows in selected.items()},
        'reserved_previous_dev_speakers': len(old_dev_speakers()),
    })
    log('manifest.log', 'SELECT_COMPLETE')


def summarize(rows, seconds_key):
    by_age = defaultdict(lambda: [0, 0.0, set()])
    for r in rows:
        slot = by_age[r['age']]
        slot[0] += 1
        slot[1] += r[seconds_key]
        slot[2].add(r['speaker'])
    return {
        'recordings': len(rows),
        'hours': sum(r[seconds_key] for r in rows) / 3600,
        'speakers': len({r['speaker'] for r in rows}),
        'style': dict(Counter(r['style'] for r in rows)),
        'by_age': {a: {'recordings': v[0], 'hours': v[1] / 3600,
                       'speakers': len(v[2])} for a, v in sorted(by_age.items())},
    }


# --- audit -----------------------------------------------------------

def _check(row):
    """Decode one recording and report what the archive actually holds."""
    try:
        blob = audio.payload(row['archive'], row['offset'], row['size'])
        x = audio.decode(blob)
        # The same crop the selection planned and training will apply; the
        # malformed-metadata policy lives inside it, not in this handler.
        cropped, policy = audio.crop(x, row['speech_start'], row['speech_end'])
        source_sha = hashlib.sha256(blob).hexdigest()
        pcm_sha = hashlib.sha256(cropped.tobytes()).hexdigest()
        # When re-auditing an existing manifest, verify the old input before
        # applying any review decision. A correction must not hide data drift.
        for key, actual in [('source_sha256', source_sha), ('pcm_sha256', pcm_sha),
                            ('samples', len(cropped)), ('span_policy', policy)]:
            if key in row and row[key] != actual:
                raise ValueError(f'{key} differs from the recorded input')
        jamo = sum(t not in (labels.BLANK, labels.BOUNDARY) for t in row['label_ids'])
        before = quality.measure(x, cropped, jamo)
        updates, decision = reviewed.apply(row, x, source_sha)
        if updates:
            cropped, policy = audio.crop(x, updates['speech_start'], updates['speech_end'])
        measured = quality.measure(x, cropped, jamo) if updates else before
    except Exception as exc:                      # corrupt or unreadable member
        return row['wav'], {'error': f'{type(exc).__name__}: {exc}'}
    return row['wav'], {
        'samples': len(cropped),
        'file_seconds': len(x) / audio.SAMPLE_RATE,
        'measured_span_policy': policy,
        'source_sha256': source_sha,
        'pcm_sha256': hashlib.sha256(cropped.tobytes()).hexdigest(),
        'peak': float(abs(cropped).max()) if len(cropped) else 0.0,
        'quality': measured,
        'row_updates': updates,
        'review_decision': decision,
        'quality_before_repair': before if decision else None,
    }


def audit(workers=8):
    repair_sha = reviewed.contract_sha256()
    quality_sha = quality.contract_sha256()
    rows = [json.loads(line) for line in
            (WORK / 'selection.jsonl').read_text(encoding='utf-8').splitlines()]
    log('manifest.log', f'AUDIT_START {len(rows):,} recordings')
    results = {}
    t0 = time.monotonic()
    with mp.Pool(workers) as pool:
        for n, (name, rec) in enumerate(
                pool.imap(_check, rows, chunksize=64), 1):
            results[name] = rec
            if n % 20000 == 0:
                log('manifest.log', f'AUDIT {n:,}/{len(rows):,} '
                    f'{time.monotonic() - t0:.0f}s')

    counts = Counter()
    policies = Counter()
    pcm_owner = {}
    kept = []
    candidates, repairs = [], []
    for r in rows:
        rec = results[r['wav']]
        if 'error' in rec:
            counts['dropped_unreadable'] += 1
            continue
        if rec['review_decision']:
            repairs.append({'wav': r['wav'], 'decision': rec['review_decision'],
                            'quality_before': rec['quality_before_repair'],
                            'quality_after': rec['quality']})
            if rec['review_decision']['action'] == 'exclude':
                counts['dropped_human_confirmed_no_speech'] += 1
                continue
        r.update(rec['row_updates'])
        if abs(rec['file_seconds'] - r['seconds_index']) > 0.05:
            counts['dropped_length_disagrees_with_index'] += 1
            continue
        if rec['samples'] < MIN_SECONDS * audio.SAMPLE_RATE:
            counts['dropped_too_short_measured'] += 1
            continue
        if rec['samples'] > MAX_SECONDS * audio.SAMPLE_RATE:
            counts['dropped_too_long_measured'] += 1
            continue
        if rec['peak'] == 0.0:
            counts['dropped_silent'] += 1
            continue
        if audio.frames(rec['samples']) < labels.ctc_width(r['label_ids']):
            counts['dropped_ctc_infeasible_measured'] += 1
            continue
        if rec['measured_span_policy'] != r['span_policy']:
            counts['dropped_span_policy_disagreed'] += 1
            continue
        policies[rec['measured_span_policy']] += 1
        prior = pcm_owner.get(rec['pcm_sha256'])
        if prior is not None:
            counts['dropped_duplicate_pcm'] += 1
            counts['duplicate_pcm_cross_split'] += prior != r['split']
            continue
        pcm_owner[rec['pcm_sha256']] = r['split']
        r.pop('order', None)
        r.update({k: rec[k] for k in
                  ['samples', 'source_sha256', 'pcm_sha256']})
        r['seconds'] = rec['samples'] / audio.SAMPLE_RATE
        if rec['quality']['flags']:
            candidates.append({'wav': r['wav'], 'split': r['split'],
                               'text': r['text'], 'source_sha256': rec['source_sha256'],
                               'pcm_sha256': rec['pcm_sha256'], **rec['quality']})
        kept.append(r)

    if repair_sha != reviewed.contract_sha256() or quality_sha != quality.contract_sha256():
        raise RuntimeError('Quality policy or listening decisions changed during audit')
    dump(WORK / 'quality-review.json', {
        'version': quality.VERSION, 'limitation': quality.LIMITATION,
        'candidates': candidates, 'applied_human_decisions': repairs,
        'candidate_count': len(candidates),
        'unreviewed_candidates_automatically_changed': 0,
    })
    by_split = defaultdict(list)
    for r in kept:
        by_split[r['split']].append(r)
    for split, group in by_split.items():
        group.sort(key=lambda r: r['wav'])
        path = WORK / f'manifest-{split}.jsonl'
        temp = path.with_suffix('.jsonl.tmp')
        with temp.open('w', encoding='utf-8') as f:
            for r in group:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        temp.replace(path)

    # Everything that decided these manifests, frozen together in the one
    # record that vouches for them. Reading the label contract from
    # selection-summary.json instead would let a later `select` run
    # rewrite it while this audit still described an older generation,
    # and both checks would pass while describing different things.
    selection = load(WORK / 'selection-summary.json')
    provenance = {
        'labels_version': labels.LABELS_VERSION,
        'labels_contract_sha256': labels.contract_sha256(),
        'selection_summary_sha256': digest(WORK / 'selection-summary.json'),
        'selection_seed': selection['seed'],
        'selection_inputs': {k: selection[k] for k in
                             ['index_sha256', 'splits_sha256',
                              'old_dev_sha256', 'bounds_seconds']},
        'source_archives': catalog.verify(),
        'reviewed_repairs_sha256': repair_sha,
        'quality_contract_sha256': quality_sha,
        'quality_review_sha256': digest(WORK / 'quality-review.json'),
    }
    summary = {
        'audited': len(rows), 'kept': len(kept),
        'drops': dict(sorted(counts.items())),
        'quality_review_candidates': len(candidates),
        'applied_human_decisions': len(repairs),
        'crop_policy_applied': dict(sorted(policies.items())),
        'provenance': provenance,
        'source_archives': provenance['source_archives'],
        'counts': {s: summarize(g, 'seconds') for s, g in sorted(by_split.items())},
        'manifest_sha256': {s: digest(WORK / f'manifest-{s}.jsonl')
                            for s in sorted(by_split)},
        'target_expression_coverage': coverage(by_split),
        'elapsed_seconds': time.monotonic() - t0,
        'limitation': 'Targets are transcript-derived pronunciation forms, '
                      'not annotations of the pronunciation the child produced.',
    }
    dump(WORK / 'audit-summary.json', summary)
    log('manifest.log', f'AUDIT_COMPLETE kept={len(kept):,} '
        f'drops={dict(counts)}')


def coverage(by_split):
    """Direct coverage of the target expressions, and of the jamo themselves.

    A substring count alone understates what the data teaches and
    overstates what it does not. This model has a jamo head: an
    expression that never appears whole is still built from units the
    corpus drills heavily, so the useful figures are (a) how often the
    expression appears as its own word rather than inside a longer one,
    and (b) whether every jamo it needs is well represented at all.
    Neither figure predicts word-level accuracy, which has to be
    measured separately.
    """
    out = {}
    for split, rows in sorted(by_split.items()):
        whole, contained, examples = Counter(), Counter(), {}
        jamo = Counter()
        for r in rows:
            words = r['text'].split()
            for word in TARGET_EXPRESSIONS:
                if word in words:
                    whole[word] += 1
                    examples.setdefault(word + '|whole', r['text'])
                elif word in r['text']:
                    contained[word] += 1
                    examples.setdefault(word + '|contained', r['text'])
            for t in r['label_ids']:
                jamo[labels.TOKENS[t]] += 1
        needed = sorted({t for w in TARGET_EXPRESSIONS
                         for t in (labels.targets(w)[2] or [])})
        out[split] = {
            'expressions': {w: {'as_whole_word': whole[w],
                                'inside_longer_word': contained[w]}
                            for w in TARGET_EXPRESSIONS},
            'examples': examples,
            'jamo_occurrences': dict(jamo.most_common()),
            'rarest_jamo_needed_by_target_expressions': {
                labels.TOKENS[t]: jamo[labels.TOKENS[t]]
                for t in sorted(needed, key=lambda t: jamo[labels.TOKENS[t]])[:8]},
        }
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['select', 'audit'])
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    if args.stage == 'select':
        select()
    else:
        audit(args.workers)
