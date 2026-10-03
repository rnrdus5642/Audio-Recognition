"""Audit all jamo errors from frozen predictions; never infer phonetic truth.

Positions belong to reference Hangul syllables. Edit alignment is textual,
not acoustic, and insertions are not assigned to a nonexistent reference.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rapidfuzz.distance import Levenshtein
import labels
from common import ROOT, WORK, digest, dump, load

OUT = ROOT / '.venv/child-jamo-web-20260916/cause-analysis-20260927'


def units(surface):
    result, word, syllable = [], 0, 0
    for ch in surface:
        if ch == ' ':
            word += 1
            continue
        n = ord(ch) - 0xAC00
        if not 0 <= n < 11172:
            raise ValueError('Reference must be a Hangul surface, not free-text notes.')
        parts = [('initial', labels.INITIALS[n // 588]),
                 ('medial', labels.MEDIALS[n // 28 % 21])]
        final = labels.FINALS[n % 28]
        if final:
            parts.append(('final', final))
        for position, token in parts:
            result.append({'token': token, 'position': position, 'syllable': syllable,
                           'character': ch, 'word': word})
        syllable += 1
    return result


def align(surface, ids):
    reference_units = units(surface)
    reference = [u['token'] for u in reference_units]
    hypothesis = [labels.TOKENS[t] for t in ids if t not in (labels.BLANK, labels.BOUNDARY)]
    events, changed = [], set()
    for op in Levenshtein.editops(reference, hypothesis):
        ref = reference_units[op.src_pos] if op.tag != 'insert' else None
        if ref:
            changed.add(op.src_pos)
        events.append({'op': op.tag, 'ref_index': op.src_pos, 'hyp_index': op.dest_pos,
                       'reference': ref['token'] if ref else None,
                       'hypothesis': hypothesis[op.dest_pos] if op.tag != 'delete' else None,
                       'position': ref['position'] if ref else None,
                       'character': ref['character'] if ref else None,
                       'word': ref['word'] if ref else None})
    assert len(events) == Levenshtein.distance(reference, hypothesis)
    return reference_units, events, changed


def metrics(rows):
    den = sum(r['reference_jamo'] for r in rows)
    errors = sum(r['errors'] for r in rows)
    return {'recordings': len(rows), 'speakers': len({r['speaker'] for r in rows}),
            'exact': sum(r['errors'] == 0 for r in rows), 'errors': errors,
            'reference_jamo': den, 'jer': errors / den if den else None}


def aggregate(records, training_tokens=None, training_positions=None):
    totals, wrong, deleted, inserted, confusion, positions = (Counter() for _ in range(6))
    position_wrong, position_deleted = Counter(), Counter()
    analyzed = []
    for record in records:
        reference, events, changed = align(record['surface'], record['ids'])
        if 'reference_ids' in record:
            assert [u['token'] for u in reference] == [labels.TOKENS[i] for i in record['reference_ids'] if i != labels.BOUNDARY]
        for i, unit in enumerate(reference):
            token, pos = unit['token'], unit['position']
            totals[token] += 1
            positions[(pos, token)] += 1
            if i not in changed:
                confusion[(token, token)] += 1
        for event in events:
            ref, hyp, pos = event['reference'], event['hypothesis'], event['position']
            confusion[(ref, hyp)] += 1
            if event['op'] == 'insert':
                inserted[hyp] += 1
            else:
                wrong[ref] += 1
                position_wrong[(pos, ref)] += 1
                if event['op'] == 'delete':
                    deleted[ref] += 1
                    position_deleted[(pos, ref)] += 1
        analyzed.append({**record, 'events': events, 'errors': len(events), 'reference_jamo': len(reference)})
    by_token = []
    for token in labels.TOKENS[2:]:
        count = totals[token]
        by_token.append({'token': token, 'type': 'vowel' if token in labels.MEDIALS else 'consonant',
                         'references': count, 'correct': count - wrong[token],
                         'substitutions': wrong[token] - deleted[token], 'deletions': deleted[token],
                         'insertions': inserted[token], 'reference_error_rate': wrong[token] / count if count else None,
                         'training_references': training_tokens[token] if training_tokens is not None else None})
    by_position = []
    for (pos, token), count in sorted(positions.items()):
        bad = position_wrong[(pos, token)]
        by_position.append({'position': pos, 'token': token, 'references': count,
                            'substitutions': bad - position_deleted[(pos, token)],
                            'deletions': position_deleted[(pos, token)], 'reference_error_rate': bad / count,
                            'training_references': training_positions[(pos, token)] if training_positions is not None else None})
    types = {}
    for kind in ('consonant', 'vowel'):
        items = [r for r in by_token if r['type'] == kind]
        denominator = sum(r['references'] for r in items)
        types[kind] = {'references': denominator,
                       'substitutions': sum(r['substitutions'] for r in items),
                       'deletions': sum(r['deletions'] for r in items),
                       'insertions': sum(r['insertions'] for r in items),
                       'reference_error_rate': sum(r['substitutions'] + r['deletions'] for r in items) / denominator if denominator else None}
    for token in totals:
        assert sum(n for (ref, _), n in confusion.items() if ref == token) == totals[token]
    return {'metrics': metrics(analyzed), 'by_type': types, 'by_token': by_token,
            'by_reference_position': by_position,
            'confusion': [{'reference': ref, 'hypothesis': hyp, 'count': n}
                          for (ref, hyp), n in sorted(confusion.items(), key=lambda item: -item[1])],
            'common_errors': [{'reference': ref, 'hypothesis': hyp, 'count': n}
                              for (ref, hyp), n in confusion.most_common() if ref != hyp][:40],
            'error_types': dict(Counter(e['op'] for r in analyzed for e in r['events'])),
            'rows': analyzed}


def human_records():
    selection = load(WORK / 'label-comparison-20260927/selection.json')
    baseline = load(WORK / 'label-comparison-20260927/baseline-evaluation.json')
    records = []
    for group in ('train', 'valid'):
        by_id = {r['id']: r for r in baseline['reviewed_' + group]}
        for r in selection[group + '_reviewed']:
            records.append({'id': r['id'], 'speaker': r['speaker'], 'surface': r['realized_surface'],
                            'ids': by_id[r['id']]['ids'], 'text': by_id[r['id']]['text'],
                            'cohort': 'inline_' + group, 'seconds': r['samples'] / 16000})
    predictions = {r['id']: r for r in baseline['diagnostics']}
    for r in selection['diagnostic_entries']:
        if not r['human_target']:
            continue
        prediction = predictions[r['id']]
        records.append({'id': r['id'], 'speaker': r['row']['speaker'], 'surface': r['human_target'],
                        'ids': prediction['ids'], 'text': prediction['text'],
                        'cohort': r['cohort'], 'seconds': r['row']['seconds']})
    return records


def main():
    prior = load(OUT / 'domain-comparison.json')
    assert prior['complete']
    assert prior['checkpoint_sha256'] == '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'
    manifests = {}
    for split in ('train', 'valid', 'test'):
        path = WORK / f'manifest-{split}.jsonl'
        assert digest(path) == prior['manifest_sha256'][split]
        manifests[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    print('Verified all three original manifests.', flush=True)
    training_tokens, training_positions, training_words = Counter(), Counter(), Counter()
    for row in manifests['train']:
        for unit in units(row['surface']):
            training_tokens[unit['token']] += 1
            training_positions[(unit['position'], unit['token'])] += 1
        training_words.update(row['surface'].split())
    print('Counted original training jamo exposures.', flush=True)
    valid = {r['wav']: r for r in manifests['valid']}
    records = []
    seen = set()
    for r in prior['entries']:
        if r['split'] != 'valid':
            continue
        wav = r['id'].removeprefix('valid:')
        manifest = valid[wav]
        assert wav not in seen
        seen.add(wav)
        assert manifest['pcm_sha256'] == r['pcm_sha256']
        assert manifest['label_ids'] == r['reference_ids']
        assert manifest['surface'] == r['reference_surface']
        words = manifest['surface'].split()
        records.append({'id': r['id'], 'speaker': r['speaker'], 'age': manifest['age'],
                        'seconds': r['seconds'], 'duration_bucket': r['duration_bucket'],
                        'surface': manifest['surface'], 'transcript': manifest['text'],
                        'reference_ids': r['reference_ids'], 'ids': r['ids'], 'text': r['text'],
                        'word_count': len(words), 'has_unseen_surface_word': any(training_words[w] == 0 for w in words),
                        'g2p_changed_text': manifest['surface'] != manifest['text'],
                        'pcm_sha256': r['pcm_sha256']})
    assert set(valid) == seen and len(records) == 1500
    validation = aggregate(records, training_tokens, training_positions)
    assert validation['metrics']['errors'] == prior['summary']['valid']['all_selected']['errors'] == 978
    assert validation['metrics']['reference_jamo'] == 33182
    rows = validation['rows']
    groups = {}
    for key in ('age', 'duration_bucket', 'g2p_changed_text', 'has_unseen_surface_word'):
        groups[key] = {str(value): metrics([r for r in rows if r[key] == value]) for value in sorted({r[key] for r in rows})}
    groups['single_word'] = {str(value): metrics([r for r in rows if (r['word_count'] == 1) == value]) for value in (False, True)}
    speakers = defaultdict(list)
    for row in rows:
        speakers[row['speaker']].append(row)
    groups['by_speaker'] = [{'speaker': speaker, **metrics(items)} for speaker, items in speakers.items()]
    human = aggregate(human_records(), training_tokens, training_positions)
    human['by_cohort'] = {kind: metrics([r for r in human['rows'] if r['cohort'] == kind]) for kind in sorted({r['cohort'] for r in human['rows']})}
    report = {
        'complete': True, 'checkpoint_sha256': prior['checkpoint_sha256'],
        'prediction_source_sha256': digest(OUT / 'domain-comparison.json'),
        'original_manifests_sha256': prior['manifest_sha256'],
        'scope': 'Full 1500 original validation recordings plus 27 previously judged clips; all 40 model jamo outputs. No new training or test evaluation.',
        'limitations': [
            'Validation targets are transcript plus project G2P, not independently listened phonetic truth. These are reference discrepancies, not verified child pronunciation errors.',
            'Human labels are provisional spelling choices. Ambiguous and approximate free-text reviews are excluded from exact scoring.',
            'Position attribution uses one deterministic minimum-edit textual alignment, not acoustic alignment. Alternative alignments can change individual attribution.',
            'Jamo ㅇ represents both silent initial and nasal final; a pooled token rate must not be called one phoneme error rate.',
            'Training exposure, word coverage and demographic associations are not randomized causes.',
            'Frozen original batch evaluation has 978 edits. The later smaller-batch baseline has 979; do not mix prediction provenance.'
        ],
        'validation': validation, 'strata': groups, 'human_judged': human,
        'dataset_exposure': prior['exposure'],
        'highest_error_recordings': sorted(rows, key=lambda r: -r['errors'])[:30],
        'training_references': dict(training_tokens),
        'training_original_rows': len(manifests['train']),
    }
    dump(OUT / 'global-jamo-error-audit.json', report)
    dump(ROOT / 'docs/data/child-jamo-global-error-audit-20260927.json', report)
    print(json.dumps({'metrics': validation['metrics'], 'types': validation['by_type'],
                      'top_errors': validation['common_errors'][:15],
                      'weak_tokens': sorted([r for r in validation['by_token'] if r['references'] >= 30], key=lambda r: -r['reference_error_rate'])[:12],
                      'strata': {k: v for k, v in groups.items() if k != 'by_speaker'},
                      'human': human['metrics']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
