"""Read bound answers and measure label/comparison readiness, without training.

Approximate notes are evidence to retain, never a source of target IDs. A whole
form choice is only a single-listener broad transcription, not a clinical label.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from inventory_phonetic import DEST, ROOT, WORK
from common import digest, dump, load
import labels


def read_snapshot(directory=DEST):
    directory = Path(directory)
    review = load(directory / 'review.json')
    review_sha = digest(directory / 'review.json')
    db_path = directory / 'answers.sqlite3'
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        metadata = dict(db.execute('SELECT key,value FROM metadata'))
        assert metadata['review_sha256'] == review_sha
        history = [dict(r) for r in db.execute('SELECT * FROM answers ORDER BY revision')]
        reveals = {r['item_id']: dict(r) for r in db.execute('SELECT * FROM reveals')}
    by_id = {i['id']: i for i in review['items']}
    answers = {}
    for entry in history:
        item = by_id[entry['item_id']]
        assert entry['binding'] == item['binding']
        assert entry['choice'] in {c['value'] for c in item['reference']['choices']}
        assert reveals[item['id']]['binding'] == item['binding']
        answers[item['id']] = entry
    return {'review_sha256': review_sha, 'total': len(by_id), 'answered': len(answers),
            'latest_revision': max((r['revision'] for r in history), default=0),
            'answer_history': history,
            'items': [{'id': i['id'], 'binding': i['binding'], 'answer': answers.get(i['id']),
                       'reveal': reveals.get(i['id'])} for i in review['items']]}


def assess(evidence, snapshot):
    answers = {r['id']: r for r in snapshot['items']}
    assert set(answers) == {str(e['number']) for e in evidence}
    rows = []
    for entry in evidence:
        reply = answers[str(entry['number'])]
        assert reply['binding'] == entry['binding']
        answer = reply['answer']
        candidate = entry['candidates'].get(answer['choice']) if answer else None
        crop_issue = answer and answer['crop_relation'] in ('cut', 'uncertain')
        accepted = candidate if candidate and not crop_issue else None
        if accepted:
            # Literal decomposition only. Never call targets() / G2P on a listener's form.
            tokens = labels.jamo(accepted['surface'])
            assert tokens and [labels.VOCAB[t] for t in tokens] == accepted['label_ids']
        rows.append({'number': entry['number'], 'case': entry['case'], 'role': entry['role'],
                     'wav': entry['row']['wav'], 'speaker': entry['row']['speaker'],
                     'pcm_sha256': entry['row']['pcm_sha256'], 'binding': entry['binding'],
                     'answer_revision': answer['revision'] if answer else None,
                     'choice': answer['choice'] if answer else None,
                     'status': ('pending' if not answer else 'crop_uncertainty_excluded' if crop_issue
                                else 'whole_form_confirmed' if accepted else 'no_whole_form_label'),
                     'original_surface': entry['row']['surface'], 'original_ids': entry['row']['label_ids'],
                     'confirmed_surface': accepted['surface'] if accepted else None,
                     'confirmed_ids': accepted['label_ids'] if accepted else None,
                     'changed_target': bool(accepted and accepted['label_ids'] != entry['row']['label_ids']),
                     'first_impression': (reply.get('reveal') or {}).get('first_heard', ''),
                     'note': answer['note'] if answer else '',
                     'note_used_as_target': False})
    train = [r for r in rows if r['role'] == 'relabel_train']
    development = [r for r in rows if r['role'] == 'matched_development']
    assert not ({r['speaker'] for r in train} & {r['speaker'] for r in development})
    assert not ({r['pcm_sha256'] for r in train} & {r['pcm_sha256'] for r in development})
    coverage = []
    for anchor in train:
        paired = [r for r in development if r['case'] == anchor['case'] and r['confirmed_ids'] is not None]
        original = [r for r in paired if r['confirmed_ids'] == anchor['original_ids']]
        changed = [r for r in paired if anchor['changed_target'] and r['confirmed_ids'] == anchor['confirmed_ids']]
        speakers = {r['speaker'] for r in original + changed}
        ready = bool(anchor['changed_target'] and original and changed and len(speakers) >= 2)
        reasons = []
        if not anchor['changed_target']:
            reasons.append('no_confirmed_changed_training_target')
        if not original:
            reasons.append('no_confirmed_original_form_on_development_speaker')
        if not changed:
            reasons.append('no_confirmed_same_changed_form_on_development_speaker')
        coverage.append({'case': anchor['case'], 'train_question': anchor['number'],
                         'original_form_development_questions': [r['number'] for r in original],
                         'same_changed_form_development_questions': [r['number'] for r in changed],
                         'matched_contrast_ready': ready, 'missing': reasons})
    pending = [r['number'] for r in rows if r['status'] == 'pending']
    qualified = [c['case'] for c in coverage if c['matched_contrast_ready']]
    corrections = [r['number'] for r in train if r['changed_target']]
    ready = not pending and bool(qualified)
    next_action = ('await_whole_form_answers' if pending else
                   'freeze_equal_training_schedule_for_qualified_contrasts' if ready else
                   'obtain_missing_contrast_coverage_before_training' if corrections else
                   'no_confirmed_relabel_intervention_do_not_train')
    return {'total': len(rows), 'answered': len(rows) - len(pending), 'pending_questions': pending,
            'latest_revision': snapshot['latest_revision'],
            'answer_counts': dict(Counter(r['choice'] for r in rows if r['choice'])),
            'confirmed_training_relabel_questions': corrections, 'coverage': coverage,
            'qualified_comparison_cases': qualified, 'ready_for_controlled_comparison': ready,
            'next_action': next_action, 'items': rows,
            'notes_used_as_targets': False, 'g2p_applied_to_confirmed_forms': False,
            'original_manifest_modified': False, 'training_started': False,
            'label_scope': 'single-listener broad Hangul/jamo, explicitly selected whole forms only',
            'evaluation_scope': 'development speakers excluded from baseline training; not an untouched final test',
            'limits': 'Same-prompt selection alone does not verify a pronunciation contrast. Missing variant coverage must stay missing; do not replace it with corpus labels or model predictions.'}


def main():
    preparation = load(DEST / 'preparation.json')
    plan = load(DEST / 'selection.json')
    for name, field in [('selection.json', 'selection_sha256'), ('review.json', 'review_sha256'),
                        ('evidence.json', 'evidence_sha256'), ('comparison-plan.json', 'comparison_plan_sha256')]:
        assert digest(DEST / name) == preparation[field], name
    assert labels.contract_sha256() == plan['labels_contract_sha256']
    assert digest(WORK / 'main/best.pt') == plan['baseline_checkpoint_sha256']
    for split, expected in plan['manifest_sha256'].items():
        assert digest(WORK / f'manifest-{split}.jsonl') == expected
    snapshot = read_snapshot()
    evidence = load(DEST / 'evidence.json')
    for entry in evidence:
        for key in ('served_train', 'served_full'):
            assert digest(DEST / 'audio' / entry[key]['file']) == entry[key]['sha256']
    report = {'analyzed_at_utc': datetime.now(timezone.utc).isoformat(),
              'selection_sha256': preparation['selection_sha256'],
              'review_sha256': preparation['review_sha256'], **assess(evidence, snapshot)}
    revision = snapshot['latest_revision']
    snapshot_path = DEST / f'answer-snapshot-revision-{revision:03d}.json'
    if snapshot_path.exists():
        assert load(snapshot_path) == snapshot
    else:
        dump(snapshot_path, snapshot)
    dump(DEST / f'analysis-revision-{revision:03d}.json', report)
    dump(DEST / 'analysis-latest.json', report)
    dump(ROOT / 'docs/data/child-jamo-phonetic-readiness-20260929.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'items'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
