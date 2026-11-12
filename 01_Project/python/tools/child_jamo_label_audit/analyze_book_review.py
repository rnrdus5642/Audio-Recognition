"""Summarize listening observations without editing training labels."""
from collections import Counter
from datetime import datetime, timezone
import json

from analyze_phonetic import read_snapshot
from prepare_book_review import DEST, ROOT, WORK
from prepare_phonetic import freeze
from common import digest, dump, load
import labels


def main():
    preparation = load(DEST / 'preparation.json')
    for filename, field in [('selection.json', 'selection_sha256'), ('review.json', 'review_sha256'),
                            ('evidence.json', 'evidence_sha256')]:
        assert digest(DEST / filename) == preparation[field]
    plan = load(DEST / 'selection.json')
    assert digest(WORK / 'manifest-train.jsonl') == plan['original_manifest_sha256']
    assert labels.contract_sha256() == plan['labels_contract_sha256']
    snapshot = read_snapshot(DEST)
    replies = {r['id']: r for r in snapshot['items']}
    evidence = load(DEST / 'evidence.json')
    assert set(replies) == {str(e['number']) for e in evidence}
    rows = []
    for entry in evidence:
        for field in ('served_train', 'served_full'):
            assert digest(DEST / 'audio' / entry[field]['file']) == entry[field]['sha256']
        reply = replies[str(entry['number'])]
        assert reply['binding'] == entry['binding']
        answer = reply['answer']
        choice = answer['choice'] if answer else None
        candidate = entry['candidates'].get(choice)
        crop_issue = bool(answer and answer['crop_relation'] in ('cut', 'uncertain'))
        status = ('pending' if not answer else 'crop_uncertainty' if crop_issue else
                  'label_agreement_reported' if candidate and candidate['matches_original_label'] else
                  'label_difference_reported' if candidate else choice)
        rows.append({'number': entry['number'], 'wav': entry['row']['wav'], 'speaker': entry['row']['speaker'],
                     'binding': entry['binding'], 'pcm_sha256': entry['row']['pcm_sha256'],
                     'original_transcript': entry['row']['text'], 'training_surface': entry['row']['surface'],
                     'previously_presented': entry['previously_presented'],
                     'first_impression': (reply['reveal'] or {}).get('first_heard', ''),
                     'answer_revision': answer['revision'] if answer else None, 'choice': choice,
                     'selected_surface': candidate['surface'] if candidate else None,
                     'note': answer['note'] if answer else '', 'status': status,
                     'crop_relation': answer['crop_relation'] if answer else '',
                     'verified_pronunciation_target': None})
    assert read_snapshot(DEST) == snapshot
    revision = snapshot['latest_revision']
    freeze(DEST / f'answer-snapshot-revision-{revision:03d}.json', snapshot)
    report = {'analyzed_at_utc': datetime.now(timezone.utc).isoformat(),
              'review_sha256': snapshot['review_sha256'], 'answer_revision': revision,
              'total': snapshot['total'], 'answered': snapshot['answered'],
              'pending_questions': [r['number'] for r in rows if r['status'] == 'pending'],
              'status_counts': dict(Counter(r['status'] for r in rows)),
              'selected_form_counts': dict(Counter(r['selected_surface'] for r in rows if r['selected_surface'])),
              'reported_difference_questions': [r['number'] for r in rows if r['status'] == 'label_difference_reported'],
              'items': rows, 'training_labels_changed': False, 'model_replaced': False,
              'limits': preparation['limits']}
    dump(DEST / f'analysis-revision-{revision:03d}.json', report)
    dump(DEST / 'analysis-latest.json', report)
    dump(ROOT / 'docs/data/child-jamo-book-review-analysis-20261003.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'items'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
