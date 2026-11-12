"""Read both completed reviews without conflating their question IDs.

Keep the original whole-form comparison gate. Preserve selected spellings
without promoting uncertain listening choices to a verified sound contrast.
"""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from analyze_phonetic import QUALIFICATIONS, assess, read_qualifications, read_snapshot
from inventory_phonetic import DEST as PRIMARY, ROOT, WORK
from common import digest, dump, load
from prepare_phonetic import freeze
import labels

FOLLOWUP = PRIMARY.parent / 'phonetic-followup-20261003'


def main():
    primary = read_snapshot(PRIMARY)
    followup = read_snapshot(FOLLOWUP)
    qualifications = read_qualifications()
    qualification_sha = digest(QUALIFICATIONS)
    assert qualifications['followup_review_sha256'] == followup['review_sha256']
    selection = load(FOLLOWUP / 'selection.json')
    assert primary == load(PRIMARY / 'answer-snapshot-revision-015.json')
    assert primary['latest_revision'] == selection['parent_answer_revision']
    assert digest(PRIMARY / 'answer-snapshot-revision-015.json') == selection['parent_snapshot_sha256']
    assert primary['review_sha256'] == selection['parent_review_sha256']
    assert labels.contract_sha256() == selection['labels_contract_sha256']
    assert digest(WORK / 'main/best.pt') == selection['baseline_checkpoint_sha256']
    for split, expected in selection['manifest_sha256'].items():
        assert digest(WORK / f'manifest-{split}.jsonl') == expected
    for directory in (PRIMARY, FOLLOWUP):
        preparation = load(directory / 'preparation.json')
        for filename, key in [('selection.json', 'selection_sha256'),
                              ('review.json', 'review_sha256'),
                              ('evidence.json', 'evidence_sha256')]:
            assert digest(directory / filename) == preparation[key]
        for entry in load(directory / 'evidence.json'):
            for key in ('served_train', 'served_full'):
                assert digest(directory / 'audio' / entry[key]['file']) == entry[key]['sha256']
    evidence = copy.deepcopy(load(PRIMARY / 'evidence.json'))
    supplemental = load(FOLLOWUP / 'evidence.json')
    combined = copy.deepcopy(primary)
    combined['latest_revision'] = {'primary': primary['latest_revision'],
                                   'followup': followup['latest_revision']}
    identity_map = {str(e['number']): {'bundle': 'primary', 'question': e['number']}
                    for e in evidence}
    for entry, reply in zip(supplemental, followup['items'], strict=True):
        assert str(entry['number']) == reply['id']
        new_number = len(evidence) + 1
        identity_map[str(new_number)] = {'bundle': 'followup', 'question': entry['number']}
        e, r = copy.deepcopy(entry), copy.deepcopy(reply)
        e['number'], r['id'] = new_number, str(new_number)
        evidence.append(e)
        combined['items'].append(r)
    assert len({e['row']['wav'] for e in evidence}) == len(evidence)
    assert len({e['row']['speaker'] for e in evidence}) == len(evidence)
    assert len({e['row']['pcm_sha256'] for e in evidence}) == len(evidence)
    result = assess(evidence, combined, qualifications['qualifications'])
    for item in result['items']:
        item['source_question'] = identity_map[str(item['number'])]

    anchor = next(i for i in result['items'] if i['number'] == 2)
    long_form = anchor['selected_surface']
    short_form = '방가워요'
    long_ids = [labels.VOCAB[t] for t in labels.jamo(long_form)]
    short_ids = [labels.VOCAB[t] for t in labels.jamo(short_form)]
    assert long_form == '방가워우요' and anchor['selected_ids'] == long_ids
    longer = [i['number'] for i in result['items'] if i['role'] == 'matched_development'
              and i['case'] == 3 and i['selected_ids'] == long_ids]
    shorter = [i['number'] for i in result['items'] if i['role'] == 'matched_development'
               and i['case'] == 3 and i['selected_ids'] == short_ids]
    eligible_longer = [i['number'] for i in result['items'] if i['number'] in longer and i['confirmed_ids'] is not None]
    eligible_shorter = [i['number'] for i in result['items'] if i['number'] in shorter and i['confirmed_ids'] is not None]
    focused = {
        'longer_form': long_form, 'shorter_form': short_form,
        'training_question': 2, 'selected_longer_development_questions': longer,
        'selected_shorter_development_questions': shorter,
        'longer_development_questions': eligible_longer,
        'shorter_development_questions': eligible_shorter,
        'selected_forms_present': bool(longer and shorter),
        'observed_contrast_available': bool(anchor['confirmed_ids'] and eligible_longer and eligible_shorter),
        'excluded_for_listener_uncertainty': bool(anchor['qualification_ids']),
        'original_form_control_missing': not any(i['confirmed_ids'] == anchor['original_ids']
                                               for i in result['items']
                                               if i['case'] == 3 and i['role'] == 'matched_development'),
        'scope': 'Different written forms were selected, but the listener subsequently questioned the sound distinction. Preserve the choices without using them as verified pronunciation labels or a training contrast.',
        'limits': 'No independent sound distinction established. Neither model scores nor spellings resolve this uncertainty. The original comparison gate is unchanged.',
    }
    assert read_snapshot(PRIMARY) == primary and read_snapshot(FOLLOWUP) == followup
    revision = followup['latest_revision']
    freeze(FOLLOWUP / f'answer-snapshot-revision-{revision:03d}.json', followup)
    report = {'analyzed_at_utc': datetime.now(timezone.utc).isoformat(),
              'primary_snapshot_sha256': selection['parent_snapshot_sha256'],
              'followup_review_sha256': followup['review_sha256'],
              'followup_evidence_sha256': digest(FOLLOWUP / 'evidence.json'),
              'qualifications_sha256': qualification_sha,
              'question_identity_map': identity_map, **result,
              'focused_realized_form_contrast': focused,
              'model_replaced': False}
    assert digest(QUALIFICATIONS) == qualification_sha
    dump(FOLLOWUP / f'analysis-primary-015-followup-{revision:03d}-qualified-{qualification_sha[:12]}.json', report)
    dump(FOLLOWUP / 'analysis-latest.json', report)
    dump(ROOT / 'docs/data/child-jamo-phonetic-followup-analysis-20261003.json', report)
    print(json.dumps({k: report[k] for k in ['answered', 'latest_revision', 'qualified_comparison_cases',
                     'ready_for_controlled_comparison', 'focused_realized_form_contrast']},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
