"""Audit human answers against frozen evidence; never manufacture phone labels.

The actual answer DB is read in one read-only transaction. QA databases are
never searched. Notes are preserved literally, without G2P or error counting.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_finetune'))
import labels
from common import dump, digest, load

DIRECTORY = ROOT / '.venv/child-jamo-web-20260916/dataset-audit-20260928'
EXPECTED_SELECTION = 'fe1a0c71f9a90300507b488dd147be98525b29d8bfea9c86f23088ff5d54ba15'
EXPECTED_MODEL = '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'


def pcm_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype='<f4').tobytes()).hexdigest()


def signal_stats(wave):
    if len(wave) == 0:
        return {'samples': 0, 'seconds': 0, 'nonzero_samples': 0, 'peak': 0, 'rms': 0}
    return {'samples': len(wave), 'seconds': len(wave) / 16000,
            'nonzero_samples': int(np.count_nonzero(wave)),
            'peak': float(np.abs(wave).max()),
            'rms': float(np.sqrt(np.mean(wave.astype('float64') ** 2)))}


def main():
    assert digest(DIRECTORY / 'selection.json') == EXPECTED_SELECTION
    definitions = load(DIRECTORY / 'review.json')
    by_id = {r['id']: r for r in definitions['items']}
    evidence = load(DIRECTORY / 'evidence.json')
    baseline = load(DIRECTORY / 'baseline-predictions.json')
    assert baseline['checkpoint_sha256'] == EXPECTED_MODEL
    assert baseline['selection_sha256'] == EXPECTED_SELECTION and not baseline['selection_modified']
    predictions = {r['number']: r for r in baseline['predictions']}
    db_path = DIRECTORY / 'answers.sqlite3'
    protected = [DIRECTORY / p for p in ('selection.json', 'review.json', 'evidence.json',
                  'baseline-predictions.json', 'answers.sqlite3')]
    protected += [ROOT / '.venv/hubert-child-20260914/manifest-train.jsonl']
    before_hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        metadata = {r['key']: r['value'] for r in db.execute('SELECT * FROM metadata')}
        history = [dict(r) for r in db.execute('SELECT * FROM answers ORDER BY revision')]
        reveals = {r['item_id']: dict(r) for r in db.execute('SELECT * FROM reveals')}
    assert metadata['review_sha256'] == digest(DIRECTORY / 'review.json')
    latest = {}
    for answer in history:
        assert answer['item_id'] in by_id
        assert answer['binding'] == by_id[answer['item_id']]['binding']
        latest[answer['item_id']] = answer
    assert set(latest) == set(by_id), 'Review incomplete; do not report it as completed'
    snapshot = {'captured_at_utc': datetime.now(timezone.utc).isoformat(),
                'database': str(db_path.relative_to(ROOT)), 'metadata': metadata,
                'answer_history': history, 'reveals': list(reveals.values())}
    snapshot_name = f'answer-snapshot-revision-{max(a["revision"] for a in history):03d}.json'
    snapshot_path = DIRECTORY / snapshot_name
    if snapshot_path.exists():
        existing = load(snapshot_path)
        assert existing['answer_history'] == history and existing['reveals'] == snapshot['reveals']
    else:
        dump(snapshot_path, snapshot)

    rows = []
    for source in evidence:
        number = source['number']
        definition, answer, reveal = by_id[str(number)], latest[str(number)], reveals[str(number)]
        prediction = predictions[number]
        assert definition['binding'] == source['binding'] == prediction['binding'] == reveal['binding']
        allowed = {c['value']: c['label'] for c in definition['reference']['choices']}
        assert answer['choice'] in allowed
        assert reveal['at'] <= answer['at']
        source_path = DIRECTORY / source['source_json']['local_copy']
        assert digest(source_path) == source['source_json']['payload_sha256']
        raw = json.loads(source_path.read_text(encoding='utf-8-sig'))
        row = source['row']
        assert labels.targets(raw['Transcription']['LabelText']) == (row['text'], row['surface'], row['label_ids'])
        assert raw['File']['FileName'] == row['wav']
        assert all(source['source_checks'].values())
        clips = {}
        for key in ('train', 'full'):
            clip = definition[key]
            path = DIRECTORY / 'audio' / clip['file']
            assert digest(path) == clip['sha256']
            wave, rate = sf.read(path, dtype='float32')
            assert rate == 16000 and wave.ndim == 1
            assert pcm_hash(wave) == clip['pcm_sha256']
            clips[key] = wave
        full, train = clips['full'], clips['train']
        assert pcm_hash(train) == row['pcm_sha256'] and len(train) == row['samples']
        start, end = [round(t * 16000) for t in source['crop_seconds']]
        assert np.array_equal(full[start:end], train)
        prefix, suffix = full[:start], full[end:]
        removed = np.concatenate((prefix, suffix))
        whole = np.array_equal(full, train)
        zero_only = bool(len(removed) and not np.count_nonzero(removed))
        assert whole == source['training_audio_is_full_source']
        role = source['choice_roles'].get(answer['choice'])
        if role == 'training_surface':
            judgment = 'training_form_selected'
        elif answer['choice'] in ('different', 'other') or role == 'hypothetical_contrast':
            judgment = 'difference_reported'
        else:
            judgment = answer['choice']
        physical_crop = 'full_source' if whole else 'digital_zero_only_removed' if zero_only else 'nonzero_signal_removed_unclassified'
        g2p_changed = row['text'] != row['surface']
        if judgment != 'difference_reported':
            origin = 'no_new_origin_claim'
        elif not (whole or zero_only) and answer['crop_relation'] == 'cut':
            origin = 'target_speech_crop_loss_reported_requires_content_verification'
        elif not (whole or zero_only) and answer['crop_relation'] == 'same':
            origin = ('difference_persists_in_original_source_vs_g2p_unresolved' if g2p_changed
                      else 'difference_persists_in_original_unchanged_transcript_reported')
        elif not (whole or zero_only):
            origin = 'crop_contribution_unresolved'
        elif g2p_changed:
            origin = 'speech_removal_excluded_source_vs_g2p_unresolved'
        else:
            origin = 'source_pair_difference_reported_no_text_change_or_speech_removal'
        model_agrees = prediction['prediction']['token_ids'] == row['label_ids']
        assert model_agrees == prediction['training_target_agreement']
        # This string comparison is explicitly NOT a phonetic accuracy metric.
        note_literal_match = (bool(reveal['first_heard']) and
                              ''.join(reveal['first_heard'].split()) == ''.join(prediction['prediction']['text'].split()))
        rows.append({'number': number, 'group': source['group'],
                     'stratum': 'random' if source['group'].startswith('random:') else 'focused',
                     'wav': row['wav'], 'speaker': row['speaker'], 'binding': source['binding'],
                     'raw_transcript': raw['Transcription']['LabelText'], 'transcript': row['text'],
                     'training_surface': row['surface'], 'label_ids': row['label_ids'],
                     'first_heard_unedited': reveal['first_heard'], 'first_heard_at': reveal['at'],
                     'answer': answer, 'selected_choice_label': allowed[answer['choice']],
                     'choice_role': role, 'judgment': judgment,
                     'question_scope': definition['reference']['kind'], 'contrast': source.get('contrast'),
                     'model_text': prediction['prediction']['text'],
                     'model_matches_training_target': model_agrees,
                     'model_text_equals_first_note_ignoring_spaces_only': note_literal_match,
                     'g2p_changes_transcript': g2p_changed, 'source_json_path': str(source_path.relative_to(ROOT)),
                     'crop': {'start_seconds': start / 16000, 'end_seconds': end / 16000,
                              'full_seconds': len(full) / 16000, 'training_seconds': len(train) / 16000,
                              'physical_category': physical_crop, 'removed_prefix': signal_stats(prefix),
                              'removed_suffix': signal_stats(suffix), 'removed_all': signal_stats(removed),
                              'retained_first_250ms': signal_stats(train[:4000]),
                              'exact_source_slice': True, 'human_full_audio_comparison': answer['crop_relation'] or None},
                     'origin_status': origin, 'new_phonetic_target': None})

    differences = [r for r in rows if r['judgment'] == 'difference_reported']
    pending_original_checks = [r['number'] for r in differences
                               if r['origin_status'] == 'crop_contribution_unresolved' and not r['answer']['crop_relation']]
    source_pair_differences = [r['number'] for r in differences if r['origin_status'] in (
        'source_pair_difference_reported_no_text_change_or_speech_removal',
        'difference_persists_in_original_unchanged_transcript_reported')]
    prior_path = ROOT / 'docs/data/child-jamo-label-review-20260927.json'
    prior = load(prior_path)
    assert prior['complete']
    assert not ({r['wav'] for r in rows} & {r['wav'] for r in prior['answered']})
    previous_differences = prior['human_reported_difference_numbers']
    checks = {'all_20_answers_bound_to_frozen_questions': True, 'original_json_hashes_and_targets_verified': 20,
              'served_train_and_full_wav_hashes_verified': 40, 'training_pcm_verified': 20,
              'zero_or_whole_sample_claims_exact_not_energy_threshold': True,
              'qa_database_used': False, 'history_preserved': True,
              'protected_files_unchanged': all(digest(ROOT / name) == sha for name, sha in before_hashes.items())}
    assert checks['protected_files_unchanged']
    report = {'complete': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
              'answer_snapshot': str(snapshot_path.relative_to(ROOT)), 'answer_snapshot_sha256': digest(snapshot_path),
              'review_sha256': metadata['review_sha256'], 'selection_sha256': EXPECTED_SELECTION,
              'checkpoint_sha256': EXPECTED_MODEL, 'answered': len(rows), 'answer_history_entries': len(history),
              'counts': dict(Counter(r['judgment'] for r in rows)),
              'by_sampling_group': {g: dict(Counter(r['judgment'] for r in rows if r['stratum'] == g)) for g in ('random', 'focused')},
              'difference_numbers': [r['number'] for r in differences],
              'differences_matching_model_target': [r['number'] for r in differences if r['model_matches_training_target']],
              'differences_with_model_literal_first_note_agreement_not_phonetic_certification': [r['number'] for r in differences if r['model_text_equals_first_note_ignoring_spaces_only']],
              'reported_differences_with_no_nonzero_samples_removed': [r['number'] for r in differences if r['crop']['physical_category'] != 'nonzero_signal_removed_unclassified'],
              'reported_differences_with_possible_crop_contribution': [r['number'] for r in differences if r['origin_status'] == 'crop_contribution_unresolved'],
              'source_pair_difference_reported_without_g2p_text_change': source_pair_differences,
              'original_comparison_completed_numbers': [r['number'] for r in rows if r['answer']['crop_relation']],
              'pending_original_comparisons': pending_original_checks,
              'first_note_details_not_certified_by_selected_answer': [2, 8, 9],
              'note_caveat': 'All first-impression text is approximate human spelling. Preserve the selected judgment; do not apply G2P to notes, count phone errors from them, or infer all segments match from a focused contrast choice.',
              'previous_audit': {'path': str(prior_path.relative_to(ROOT)), 'sha256': digest(prior_path),
                                 'questions': len(prior['answered']), 'reported_difference_numbers': previous_differences,
                                 'new_sample_overlap': 0, 'combined_distinct_questions': len(prior['answered']) + len(rows),
                                 'combined_reported_differences': len(previous_differences) + len(differences),
                                 'population_rate_estimate': None},
              'next_recommendation': {'priority': ('Resolve the pending original/cropped audio comparisons before assigning origin.' if pending_original_checks else
                                                   'Source/crop follow-up is complete. Curate actual-pronunciation evaluation labels and matched normal/variant contrasts in unseen speakers before another label-only training comparison.'),
                                      'already_reviewed_difference_items_to_curate': [3, 4, 6, 10, 15],
                                      'minimal_additional_listening': pending_original_checks,
                                      'unseen_speaker_evaluation_required': True,
                                      'prior_6_train_4_valid_trial_generalization_improvement': False,
                                      'new_training_started': False},
              'limitations': ['This is a small stratified and contrast-enriched training sample, not a population error-rate estimate.',
                              'Model-target agreement and note string agreement are not certified phonetic accuracy.',
                              'Digital-zero removal excludes waveform speech deletion; it does not exclude a change in model context.',
                              'A large removed signal is not proof of cropped target speech; it may be background or another utterance.',
                              'A same-in-original judgment means the reported discrepancy persists before cropping; it does not identify the removed signal or certify every crop boundary.',
                              'These items were used by the baseline model in training; they cannot establish unseen-speaker generalization.',
                              'Dataset suitability for pronunciation preservation is distinct from a supplier transcription-policy violation.'],
              'checks': checks, 'protected_file_hashes': before_hashes, 'rows': rows,
              'labels_modified': False, 'model_modified': False, 'training_started': False}
    latest_path = DIRECTORY / 'analysis-20260929.json'
    if latest_path.exists():
        previous = load(latest_path)
        previous_revision = max(r['answer']['revision'] for r in previous['rows'])
        previous_archive = DIRECTORY / f'analysis-revision-{previous_revision:03d}.json'
        if not previous_archive.exists():
            previous_archive.write_bytes(latest_path.read_bytes())
    dump(latest_path, report)
    current_archive = DIRECTORY / f'analysis-revision-{max(a["revision"] for a in history):03d}.json'
    if not current_archive.exists():
        current_archive.write_bytes(latest_path.read_bytes())
    dump(ROOT / 'docs/data/child-jamo-dataset-audit-review-20260929.json', report)
    print(json.dumps({k: report[k] for k in ('counts', 'by_sampling_group', 'difference_numbers',
                                           'differences_matching_model_target',
                                           'reported_differences_with_no_nonzero_samples_removed',
                                           'reported_differences_with_possible_crop_contribution',
                                           'source_pair_difference_reported_without_g2p_text_change',
                                           'pending_original_comparisons', 'checks')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
