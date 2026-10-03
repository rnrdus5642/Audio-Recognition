"""Bind five relabel questions and ten fresh, matched development recordings.

Selection and proposed forms are frozen before reading selected audio. None of
the proposed forms is a human label until explicitly chosen for the WHOLE clip.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import soundfile as sf

from inventory_phonetic import DEST, NUMBERS, OLD, ROOT, SEED, WORK
from prepare import collect_wavs, pcm_hash, read_verified_member
from common import RAW, digest, dump, load
from workspace_paths import physical_path
import audio
import catalog
import labels

VERSION = 'whole-form-adjudication-20260929-v1'
MODEL_SHA = '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'
FORMS = {
    3: ['반가워요', '방가워요', '반가워우요', '방가워우요'],
    4: ['배콰저메 가요', '배콰강에 가요'],
    6: ['고양이가 조라요', '고양이가 조아요'],
    10: ['장미가 예뻐요', '감미가 예뻐요', '강미가 예뻐요', '잠미가 예뻐요'],
    15: ['기뻐써요', '그뻐써요', '굳뻗써요'],
}
GENERAL = [
    ('ambiguous', '둘 이상으로 들려 하나로 확정하기 어려움'),
    ('partial', '일부 소리는 알겠지만 문장 전체 표기는 확정하기 어려움'),
    ('other', '보기와 다른 소리로 들림'),
    ('unusable', '잡음·겹침·잘림·말소리 없음으로 판단하기 어려움'),
]


def key(value, purpose):
    return hashlib.sha256(f'{SEED}:{purpose}:{value}'.encode()).hexdigest()


def freeze(path, value):
    if path.exists():
        if load(path) != value:
            raise ValueError(f'Frozen file differs: {path.name}')
    else:
        dump(path, value)


def select(inventory):
    excluded = set(collect_wavs(load(OLD / 'previous-reviews.json')))
    excluded.update(e['row']['wav'] for e in load(OLD / 'evidence.json'))
    items, speakers, counts = [], set(), {}
    for case in NUMBERS:
        anchor = inventory['anchors'][str(case)]
        items.append({'role': 'relabel_train', 'case': case, 'wav': anchor['row']['wav'],
                      'speaker': anchor['row']['speaker'], 'anchor': anchor})
        speakers.add(anchor['row']['speaker'])
    for case in NUMBERS:
        pool = [r for r in inventory['pool'] if r['case'] == case
                and r['speaker_split'] == 'valid' and r['exact_source_transcript']
                and not r['has_inline_notation'] and not r['speaker_used_in_baseline_training']
                and not r['recording_used_in_baseline_any_split'] and r['index']['wav'] not in excluded]
        counts[str(case)] = len(pool)
        chosen = 0
        for record in sorted(pool, key=lambda r: r['sort_key']):
            row = record['index']
            if row['speaker'] in speakers:
                continue
            items.append({'role': 'matched_development', 'case': case,
                          'wav': row['wav'], 'speaker': row['speaker'], 'candidate': record})
            speakers.add(row['speaker'])
            chosen += 1
            if chosen == 2:
                break
        if chosen != 2:
            raise ValueError(f'Insufficient independent speakers for case {case}')
    items.sort(key=lambda r: key(r['wav'], 'display'))
    return {'version': VERSION, 'seed': SEED, 'inventory_sha256': digest(DEST / 'inventory.json'),
            'previous_review_sha256': digest(OLD / 'review.json'),
            'previous_analysis_sha256': digest(OLD / 'analysis-revision-021.json'),
            'baseline_checkpoint_sha256': MODEL_SHA,
            'labels_contract_sha256': labels.contract_sha256(),
            'manifest_sha256': {s: digest(WORK / f'manifest-{s}.jsonl') for s in ('train', 'valid', 'test')},
            'forms_are_unconfirmed_hypotheses': True, 'forms': {str(k): v for k, v in FORMS.items()},
            'new_clip_selection_used_predictions_or_listening': False,
            'anchor_selection': 'Five previously reported differences; targeted follow-up, not a prevalence sample.',
            'new_clip_selection': 'Two per exact normalized transcript, ordered by fixed hash; globally unique speakers. No replacement after hearing difficult clips.',
            'eligible_exact_transcript_counts': counts, 'items': items,
            'limitations': [
                'Held-out here means absent from this baseline fine-tuning TRAIN split, not unseen in encoder pretraining.',
                'New recordings are absent from baseline train/valid/test. Their speakers have other recordings in baseline validation; this is a development comparison, not an untouched final test.',
                'Same corpus transcript does not establish what sound was produced or guarantee a normal/variant pair.',
                'Proposed alternatives use old approximate notes as hypotheses, never as accepted sound labels.',
                'Whole-form choices are single-listener broad Hangul/jamo judgments, not expert narrow phonetic annotation.',
            ]}


def locations(names):
    """Read existing catalogues; don't load millions of offsets into memory."""
    found, bindings = {}, {}
    for table in sorted((WORK / 'catalog').glob('*.tar.csv')):
        selected = []
        with table.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                if row['wav'] in names:
                    selected.append(row)
        if not selected:
            continue
        info = load(table.with_suffix('.json'))
        path = physical_path(info['archive']['path'])
        assert catalog.identity(path) == info['archive'], path
        assert digest(table) == info['catalog_sha256'], table
        bindings[path.name] = info
        for row in selected:
            assert row['wav'] not in found, row['wav']
            found[row['wav']] = {'archive': row['archive'], 'path': path,
                                 'offset': int(row['offset']), 'size': int(row['size'])}
    assert found.keys() == names, names - found.keys()
    return found, bindings


def read_source_labels(locations_by_wav):
    groups = {}
    for wav, entry in locations_by_wav.items():
        name = entry['archive'][:1] + 'L' + entry['archive'][2:]
        groups.setdefault(name.lower(), set()).add(wav)
    archives = {}
    for path in RAW.rglob('*.tar'):
        if path.name.lower() in groups:
            assert path.name.lower() not in archives
            archives[path.name.lower()] = path
    assert archives.keys() == groups.keys()
    result = {}
    for name, wanted in groups.items():
        path = archives[name]
        print('Verifying source labels:', path.name, flush=True)
        with tarfile.open(path, 'r:') as tar:
            for member in tar:
                wav = Path(member.name).with_suffix('.wav').name
                if not member.isfile() or not member.name.lower().endswith('.json') or wav not in wanted:
                    continue
                assert wav not in result, wav
                raw = tar.extractfile(member).read()
                result[wav] = {'blob': raw, 'archive': path.name, 'member': member.name,
                               'offset': member.offset_data, 'size': member.size,
                               'payload_sha256': hashlib.sha256(raw).hexdigest()}
            assert wanted <= result.keys(), wanted - result.keys()
    return result


def write_audio(wave, filename):
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format='WAV', subtype='FLOAT')
    blob = buffer.getvalue()
    restored, rate = sf.read(io.BytesIO(blob), dtype='float32')
    assert rate == 16000 and np.array_equal(wave, restored)
    (DEST / 'audio' / filename).write_bytes(blob)
    return {'file': filename, 'sha256': hashlib.sha256(blob).hexdigest(),
            'pcm_sha256': pcm_hash(wave), 'seconds': len(wave) / 16000}


def main():
    if (DEST / 'answers.sqlite3').exists():
        raise RuntimeError('An active answer database exists; refuse to regenerate questions.')
    inventory = load(DEST / 'inventory.json')
    plan = select(inventory)
    # BEFORE waveform or selected JSON payload access.
    freeze(DEST / 'selection.json', plan)
    for name in ('audio', 'source-json'):
        (DEST / name).mkdir(exist_ok=True)
    new_names = {r['wav'] for r in plan['items'] if r['role'] == 'matched_development'}
    new_locations, archive_bindings = locations(new_names)
    sources = read_source_labels(new_locations)
    old_review = {i['number']: i for i in load(OLD / 'review.json')['items']}
    items, evidence = [], []
    for number, selected in enumerate(plan['items'], 1):
        case, wav = selected['case'], selected['wav']
        if selected['role'] == 'relabel_train':
            old = selected['anchor']
            row = dict(old['row'])
            source = dict(old['source_json'])
            blob = (OLD / source['local_copy']).read_bytes()
            assert hashlib.sha256(blob).hexdigest() == source['payload_sha256']
            clip = old_review[case]
            for name in ('train', 'full'):
                assert digest(OLD / 'audio' / clip[name]['file']) == clip[name]['sha256']
            wave = audio.decode((OLD / 'audio' / clip['train']['file']).read_bytes())
            full = audio.decode((OLD / 'audio' / clip['full']['file']).read_bytes())
            assert pcm_hash(wave) == row['pcm_sha256'] and len(wave) == row['samples']
            source_checks = dict(old['source_checks'])
            policy = old['crop_policy']
        else:
            index = selected['candidate']['index']
            source = dict(sources[wav])
            blob = source.pop('blob')
            raw = json.loads(blob.decode('utf-8-sig'))
            normalized, surface, ids = labels.targets(raw['Transcription']['LabelText'])
            source_checks = {
                'filename': raw['File']['FileName'] == wav,
                'length': float(raw['File']['FileLength']) == float(index['seconds']),
                'start': str(raw['Miscellaneous_Info']['SpeechStart']) == index['speech_start'],
                'end': str(raw['Miscellaneous_Info']['SpeechEnd']) == index['speech_end'],
                'transcript': raw['Transcription']['LabelText'] == index['text'],
                'normalized_transcript': normalized == inventory['anchors'][str(case)]['row']['text'],
                'age': str(raw['Speaker']['Age']) == index['age'],
                'speaker': hashlib.sha256(str(raw['Basic']['NumberOfSpeaker']).upper().encode()).hexdigest()[:12] == index['speaker'],
            }
            assert all(source_checks.values()), (wav, source_checks)
            loc = new_locations[wav]
            # Verify TAR member header and content identity, even for fresh files.
            raw_wav = audio.payload(loc['archive'], loc['offset'], loc['size'])
            source_hash = hashlib.sha256(raw_wav).hexdigest()
            verified, member = read_verified_member(loc['path'], loc['offset'], loc['size'], source_hash, wav)
            full = audio.decode(verified)
            assert len(full) / 16000 == float(index['seconds'])
            wave, policy = audio.crop(full, index['speech_start'], index['speech_end'])
            row = {'wav': wav, 'archive': loc['archive'], 'offset': loc['offset'], 'size': loc['size'],
                   'speaker': index['speaker'], 'age': index['age'], 'style': index['style'], 'split': 'valid',
                   'seconds_index': float(index['seconds']), 'speech_start': index['speech_start'],
                   'speech_end': index['speech_end'], 'text': normalized, 'surface': surface, 'label_ids': ids,
                   'samples': len(wave), 'seconds': len(wave) / 16000,
                   'source_sha256': source_hash, 'pcm_sha256': pcm_hash(wave), 'source_wav_member': member}
        (DEST / 'source-json' / f'{number:02d}.json').write_bytes(blob)
        source['local_copy'] = f'source-json/{number:02d}.json'
        train_clip = write_audio(wave, f'{number:02d}-clip.wav')
        same = np.array_equal(wave, full)
        full_clip = train_clip if same else write_audio(full, f'{number:02d}-full.wav')
        forms = sorted(FORMS[case], key=lambda f: key(wav + ':' + f, 'options'))
        choices, candidates = [], {}
        for i, form in enumerate(forms, 1):
            tokens = labels.jamo(form)
            assert tokens
            ids = [labels.VOCAB[t] for t in tokens]
            assert audio.frames(len(wave)) >= labels.ctc_width(ids)
            value = f'whole_form_{i}'
            choices.append({'value': value, 'label': form,
                            'meaning': '문장 전체가 이 소리로 들림'})
            candidates[value] = {'surface': form, 'label_ids': ids,
                                 'is_original_training_target': ids == row['label_ids']}
        assert sum(c['is_original_training_target'] for c in candidates.values()) == 1
        choices += [{'value': value, 'label': label} for value, label in GENERAL]
        reference = {'kind': 'whole_form',
                     'question': '문장 전체가 분명하게 같은 소리로 들리는 표기가 있나요? 일부만 맞거나 둘 다 가능하면 아래의 판단 항목을 골라 주세요.',
                     'choices': choices}
        identity = {'version': VERSION, 'wav': wav, 'role': selected['role'],
                    'clip_pcm_sha256': pcm_hash(wave), 'full_pcm_sha256': pcm_hash(full),
                    'source_json_sha256': source['payload_sha256'], 'reference': reference,
                    'candidates': candidates, 'original_target_ids': row['label_ids']}
        binding = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        items.append({'id': str(number), 'number': number, 'binding': binding,
                      'seconds': len(wave) / 16000, 'train': train_clip, 'full': full_clip,
                      'cropped': not same, 'reference': reference})
        evidence.append({'number': number, 'case': case, 'role': selected['role'], 'row': row,
                         'binding': binding, 'identity': identity, 'candidates': candidates,
                         'source_json': source, 'source_checks': source_checks, 'crop_policy': policy,
                         'served_train': train_clip, 'served_full': full_clip, 'pcm_roundtrip_exact': True})
        print('Prepared', number, selected['role'], wav, flush=True)
    audio.close()
    assert len({e['row']['speaker'] for e in evidence}) == 15
    assert len({e['row']['pcm_sha256'] for e in evidence}) == 15
    freeze(DEST / 'review.json', {'version': VERSION, 'items': items})
    freeze(DEST / 'evidence.json', evidence)
    freeze(DEST / 'comparison-plan.json', {
        'version': VERSION, 'selection_sha256': digest(DEST / 'selection.json'),
        'review_sha256': digest(DEST / 'review.json'), 'baseline_checkpoint_sha256': MODEL_SHA,
        'arms': ['original_manifest_targets', 'listener_confirmed_literal_jamo_targets'],
        'labels_are_broad_single_listener_review': True,
        'g2p_applied_to_confirmed_sound_forms': False,
        'before_training': [
            'All fifteen primary answers saved; bindings and unchanged baseline verified.',
            'At least one train clip explicitly confirmed with a changed whole-form target.',
            'For each evaluable case, independent-speaker development clips explicitly cover BOTH its original target and the same changed form. Same text alone is not contrast coverage.',
            'Ambiguous, partial, other, unusable and crop-loss/uncertain cases produce no training target. Free text is never promoted automatically.',
            'Freeze an identical audio/replay schedule and optimizer settings for both arms, with no development labels entering gradients or checkpoint selection.',
        ],
        'comparison_scope': 'Only contrasts with verified normal AND changed-form development coverage; report in-sample fit separately. Uncovered contrasts cannot support transfer claims.',
        'report': ['per-case original-vs-variant discrimination', 'literal jamo edits against confirmed labels',
                   'normal-pronunciation false-change rate', 'training fit', 'existing validation regression'],
        'not_authorized_by_a_choice': 'No automatic replacement of the microphone model or original manifests.',
        'no_population_or_clinical_claims': True,
    })
    report = {'version': VERSION, 'status': 'awaiting_whole_form_review',
              'questions': len(items), 'training_relabel_questions': 5, 'new_development_questions': 10,
              'unique_speakers': 15, 'new_development_speakers_absent_from_baseline_train': 10,
              'new_recordings_absent_from_all_baseline_manifests': 10,
              'total_clip_seconds': round(sum(i['seconds'] for i in items), 3),
              'cropped_with_full_original_available': sum(i['cropped'] for i in items),
              'selection_sha256': digest(DEST / 'selection.json'), 'review_sha256': digest(DEST / 'review.json'),
              'evidence_sha256': digest(DEST / 'evidence.json'),
              'comparison_plan_sha256': digest(DEST / 'comparison-plan.json'),
              'source_checks_passed': all(all(e['source_checks'].values()) for e in evidence),
              'pcm_roundtrips_verified': 15, 'source_archives': archive_bindings,
              'automatic_target_labels_created': 0, 'training_started': False, 'model_replaced': False,
              'limitations': plan['limitations']}
    freeze(DEST / 'preparation.json', report)
    dump(ROOT / 'docs/data/child-jamo-phonetic-preparation-20260929.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_archives'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
