"""Freeze two additional speakers for the incomplete case-3 contrast.

Only the existing inventory order selects recordings. No prediction or audio
is consulted before selection; neither a normal nor a variant sound is assumed.
The completed fifteen-question bundle and answer database stay untouched.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import prepare_phonetic as source
from analyze_phonetic import read_snapshot
from common import INDEX, SPLITS, digest, dump, load
from prepare import collect_wavs, pcm_hash, read_verified_member
import audio
import labels

DEST = source.DEST.parent / 'phonetic-followup-20261003'
VERSION = 'case3-contrast-followup-20261003-v1'


def write_wave(wave, name):
    stream = io.BytesIO()
    sf.write(stream, wave, 16000, format='WAV', subtype='FLOAT')
    blob = stream.getvalue()
    restored, rate = sf.read(io.BytesIO(blob), dtype='float32')
    assert rate == 16000 and np.array_equal(wave, restored)
    (DEST / 'audio' / name).write_bytes(blob)
    return {'file': name, 'sha256': hashlib.sha256(blob).hexdigest(),
            'pcm_sha256': pcm_hash(wave), 'seconds': len(wave) / 16000}


def main():
    if (DEST / 'answers.sqlite3').exists() or (DEST / 'review.json').exists():
        raise RuntimeError('The follow-up already exists; keep its questions and answers.')
    snapshot = read_snapshot()
    assert snapshot['answered'] == 15 and snapshot['latest_revision'] == 15
    original = load(source.DEST / 'selection.json')
    inventory = load(source.DEST / 'inventory.json')
    assert digest(source.DEST / 'inventory.json') == original['inventory_sha256']
    assert digest(INDEX) == inventory['index_sha256']
    assert digest(SPLITS) == inventory['speaker_split_sha256']
    assert labels.contract_sha256() == original['labels_contract_sha256']
    for split, expected in original['manifest_sha256'].items():
        assert digest(source.WORK / f'manifest-{split}.jsonl') == expected
    excluded = set(collect_wavs(load(source.OLD / 'previous-reviews.json')))
    excluded.update(e['row']['wav'] for e in load(source.OLD / 'evidence.json'))
    excluded.update(e['wav'] for e in original['items'])
    speakers = {e['speaker'] for e in original['items']}
    pool = [r for r in inventory['pool'] if r['case'] == 3 and r['speaker_split'] == 'valid'
            and r['exact_source_transcript'] and not r['has_inline_notation']
            and not r['speaker_used_in_baseline_training']
            and not r['recording_used_in_baseline_any_split']
            and r['index']['wav'] not in excluded and r['index']['speaker'] not in speakers]
    chosen = []
    for row in sorted(pool, key=lambda r: r['sort_key']):
        if row['index']['speaker'] in speakers:
            continue
        chosen.append(row)
        speakers.add(row['index']['speaker'])
        if len(chosen) == 2:
            break
    assert len(chosen) == 2
    plan = {
        'version': VERSION, 'parent_review_sha256': snapshot['review_sha256'],
        'parent_answer_revision': snapshot['latest_revision'],
        'parent_snapshot_sha256': digest(source.DEST / 'answer-snapshot-revision-015.json'),
        'inventory_sha256': original['inventory_sha256'],
        'baseline_checkpoint_sha256': original['baseline_checkpoint_sha256'],
        'labels_contract_sha256': original['labels_contract_sha256'],
        'manifest_sha256': original['manifest_sha256'],
        'case': 3, 'eligible_remaining': len(pool), 'items': chosen,
        'forms': source.FORMS[3],
        'selection': 'Next two eligible unique speakers in the original fixed hash order, excluding all earlier reviewed recordings and all fifteen previous speakers.',
        'purpose': 'Check the missing original-form development control for case 3. Actual sounds remain unknown until review.',
        'selection_used_audio_or_predictions': False,
        'scope': 'Targeted development follow-up; not an untouched final test or a prevalence sample.',
    }
    DEST.mkdir(parents=True, exist_ok=True)
    source.freeze(DEST / 'selection.json', plan)
    for name in ('audio', 'source-json'):
        (DEST / name).mkdir(exist_ok=True)
    locations, bindings = source.locations({r['index']['wav'] for r in chosen})
    raw_labels = source.read_source_labels(locations)
    items, evidence = [], []
    for number, selected in enumerate(chosen, 1):
        index = selected['index']
        wav = index['wav']
        src = dict(raw_labels[wav])
        blob = src.pop('blob')
        raw = json.loads(blob.decode('utf-8-sig'))
        text, surface, ids = labels.targets(raw['Transcription']['LabelText'])
        checks = {
            'filename': raw['File']['FileName'] == wav,
            'length': float(raw['File']['FileLength']) == float(index['seconds']),
            'start': str(raw['Miscellaneous_Info']['SpeechStart']) == index['speech_start'],
            'end': str(raw['Miscellaneous_Info']['SpeechEnd']) == index['speech_end'],
            'transcript': raw['Transcription']['LabelText'] == index['text'],
            'normalized_transcript': text == inventory['anchors']['3']['row']['text'],
            'age': str(raw['Speaker']['Age']) == index['age'],
            'speaker': hashlib.sha256(str(raw['Basic']['NumberOfSpeaker']).upper().encode()).hexdigest()[:12] == index['speaker'],
        }
        assert all(checks.values()), checks
        loc = locations[wav]
        payload = audio.payload(loc['archive'], loc['offset'], loc['size'])
        payload_sha = hashlib.sha256(payload).hexdigest()
        verified, member = read_verified_member(loc['path'], loc['offset'], loc['size'], payload_sha, wav)
        full = audio.decode(verified)
        assert len(full) / 16000 == float(index['seconds'])
        wave, policy = audio.crop(full, index['speech_start'], index['speech_end'])
        row = {'wav': wav, 'archive': loc['archive'], 'offset': loc['offset'], 'size': loc['size'],
               'speaker': index['speaker'], 'age': index['age'], 'style': index['style'], 'split': 'valid',
               'speech_start': index['speech_start'], 'speech_end': index['speech_end'],
               'text': text, 'surface': surface, 'label_ids': ids, 'samples': len(wave),
               'seconds': len(wave) / 16000, 'source_sha256': payload_sha,
               'pcm_sha256': pcm_hash(wave), 'source_wav_member': member}
        src['local_copy'] = f'source-json/{number:02d}.json'
        (DEST / src['local_copy']).write_bytes(blob)
        clip = write_wave(wave, f'{number:02d}-clip.wav')
        same = np.array_equal(wave, full)
        original_clip = clip if same else write_wave(full, f'{number:02d}-full.wav')
        choices, candidates = [], {}
        forms = sorted(source.FORMS[3], key=lambda form: source.key(wav + ':' + form, 'options'))
        for n, form in enumerate(forms, 1):
            token_ids = [labels.VOCAB[t] for t in labels.jamo(form)]
            assert audio.frames(len(wave)) >= labels.ctc_width(token_ids)
            value = f'whole_form_{n}'
            choices.append({'value': value, 'label': form, 'meaning': '문장 전체가 이 소리로 들림'})
            candidates[value] = {'surface': form, 'label_ids': token_ids,
                                 'is_original_training_target': token_ids == ids}
        choices += [{'value': value, 'label': label} for value, label in source.GENERAL]
        reference = {'kind': 'whole_form', 'question': '문장 전체가 분명하게 같은 소리로 들리는 표기가 있나요? 일부만 맞거나 둘 다 가능하면 아래의 판단 항목을 골라 주세요.', 'choices': choices}
        identity = {'version': VERSION, 'wav': wav, 'role': 'matched_development',
                    'clip_pcm_sha256': pcm_hash(wave), 'full_pcm_sha256': pcm_hash(full),
                    'source_json_sha256': src['payload_sha256'], 'reference': reference,
                    'candidates': candidates, 'original_target_ids': ids}
        binding = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        items.append({'id': str(number), 'number': number, 'binding': binding,
                      'seconds': len(wave) / 16000, 'train': clip, 'full': original_clip,
                      'cropped': not same, 'reference': reference})
        evidence.append({'number': number, 'case': 3, 'role': 'matched_development', 'row': row,
                         'binding': binding, 'identity': identity, 'candidates': candidates,
                         'source_json': src, 'source_checks': checks, 'crop_policy': policy,
                         'served_train': clip, 'served_full': original_clip, 'pcm_roundtrip_exact': True})
    audio.close()
    assert len({e['row']['speaker'] for e in evidence}) == 2
    old_pcm = {e['row']['pcm_sha256'] for e in load(source.DEST / 'evidence.json')}
    assert not old_pcm.intersection(e['row']['pcm_sha256'] for e in evidence)
    assert len({e['row']['pcm_sha256'] for e in evidence}) == 2
    assert read_snapshot() == snapshot
    source.freeze(DEST / 'review.json', {'version': VERSION, 'items': items})
    source.freeze(DEST / 'evidence.json', evidence)
    page = (Path(__file__).parent / 'phonetic.html').read_text(encoding='utf-8')
    page = page.replace('들은 소리 확인 · 15개', '들은 소리 확인 · 추가 2개')
    page = page.replace('max="15"', 'max="2"').replace('15개 모두 저장했습니다.', '추가 2개 모두 저장했습니다.')
    page = page.replace('이전에 다르게 들렸던 5개와 다른 화자의 새 녹음 10개입니다.',
                        '다른 화자의 새 녹음 2개입니다.')
    (DEST / 'listen.html').write_text(page, encoding='utf-8')
    report = {'version': VERSION, 'questions': 2, 'parent_answer_revision': 15,
              'selection_sha256': digest(DEST / 'selection.json'),
              'review_sha256': digest(DEST / 'review.json'), 'evidence_sha256': digest(DEST / 'evidence.json'),
              'page_sha256': digest(DEST / 'listen.html'), 'source_archives': bindings,
              'seconds': sum(i['seconds'] for i in items),
              'source_checks_passed': True, 'pcm_roundtrips_verified': 2,
              'training_started': False, 'model_replaced': False,
              'status': 'awaiting_two_additional_whole_form_reviews'}
    source.freeze(DEST / 'preparation.json', report)
    dump(source.ROOT / 'docs/data/child-jamo-phonetic-followup-preparation-20261003.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'source_archives'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
