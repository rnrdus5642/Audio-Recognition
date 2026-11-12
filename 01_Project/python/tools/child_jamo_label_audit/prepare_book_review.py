"""Collect every standalone '책' label in the actual baseline training set.

This is a census of one label, not a search for model disagreements. Freeze
membership before reading audio; keep previously presented files in the set.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from prepare import (MANIFEST_SHA, RAW, ROOT, WEB, WORK, collect_wavs,
                     existing_reviews, pcm_hash, read_verified_member)
from prepare_phonetic import freeze
from common import digest, dump, load
import audio
import labels

DEST = WEB / 'book-label-review-20261003'
VERSION = 'standalone-book-training-census-20261003-v1'
WORD = '책'


def order(value, purpose):
    return hashlib.sha256(f'{VERSION}:{purpose}:{value}'.encode()).hexdigest()


def write_clip(wave, filename):
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format='WAV', subtype='FLOAT')
    blob = buffer.getvalue()
    restored, rate = sf.read(io.BytesIO(blob), dtype='float32')
    assert rate == 16000 and np.array_equal(restored, wave)
    (DEST / 'audio' / filename).write_bytes(blob)
    return {'file': filename, 'sha256': hashlib.sha256(blob).hexdigest(),
            'pcm_sha256': pcm_hash(wave), 'seconds': len(wave) / 16000}


def main():
    if (DEST / 'answers.sqlite3').exists():
        raise RuntimeError('A user answer database exists. Do not regenerate this review.')
    manifest = WORK / 'manifest-train.jsonl'
    assert digest(manifest) == MANIFEST_SHA
    with manifest.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 150253 and all(r['split'] == 'train' for r in rows)
    selected = [r for r in rows if r['text'] == WORD]
    assert {r['wav'] for r in selected} == {r['wav'] for r in rows if r['surface'] == WORD}
    target_ids = [labels.VOCAB[t] for t in labels.jamo(WORD)]
    assert selected and all(r['label_ids'] == target_ids for r in selected)
    assert len({r['wav'] for r in selected}) == len(selected)
    previously_presented, _history = existing_reviews()
    for directory in ('dataset-audit-20260928', 'phonetic-labels-20260929', 'phonetic-followup-20261003'):
        previously_presented.update(collect_wavs(load(WEB / directory / 'evidence.json')))
    selected.sort(key=lambda r: order(r['wav'], 'display'))
    plan = {'version': VERSION, 'original_manifest_sha256': MANIFEST_SHA,
            'labels_contract_sha256': labels.contract_sha256(), 'population': len(rows),
            'criterion': 'Original baseline train manifest: text == 책; verified surface == 책 and IDs == ㅊㅐㄱ. Include every matching recording.',
            'exact_label_count': len(selected),
            'sentence_rows_with_separate_book_token': sum(WORD in r['text'].split() and r['text'] != WORD for r in rows),
            'model_predictions_used': False, 'listening_used_to_select': False,
            'items': [{'number': n, 'row': row, 'previously_presented': row['wav'] in previously_presented}
                      for n, row in enumerate(selected, 1)]}
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'audio').mkdir(exist_ok=True)
    (DEST / 'source-json').mkdir(exist_ok=True)
    freeze(DEST / 'selection.json', plan)

    annotation = load(ROOT / 'docs/data/child-jamo-training-annotation-fields-20260927.json')
    journal = ROOT / annotation['provenance_journal']
    assert digest(journal) == annotation['provenance_journal_sha256']
    needed = {r['wav'] for r in selected}
    sources = {}
    with journal.open(encoding='utf-8') as stream:
        for line in stream:
            entry = json.loads(line)
            if entry['wav'] in needed:
                assert entry['wav'] not in sources
                sources[entry['wav']] = entry
    assert sources.keys() == needed
    wanted = {r['archive'].lower() for r in selected} | {s['archive'].lower() for s in sources.values()}
    archives = {}
    for path in RAW.rglob('*.tar'):
        if path.name.lower() in wanted:
            assert path.name.lower() not in archives
            archives[path.name.lower()] = path
    assert archives.keys() == wanted

    items, evidence = [], []
    for entry in plan['items']:
        number, row = entry['number'], entry['row']
        source = sources[row['wav']]
        raw_blob, source_member = read_verified_member(
            archives[source['archive'].lower()], source['offset'], source['size'],
            source['payload_sha256'], Path(row['wav']).with_suffix('.json').name)
        raw = json.loads(raw_blob.decode('utf-8-sig'))
        text, surface, ids = labels.targets(raw['Transcription']['LabelText'])
        checks = {'source_filename': raw['File']['FileName'] == row['wav'],
                  'source_length': float(raw['File']['FileLength']) == row['seconds_index'],
                  'speech_start': str(raw['Miscellaneous_Info']['SpeechStart']) == str(row['speech_start']),
                  'speech_end': str(raw['Miscellaneous_Info']['SpeechEnd']) == str(row['speech_end']),
                  'normalized_transcript': text == row['text'] == WORD,
                  'g2p_surface': surface == row['surface'] == WORD,
                  'training_ids': ids == row['label_ids'] == target_ids}
        assert all(checks.values()), (number, checks)
        (DEST / 'source-json' / f'{number:02d}.json').write_bytes(raw_blob)
        blob, wav_member = read_verified_member(
            archives[row['archive'].lower()], row['offset'], row['size'], row['source_sha256'], row['wav'])
        full = audio.decode(blob)
        wave, policy = audio.crop(full, row['speech_start'], row['speech_end'])
        assert pcm_hash(wave) == row['pcm_sha256'] and len(wave) == row['samples']
        assert policy == row['span_policy']
        clip = write_clip(wave, f'{number:02d}-train.wav')
        same = np.array_equal(wave, full)
        original = clip if same else write_clip(full, f'{number:02d}-full.wav')

        candidates, choices = {}, []
        for n, form in enumerate(sorted(['책', '짹', '택', '쌕'], key=lambda f: order(row['wav'] + ':' + f, 'choices')), 1):
            value = f'whole_form_{n}'
            choices.append({'value': value, 'label': form})
            candidates[value] = {'surface': form, 'label_ids': [labels.VOCAB[t] for t in labels.jamo(form)],
                                 'matches_original_label': form == WORD}
        choices += [{'value': value, 'label': label} for value, label in [
            ('other', '보기와 다른 소리로 들림'),
            ('ambiguous', '둘 이상으로 들려 하나로 고르기 어려움'),
            ('partial', '일부 소리만 알겠고 단어 전체는 판단하기 어려움'),
            ('no_speech', '말소리가 없음'),
            ('unusable', '잡음·겹침·잘림 등으로 판단하기 어려움')]]
        reference = {'kind': 'whole_form', 'choices': choices,
                     'question': '맞춤법이나 의도한 단어를 추측하지 말고, 실제로 들린 소리를 골라 주세요. 보기는 가능한 예시이며, 다르게 들리면 「보기와 다른 소리」를 고를 수 있습니다.'}
        identity = {'version': VERSION, 'wav': row['wav'], 'train_pcm_sha256': pcm_hash(wave),
                    'full_pcm_sha256': pcm_hash(full), 'source_json_sha256': source['payload_sha256'],
                    'reference': reference, 'original_target_ids': ids}
        binding = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        items.append({'id': str(number), 'number': number, 'binding': binding, 'seconds': len(wave) / 16000,
                      'train': clip, 'full': original, 'cropped': not same, 'reference': reference})
        evidence.append({**entry, 'binding': binding, 'identity': identity, 'candidates': candidates,
                         'source_json': {**source, 'verified_member': source_member,
                                         'local_copy': f'source-json/{number:02d}.json'},
                         'source_checks': checks, 'source_wav_member': wav_member, 'crop_policy': policy,
                         'served_train': clip, 'served_full': original, 'pcm_roundtrip_exact': True})
    freeze(DEST / 'review.json', {'version': VERSION, 'items': items})
    freeze(DEST / 'evidence.json', evidence)
    count = len(items)
    speakers = len({r['speaker'] for r in selected})
    seconds = round(sum(i['seconds'] for i in items), 3)
    page = Path(__file__).with_name('phonetic.html').read_text(encoding='utf-8')
    replacements = {
        '들은 소리 확인 · 15개': f'책 라벨 음성 확인 · {count}개',
        '들린 소리를 그대로 골라 주세요': '실제로 어떤 소리로 들리나요?',
        '이전에 다르게 들렸던 5개와 다른 화자의 새 녹음 10개입니다.':
            f'본학습에서 단독 「책」으로 라벨링된 녹음 {count}개 전부입니다. {speakers}명, 소리 합계 약 {seconds:.0f}초입니다.',
        '문장 전체': '단어 전체',
        'max="15"': f'max="{count}"',
        '15개 모두 저장했습니다.': f'{count}개 모두 저장했습니다.',
        '예: 가운데 ㄹ 소리는 안 들림 / 첫 음절만 헷갈림. 이 메모만으로 정답 표기를 확정하지 않습니다.':
            '다르게 들린 소리나 확실한 부분만 적어 주세요. 메모는 자동으로 학습 정답이 되지 않습니다.',
    }
    for old, new in replacements.items():
        assert old in page, old
        page = page.replace(old, new)
    (DEST / 'listen.html').write_text(page, encoding='utf-8')
    report = {'version': VERSION, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
              'questions': count, 'distinct_speakers': speakers, 'total_training_audio_seconds': seconds,
              'age_counts': dict(Counter(r['age'] for r in selected)),
              'previously_presented_recordings_included': sum(e['previously_presented'] for e in evidence),
              'cropped_items_with_original_available': sum(i['cropped'] for i in items),
              'selection_sha256': digest(DEST / 'selection.json'),
              'review_sha256': digest(DEST / 'review.json'), 'evidence_sha256': digest(DEST / 'evidence.json'),
              'page_sha256': digest(DEST / 'listen.html'), 'original_manifest_sha256': MANIFEST_SHA,
              'source_and_training_pcm_verified': count, 'source_and_g2p_verified': count,
              'model_predictions_used': False, 'human_answers_created': False,
              'labels_changed': False, 'training_started': False,
              'limits': ['All standalone 책 recordings in this baseline training manifest, not all corpus uses of the word.',
                         'The listener knows this is a 책-label set; hiding choices until playback is not full blinding.',
                         'Previously presented recordings are retained for complete coverage, not counted as independent new evidence.',
                         'Choices are listening observations, not automatically verified phonetic targets or proof of the cause of model errors.']}
    dump(DEST / 'preparation.json', report)
    dump(ROOT / 'docs/data/child-jamo-book-review-preparation-20261003.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
