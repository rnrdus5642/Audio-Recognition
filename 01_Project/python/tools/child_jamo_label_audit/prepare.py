"""Freeze a prediction-independent sample, then verify its training provenance.

Run with .venv-gpu/Scripts/python.exe -X utf8. This never changes training data.
The selection is saved BEFORE any selected waveform or prediction is read.
"""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tarfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/tools/child_jamo_finetune'))
import audio
import labels
from common import RAW, WORK, digest, dump, load

WEB = ROOT / '.venv/child-jamo-web-20260916'
OLD = WEB / 'cause-analysis-20260927'
DEST = WEB / 'dataset-audit-20260928'
SEED = 'original-training-pronunciation-audit-20260928-v1'
MANIFEST_SHA = '6b15a65983f64d8b01603161fbb901e1dbcee164fce605d39560ac2a71681a04'
GROUPS = ['velar_tense', 'affricate_stop', 'onset_presence', 'coda_presence', 'vowel_quality']
INITIALS = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
VOWELS = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
FINALS = ' ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'


def ordered_key(value, purpose):
    return hashlib.sha256(f'{SEED}:{purpose}:{value}'.encode()).hexdigest()


def pcm_hash(wave):
    return hashlib.sha256(np.asarray(wave, dtype='<f4').tobytes()).hexdigest()


def collect_wavs(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'wav' and isinstance(item, str):
                yield item
            else:
                yield from collect_wavs(item)
    elif isinstance(value, list):
        for item in value:
            yield from collect_wavs(item)


def existing_reviews():
    """Read old answer histories without writing to any previous review database."""
    excluded, records = set(), []
    paths = [OLD / 'listening-evidence.json']
    paths += sorted(OLD.glob('*/listening-evidence.json'))
    for path in paths:
        evidence = load(path)
        excluded.update(collect_wavs(evidence))
        db_path = path.parent / 'listening-answers.sqlite3'
        with closing(sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            answers = [dict(r) for r in db.execute('SELECT * FROM answers ORDER BY revision')]
        records.append({'evidence': str(path.relative_to(ROOT)), 'evidence_sha256': digest(path),
                        'database': str(db_path.relative_to(ROOT)), 'answer_history': answers})
    # The first 18 questions were personal recordings, outside the training set.
    # Keep a read-only snapshot too; do not confuse these with dataset evidence.
    first = WEB / 'consonant-review/reviews.sqlite3'
    with closing(sqlite3.connect(first.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
                  if not r[0].startswith('sqlite_')]
        snapshot = {t: [dict(r) for r in db.execute('SELECT * FROM "' + t.replace('"', '""') + '"')]
                    for t in tables}
    return excluded, {'previous_batches': records, 'first_personal_review': snapshot,
                      'first_personal_review_database': str(first.relative_to(ROOT)),
                      'reuse_policy': 'Preserve all old answers and uncertainty. Exclude all previously presented WAVs, not just answered ones. Free text is not a new phonetic target.'}


def quotas(counts, total=10):
    """Largest remainders, with at least one from each observed age/style stratum."""
    ideals = {k: total * n / sum(counts.values()) for k, n in counts.items()}
    result = {k: max(1, int(v)) for k, v in ideals.items()}
    if sum(result.values()) > total:
        raise ValueError('Too many strata for this pilot')
    for key in sorted(counts, key=lambda k: (-(ideals[k] - result[k]), k))[:total - sum(result.values())]:
        result[key] += 1
    return result


def contrast(surface, group):
    """Literal one-position contrasts, NOT alternative G2P targets or diagnoses."""
    if ' ' in surface or not 1 <= len(surface) <= 4:
        return None
    for position, ch in enumerate(surface):
        n = ord(ch) - 0xAC00
        if not 0 <= n < 11172:
            return None
        onset, vowel, coda = n // 588, n // 28 % 21, n % 28
        forms, focus = [], ''
        if group == 'velar_tense' and INITIALS[onset] in 'ㄱㄲ':
            forms = [chr(0xAC00 + INITIALS.index(c) * 588 + vowel * 28 + coda) for c in 'ㄱㄲ']
            focus = 'ㄱ / ㄲ'
        elif group == 'affricate_stop' and INITIALS[onset] in 'ㅊㅉㅌ':
            forms = [chr(0xAC00 + INITIALS.index(c) * 588 + vowel * 28 + coda) for c in 'ㅊㅉㅌ']
            focus = 'ㅊ / ㅉ / ㅌ'
        elif group == 'onset_presence' and INITIALS[onset] in 'ㅂㄴㄹ':
            forms = [ch, chr(0xAC00 + INITIALS.index('ㅇ') * 588 + vowel * 28 + coda)]
            focus = f'{INITIALS[onset]} 초성의 유무'
        elif group == 'coda_presence' and position == len(surface) - 1 and FINALS[coda] in 'ㄱㄴㄷㄹㅁㅂㅇ':
            forms = [ch, chr(ord(ch) - coda)]
            focus = f'{FINALS[coda]} 받침의 유무'
        elif group == 'vowel_quality' and VOWELS[vowel] in 'ㅓㅗㅜㅡ':
            pair = 'ㅓㅗ' if VOWELS[vowel] in 'ㅓㅗ' else 'ㅜㅡ'
            forms = [chr(0xAC00 + onset * 588 + VOWELS.index(c) * 28 + coda) for c in pair]
            focus = ' / '.join(pair)
        if forms:
            return {'position': position, 'focus': focus,
                    'forms': [surface[:position] + c + surface[position + 1:] for c in forms]}
    return None


def select(rows, excluded):
    remaining = [r for r in rows if r['wav'] not in excluded]
    counts = Counter((r['age'], r['style']) for r in remaining)
    allocation = quotas(counts)
    selected, speakers, names, texts = [], set(), set(), set()

    def take(pool, group, count, unique_text=False):
        added = 0
        for row in sorted(pool, key=lambda r: ordered_key(r['wav'], group)):
            if row['speaker'] in speakers or row['wav'] in names or (unique_text and row['surface'] in texts):
                continue
            item = {'group': group, 'row': row}
            if group in GROUPS:
                item['contrast'] = contrast(row['surface'], group)
            selected.append(item)
            speakers.add(row['speaker'])
            names.add(row['wav'])
            texts.add(row['surface'])
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError(f'Insufficient candidates: {group}, {added}/{count}')

    for (age, style), number in sorted(allocation.items()):
        take([r for r in remaining if (r['age'], r['style']) == (age, style)], f'random:{age}:{style}', number)
    eligible = {}
    for group in GROUPS:
        pool = [r for r in remaining if .3 <= r['seconds'] <= 6 and contrast(r['surface'], group)]
        eligible[group] = len(pool)
        take(pool, group, 2, unique_text=True)
    selected.sort(key=lambda s: ordered_key(s['row']['wav'], 'display'))
    return {'seed': SEED, 'manifest_sha256': MANIFEST_SHA, 'population': len(rows),
            'selection_used_predictions': False, 'selection_used_audio_listening': False,
            'method': '10 age/style-stratified hash-order draws with minimum one per observed stratum; 10 short-word contrast draws (2 per family); unique speakers across all 20 and unique surfaces for focused draws. Exclude previously presented files. No replacement for difficult/ambiguous audio.',
            'random_strata': [{'age': a, 'style': s, 'remaining': counts[a, s], 'draws': n}
                              for (a, s), n in sorted(allocation.items())],
            'focused_eligible_counts_before_speaker_and_word_filters': eligible,
            'excluded_previously_presented_training_count': sum(r['wav'] in excluded for r in rows),
            'excluded_wav_names': sorted(excluded), 'items': selected,
            'limitations': ['Exploratory detection, not an estimate of population prevalence.',
                            'No model disagreement filter; agreement is assessed only after freezing.',
                            'Focused alternatives are hypothetical literal contrasts, not known mistakes.',
                            'One listener and optional approximate text do not establish expert phonetic labels.']}


def read_verified_member(path, offset, size, sha, basename):
    with path.open('rb') as stream:
        stream.seek(offset - 512)
        info = tarfile.TarInfo.frombuf(stream.read(512), 'utf-8', 'surrogateescape')
        blob = stream.read(size)
    assert Path(info.name).name == basename and info.size == size, basename
    assert len(blob) == size and hashlib.sha256(blob).hexdigest() == sha, basename
    return blob, info.name


def write_wav(wave, name):
    buffer = io.BytesIO()
    # FLOAT preserves even a non-integer stereo average exactly.
    sf.write(buffer, wave, 16000, format='WAV', subtype='FLOAT')
    blob = buffer.getvalue()
    restored, rate = sf.read(io.BytesIO(blob), dtype='float32')
    assert rate == 16000 and np.array_equal(restored, wave)
    (DEST / 'audio' / name).write_bytes(blob)
    return {'file': name, 'sha256': hashlib.sha256(blob).hexdigest(),
            'pcm_sha256': pcm_hash(wave), 'seconds': len(wave) / 16000}


def main():
    if (DEST / 'answers.sqlite3').exists():
        raise RuntimeError('A review database exists. Never regenerate an active question set.')
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'audio').mkdir(exist_ok=True)
    (DEST / 'source-json').mkdir(exist_ok=True)
    manifest = WORK / 'manifest-train.jsonl'
    assert digest(manifest) == MANIFEST_SHA
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 150253 and all(r['split'] == 'train' for r in rows)
    excluded, reuse = existing_reviews()
    proposal = select(rows, excluded)
    if (DEST / 'selection.json').exists():
        plan = load(DEST / 'selection.json')
        assert plan == proposal, 'Frozen selection differs; refuse reselection'
    else:
        dump(DEST / 'selection.json', proposal)
        plan = proposal
    dump(DEST / 'previous-reviews.json', reuse)
    needed = {s['row']['wav'] for s in plan['items']}
    journal_report = load(ROOT / 'docs/data/child-jamo-training-annotation-fields-20260927.json')
    journal = ROOT / journal_report['provenance_journal']
    assert digest(journal) == journal_report['provenance_journal_sha256']
    sources = {}
    with journal.open(encoding='utf-8') as stream:
        for line in stream:
            entry = json.loads(line)
            if entry['wav'] in needed:
                assert entry['wav'] not in sources
                sources[entry['wav']] = entry
    assert sources.keys() == needed
    archives = {}
    wanted = {r['archive'].lower() for r in sources.values()} | {s['row']['archive'].lower() for s in plan['items']}
    for path in RAW.rglob('*.tar'):
        if path.name.lower() in wanted:
            assert path.name.lower() not in archives
            archives[path.name.lower()] = path
    assert archives.keys() == wanted

    items, evidence = [], []
    general = [{'value': c, 'label': t} for c, t in [
        ('match', '소리가 비교 표기와 같게 들림'),
        ('different', '다르게 들리는 부분이 있음'),
        ('ambiguous', '같게도 다르게도 들려 확정하기 어려움'),
        ('no_speech', '말소리가 없음'),
        ('unusable', '잡음·겹침·잘림 등으로 판단하기 어려움')]]
    for number, selection in enumerate(plan['items'], 1):
        row = selection['row']
        source = sources[row['wav']]
        raw_blob, member = read_verified_member(archives[source['archive'].lower()], source['offset'],
                                               source['size'], source['payload_sha256'], Path(row['wav']).with_suffix('.json').name)
        raw = json.loads(raw_blob.decode('utf-8-sig'))
        (DEST / 'source-json' / f'{number:02d}.json').write_bytes(raw_blob)
        text, surface, ids = labels.targets(raw['Transcription']['LabelText'])
        checks = {
            'source_filename': raw['File']['FileName'] == row['wav'],
            'source_length': float(raw['File']['FileLength']) == row['seconds_index'],
            'speech_start': str(raw['Miscellaneous_Info']['SpeechStart']) == str(row['speech_start']),
            'speech_end': str(raw['Miscellaneous_Info']['SpeechEnd']) == str(row['speech_end']),
            'normalized_transcript': text == row['text'], 'g2p_surface': surface == row['surface'],
            'label_ids': ids == row['label_ids'],
        }
        assert all(checks.values()), (number, checks)
        blob, wav_member = read_verified_member(archives[row['archive'].lower()], row['offset'], row['size'],
                                               row['source_sha256'], row['wav'])
        full = audio.decode(blob)
        wave, policy = audio.crop(full, row['speech_start'], row['speech_end'])
        assert pcm_hash(wave) == row['pcm_sha256'] and len(wave) == row['samples']
        train_clip = write_wav(wave, f'{number:02d}-train.wav')
        same_audio = np.array_equal(wave, full)
        full_clip = train_clip if same_audio else write_wav(full, f'{number:02d}-full.wav')
        if selection['group'].startswith('random:'):
            reference = {'kind': 'general', 'surface': surface, 'choices': general,
                         'question': '맞춤법이 아니라, 들린 소리와 이 표기가 같은지 골라 주세요.'}
            roles = {'match': 'training_surface', 'different': 'reported_mismatch'}
        else:
            spec = selection['contrast']
            forms = sorted(spec['forms'], key=lambda s: ordered_key(row['wav'] + ':' + s, 'options'))
            choices = [{'value': f'form_{i}', 'label': form, 'focus_position': spec['position']}
                       for i, form in enumerate(forms, 1)]
            roles = {c['value']: ('training_surface' if c['label'] == surface else 'hypothetical_contrast') for c in choices}
            choices += [{'value': c, 'label': t} for c, t in [
                ('ambiguous', '둘 이상으로 들려 구분하기 어려움'), ('other', '보기와 다른 소리로 들림'),
                ('no_speech', '말소리가 없음'), ('unusable', '잡음·겹침·잘림 등으로 판단하기 어려움')]]
            reference = {'kind': 'contrast', 'choices': choices,
                         'question': '굵게 표시한 음절을 중심으로, 들린 소리에 가장 가까운 표기를 골라 주세요. 다른 부분도 다르면 「보기와 다른 소리」를 골라도 됩니다.'}
        identity = {'version': SEED, 'wav': row['wav'], 'train_pcm_sha256': pcm_hash(wave),
                    'full_pcm_sha256': pcm_hash(full), 'source_json_sha256': source['payload_sha256'],
                    'reference': reference, 'training_ids': ids}
        binding = hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        items.append({'id': str(number), 'number': number, 'binding': binding, 'seconds': row['seconds'],
                      'train': train_clip, 'full': full_clip, 'cropped': not same_audio, 'reference': reference})
        evidence.append({'number': number, **selection, 'binding': binding, 'identity': identity,
                         'choice_roles': roles, 'source_json': {**source, 'verified_member': member,
                         'local_copy': f'source-json/{number:02d}.json'}, 'source_checks': checks,
                         'source_wav_member': wav_member, 'crop_policy': policy,
                         'crop_seconds': audio.span(row['speech_start'], row['speech_end'], len(full) / 16000)[:2],
                         'full_seconds': len(full) / 16000, 'training_audio_is_full_source': same_audio,
                         'served_train': train_clip, 'served_full': full_clip, 'pcm_roundtrip_exact': True})
    report = {'version': SEED, 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
              'questions': len(items), 'random_questions': 10, 'focused_questions': 10,
              'distinct_speakers': len({e['row']['speaker'] for e in evidence}),
              'selection_sha256': digest(DEST / 'selection.json'),
              'original_manifest_sha256': MANIFEST_SHA,
              'source_and_training_pcm_verified': len(items), 'source_and_g2p_verified': len(items),
              'total_training_audio_seconds': round(sum(i['seconds'] for i in items), 3),
              'cropped_items_with_original_available': sum(i['cropped'] for i in items),
              'previously_presented_training_wavs_excluded': plan['excluded_previously_presented_training_count'],
              'model_predictions_used_in_selection': False, 'human_answers_created': False,
              'labels_changed': False, 'training_started': False, 'status': 'prepared_awaiting_listening',
              'items': [{'number': e['number'], 'group': e['group'], 'wav': e['row']['wav'],
                         'age': e['row']['age'], 'style': e['row']['style']} for e in evidence],
              'limitations': plan['limitations']}
    dump(DEST / 'review.json', {'version': SEED, 'items': items})
    dump(DEST / 'evidence.json', evidence)
    dump(DEST / 'preparation.json', report)
    dump(ROOT / 'docs/data/child-jamo-dataset-audit-preparation-20260928.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'items'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
