"""Regression tests for lost speech, noise-only pairs and bounded repairs."""
from pathlib import Path
import hashlib
import io
import json
import sqlite3
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
import integrity
import labels
import manifest
import quality
import repair
import reviewed
from common import digest, dump, load


def wav_row(name, wave, start, end, speaker, split='train'):
    stream = io.BytesIO()
    sf.write(stream, wave, 16000, format='WAV', subtype='PCM_16')
    blob = stream.getvalue()
    full = audio.decode(blob)
    crop, policy = audio.crop(full, start, end)
    return dict(wav=name, archive=name + '.tar', offset=0, size=len(blob),
                speaker=speaker, age='6', style='formatted', split=split,
                seconds_index=len(full) / 16000, speech_start=start, speech_end=end,
                seconds_planned=len(crop) / 16000, span_policy=policy,
                text='책', surface='책', label_ids=labels.targets('책')[2],
                samples=len(crop), seconds=len(crop) / 16000,
                source_sha256=hashlib.sha256(blob).hexdigest(),
                pcm_sha256=hashlib.sha256(crop.tobytes()).hexdigest()), blob, full


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    # Strong speech-like content early in a file, only weak noise in its crop.
    full = np.full(46080, .001, dtype=np.float32)
    full[1600:16000] = .2 * np.sin(np.arange(14400) * .1)
    samples = [wav_row('cut.wav', full, '1.690', '2.740', 's1'),
               wav_row('noise.wav', np.full(14336, .0003), '0.02', '0.80', 's2'),
               wav_row('quiet.wav', np.full(24000, .001), '0', '1.5', 's3'),
               wav_row('valid.wav', np.full(16000, .05), '0', '1', 's4', 'valid'),
               wav_row('test.wav', np.full(17600, .06), '0', '1.1', 's5', 'test')]
    rows = [r for r, _, _ in samples]
    blobs = {r['archive']: b for r, b, _ in samples}
    monkeypatch.setattr(audio, 'payload', lambda archive, offset, size: blobs[archive])
    for module in (manifest, integrity, repair):
        monkeypatch.setattr(module, 'WORK', tmp_path)
    monkeypatch.setattr(repair, 'ROOT', tmp_path)
    monkeypatch.setattr(integrity, 'ROOT', tmp_path)
    monkeypatch.setattr(manifest, 'log', lambda *args: None)
    bindings = {'fixture.tar': {'identity': 'unchanged'}}
    monkeypatch.setattr(integrity.catalog, 'verify', lambda: bindings)
    for split in ('train', 'valid', 'test'):
        (tmp_path / f'manifest-{split}.jsonl').write_bytes(b''.join(
            (json.dumps(r, ensure_ascii=False) + '\r\n').encode('utf-8')
            for r in rows if r['split'] == split))
    dump(tmp_path / 'selection-summary.json', {
        'seed': 'fixture', 'index_sha256': 'index', 'splits_sha256': 'splits',
        'old_dev_sha256': 'old-dev', 'bounds_seconds': [.5, 15]})
    baseline = {s: digest(tmp_path / f'manifest-{s}.jsonl') for s in ('train', 'valid', 'test')}
    dump(tmp_path / 'audit-summary.json', {
        'manifest_sha256': baseline,
        'provenance': {'labels_version': labels.LABELS_VERSION,
                       'labels_contract_sha256': labels.contract_sha256(),
                       'selection_summary_sha256': digest(tmp_path / 'selection-summary.json'),
                       'source_archives': bindings}})
    entries = []
    answers = []
    database = tmp_path / 'followup-listening/listening-answers.sqlite3'
    database.parent.mkdir()
    with sqlite3.connect(database) as connection:
        connection.execute('CREATE TABLE answers '
                           '(revision INTEGER, item_id TEXT, binding TEXT, choice TEXT)')
        for i, action in enumerate(('whole_source', 'exclude')):
            row, _, full_wave = samples[i]
            review = dict(item_id=str(i + 1), revision=i + 1, binding='bound-' + str(i),
                          choice='full_only' if i == 0 else 'no_speech', reason='listened')
            entry = {k: row[k] for k in ('wav', 'text', 'surface', 'label_ids', 'source_sha256')}
            entry.update(action=action, original_pcm_sha256=row['pcm_sha256'], review=review)
            if i == 0:
                entry.update(full_pcm_sha256=hashlib.sha256(full_wave.tobytes()).hexdigest(),
                             full_samples=len(full_wave))
            entries.append(entry)
            answers.append(dict(review, source_id='train:' + row['wav'],
                                source_pcm_sha256=row['pcm_sha256'],
                                played_audio_and_binding_verified=True))
            connection.execute('INSERT INTO answers VALUES (?,?,?,?)',
                               (review['revision'], review['item_id'], review['binding'], review['choice']))
    dump(tmp_path / 'review.json', {'answered': answers})
    ledger = tmp_path / 'repairs.json'
    dump(ledger, dict(version=1, source_manifest_sha256=baseline['train'],
                      review_report='review.json', decisions=entries))
    monkeypatch.setattr(reviewed, 'LEDGER', ledger)
    reviewed.decisions.cache_clear()

    class LocalPool:
        def __init__(self, workers):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def imap(self, function, values, chunksize):
            return map(function, values)

    monkeypatch.setattr(repair.mp, 'Pool', LocalPool)
    yield dict(root=tmp_path, rows=rows, samples=samples, blobs=blobs, baseline=baseline)
    reviewed.decisions.cache_clear()


def test_wrong_timestamp_is_flagged_and_review_restores_exact_audio(dataset):
    row, _, full = dataset['samples'][0]
    original_crop, _ = audio.crop(full, row['speech_start'], row['speech_end'])
    assert abs(original_crop).max() > 0  # The old silence test missed it.
    assert 'crop_excludes_most_strong_audio' in quality.measure(full, original_crop, 3)['flags']
    _, result = manifest._check(row)
    after = repair.repaired_row(row, result)
    actual, _ = audio.crop(full, after['speech_start'], after['speech_end'])
    np.testing.assert_array_equal(actual, full)
    assert after['label_ids'] == row['label_ids']
    assert after['samples'] == 46080
    assert not result['quality']['flags']


def test_noise_only_pair_is_excluded_by_review_not_a_zero_threshold(dataset):
    row = dataset['rows'][1]
    _, result = manifest._check(row)
    assert result['peak'] > 0
    assert 'low_crop_rms_below_minus50_dbfs' in result['quality']['flags']
    assert repair.repaired_row(row, result) is None


def test_quiet_unreviewed_audio_is_flagged_but_unchanged(dataset):
    row = dataset['rows'][2]
    _, result = manifest._check(row)
    assert result['quality']['flags']
    assert result['review_decision'] is None
    assert repair.repaired_row(row, result) == row


@pytest.mark.parametrize('field,value', [('source_sha256', 'changed'),
                                       ('pcm_sha256', 'changed'), ('samples', 1)])
def test_data_drift_is_not_hidden_by_a_repair(dataset, field, value):
    row = dict(dataset['rows'][0], **{field: value})
    _, result = manifest._check(row)
    with pytest.raises(RuntimeError, match='differs from the recorded input'):
        repair.repaired_row(row, result)


def test_same_audio_with_different_label_is_not_covered_by_review(dataset):
    row = dict(dataset['rows'][0], text='다른 정답')
    _, result = manifest._check(row)
    with pytest.raises(RuntimeError, match='differs from the reviewed label'):
        repair.repaired_row(row, result)


def test_fresh_training_rejects_both_confirmed_defects(dataset):
    for row in dataset['rows'][:2]:
        with pytest.raises(RuntimeError, match='confirmed defective'):
            reviewed.require_repaired([row])


def test_materialized_split_has_exactly_the_two_changes_and_valid_provenance(dataset):
    root = dataset['root']
    originals = {s: (root / f'manifest-{s}.jsonl').read_bytes() for s in dataset['baseline']}
    summary = repair.build('train-repaired-test', 1)
    rows = [json.loads(line) for line in (root / 'manifest-train-repaired-test.jsonl').read_bytes().splitlines()]
    assert [r['wav'] for r in rows] == ['cut.wav', 'quiet.wav']
    assert summary['recordings'] == 2 and len(summary['changes']) == 2
    assert summary['quality_review_candidates'] == 1
    assert rows[1] == dataset['rows'][2]
    assert (root / 'manifest-train-repaired-test.jsonl').read_bytes().endswith(
        originals['train'].splitlines(keepends=True)[2])
    reviewed.require_repaired(rows)
    assert integrity._provenance('train-repaired-test')['vouched_by'] == 'audit-train-repaired-test.json'
    for split, original in originals.items():
        assert (root / f'manifest-{split}.jsonl').read_bytes() == original
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        repair.build('train-repaired-test', 1)


@pytest.mark.parametrize('filename,match', [
    ('quality-review-train-repaired-test.json', 'Quality review report changed'),
    ('review.json', 'Listening evidence changed'),
    ('repairs.json', 'Listening decisions changed'),
    ('manifest-train-repaired-test.jsonl', 'not produced by the recorded'),
    ('selection-summary.json', 'selection-summary.json has changed'),
])
def test_repaired_data_remains_bound_to_its_evidence(dataset, filename, match):
    repair.build('train-repaired-test', 1)
    path = dataset['root'] / filename
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(RuntimeError, match=match):
        integrity._provenance('train-repaired-test')


def test_newer_listening_answer_stops_materialization(dataset):
    root = dataset['root']
    with sqlite3.connect(root / 'followup-listening/listening-answers.sqlite3') as connection:
        connection.execute('INSERT INTO answers VALUES (?,?,?,?)', (10, '1', 'bound-0', 'unclear'))
    with pytest.raises(RuntimeError, match='newer listening answer'):
        repair.build('train-repaired-test', 1)
    assert not (root / 'manifest-train-repaired-test.jsonl').exists()


def test_corrupt_source_cannot_publish_a_partial_repair(dataset):
    dataset['blobs']['cut.wav.tar'] = dataset['blobs']['quiet.wav.tar']
    with pytest.raises(RuntimeError, match='differs from the recorded input'):
        repair.build('train-repaired-test', 1)
    assert not (dataset['root'] / 'manifest-train-repaired-test.jsonl').exists()


def test_regular_audit_applies_the_same_repairs_and_records_review_candidates(dataset):
    root = dataset['root']
    (root / 'selection.jsonl').write_text('\n'.join(
        json.dumps(row, ensure_ascii=False) for row in dataset['rows']), encoding='utf-8')
    manifest.audit(1)
    summary = load(root / 'audit-summary.json')
    assert summary['audited'] == 5 and summary['kept'] == 4
    assert summary['drops']['dropped_human_confirmed_no_speech'] == 1
    assert summary['applied_human_decisions'] == 2
    assert summary['quality_review_candidates'] == 1
    assert integrity._provenance('train')['vouched_by'] == 'audit-summary.json'
    rows = [json.loads(line) for line in (root / 'manifest-train.jsonl').read_bytes().splitlines()]
    reviewed.require_repaired(rows)
    assert rows[0]['samples'] == 46080


def test_training_guard_runs_before_loading_the_encoder(dataset, monkeypatch):
    import train

    class NoWriter:
        def __init__(self, *args):
            pass

        def acquire(self):
            pass

    monkeypatch.setattr(train, 'Owned', NoWriter)
    monkeypatch.setattr(train.data, 'rows', lambda split: dataset['rows'][:3])
    monkeypatch.setattr(train, 'build', lambda args: pytest.fail('Encoder must not be loaded'))
    with pytest.raises(RuntimeError, match='confirmed defective training pair'):
        train.main(train.parser().parse_args([]))
