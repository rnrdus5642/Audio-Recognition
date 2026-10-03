"""The counting rules, on cases where the right answer is known.

These are about the audit's arithmetic and its honesty, not about the
model: no checkpoint is loaded. Each test fabricates a reference and a
hypothesis whose correct classification can be stated in advance.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_web/test_audit.py -q
"""
from pathlib import Path
import json
import sys

import pytest

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))
import audit                                                   # noqa: E402
import labels                                                  # noqa: E402


def ids(text):
    """Label ids for a surface form, the way the manifest holds them."""
    _text, surface, label_ids = labels.targets(text)
    assert label_ids is not None, text
    return surface, label_ids


def record(transcript, predicted):
    surface, reference = ids(transcript)
    _s, hypothesis = ids(predicted)
    return {'reference_surface': surface, 'reference_ids': reference,
            'hypothesis_ids': hypothesis}


def events_of(transcript, predicted):
    events, trusted = audit.analyse_record(record(transcript, predicted))
    assert trusted
    return [e for e in events if e['op'] != 'equal']


# --- syllable roles ---------------------------------------------------

def test_onset_and_coda_come_from_the_expected_hangul():
    roles, trusted = audit.reference_roles(
        {'surface': '책', 'label_ids': ids('책')[1]})
    assert trusted
    assert roles == [('ㅊ', 'onset'), ('ㅐ', 'vowel'), ('ㄱ', 'coda')]


def test_a_substituted_onset_is_labelled_onset():
    events = events_of('책', '샙')
    assert {'op': 'substitute', 'reference': 'ㅊ', 'hypothesis': 'ㅅ',
            'role': 'onset', 'hypothesis_is_consonant': True} in events


def test_a_substituted_coda_is_labelled_coda():
    events = events_of('책', '샙')
    assert {'op': 'substitute', 'reference': 'ㄱ', 'hypothesis': 'ㅂ',
            'role': 'coda', 'hypothesis_is_consonant': True} in events


def test_the_same_consonant_is_counted_by_its_own_position():
    """ㄱ as an onset and ㄱ as a coda are different questions."""
    roles, trusted = audit.reference_roles(
        {'surface': '국', 'label_ids': ids('국')[1]})
    assert trusted
    assert [r for _, r in roles] == ['onset', 'vowel', 'coda']


# --- the edit classes -------------------------------------------------

def test_a_missing_onset_is_a_deletion_not_a_substitution():
    events = events_of('책', '액')          # ㅊ -> ㅇ is a real onset here
    assert all(e['op'] != 'delete' for e in events)
    # and a genuinely absent one:
    reference = record('책', '책')
    reference['hypothesis_ids'] = [i for i in reference['hypothesis_ids']
                                   if labels.TOKENS[i] != 'ㅊ']
    events, _ = audit.analyse_record(reference)
    dropped = [e for e in events if e['op'] == 'delete']
    assert dropped == [{'op': 'delete', 'reference': 'ㅊ', 'hypothesis': None,
                        'role': 'onset'}]


def test_an_extra_consonant_is_an_insertion_with_no_syllable_role():
    reference = record('아', '아')
    surface, extra = ids('악')
    reference['hypothesis_ids'] = extra
    events, _ = audit.analyse_record(reference)
    inserted = [e for e in events if e['op'] == 'insert']
    assert inserted == [{'op': 'insert', 'reference': None,
                         'hypothesis': 'ㄱ', 'role': 'none'}]


def test_vowel_errors_are_not_counted_as_consonant_errors():
    events = events_of('야옹이', '야웅이')        # ㅗ -> ㅜ
    assert events == []


# --- what the reference actually is -----------------------------------

def test_expected_nasal_assimilation_is_not_an_error():
    """씩 넣고 -> 씽 너코 is the target, so matching it is not a mistake."""
    surface, reference = ids('두개씩 넣고')
    assert '씽' in surface and '너코' in surface
    events, trusted = audit.analyse_record(
        {'reference_surface': surface, 'reference_ids': reference,
         'hypothesis_ids': reference})
    assert trusted
    assert [e for e in events if e['op'] != 'equal'] == []


def test_predicting_the_spelling_instead_of_the_pronunciation_is_an_error():
    """The mirror image: a model that emitted 씩 where the target says 씽.

    The hypothesis is built from jamo directly. Putting the string through
    `targets` would apply the same rules to it and quietly turn it back
    into 씽, testing nothing.
    """
    surface, reference = ids('두개씩 넣고')
    assert '씽' in surface
    hypothesis = [labels.VOCAB[j] for j in
                  list('ㄷㅜㄱㅐㅆㅣㄱ') + ['|'] + list('ㄴㅓㅋㅗ')]
    events, trusted = audit.analyse_record(
        {'reference_surface': surface, 'reference_ids': reference,
         'hypothesis_ids': hypothesis})
    assert trusted
    assert {'op': 'substitute', 'reference': 'ㅇ', 'hypothesis': 'ㄱ',
            'role': 'coda', 'hypothesis_is_consonant': True} in events


def test_roles_are_unknown_rather_than_guessed_when_they_disagree():
    """A surface that does not match the label ids must not be annotated."""
    _surface, reference = ids('책')
    roles, trusted = audit.reference_roles(
        {'surface': '사과', 'label_ids': reference})
    assert not trusted
    assert [r for _, r in roles] == ['unknown', 'unknown', 'unknown']


# --- aggregation ------------------------------------------------------

def test_denominators_count_every_reference_consonant():
    surface, reference = ids('책')
    _s, hypothesis = ids('샙')
    pools, per_record = audit.analyse({'x': {
        'id': 'x', 'pool': 'personal', 'speaker': 'child1', 'child': 'child1',
        'seconds': 1.0, 'transcript': '책', 'reference_surface': surface,
        'reference_ids': reference, 'hypothesis_ids': hypothesis,
        'predicted_text': '샙', 'audio_pcm_sha256': 'a', 'baseline_key': 'k',
        'metrics': {'errors': 2, 'reference_jamo': 3, 'exact': False},
    }})
    pool = pools['personal']
    assert pool['reference_consonants'] == 2         # ㅊ and ㄱ
    assert pool['substitutions'] == 2 and pool['deletions'] == 0
    assert pool['per_consonant']['ㅊ']['reference_total'] == 1
    assert pool['per_consonant']['ㅊ']['to']['ㅅ'] == 1
    assert per_record['x']['consonant_mistakes'] == 2


def test_a_correct_consonant_still_gets_a_denominator():
    surface, reference = ids('가')
    pools, _ = audit.analyse({'y': {
        'id': 'y', 'pool': 'valid', 'speaker': 's', 'child': None,
        'seconds': 1.0, 'transcript': '가', 'reference_surface': surface,
        'reference_ids': reference, 'hypothesis_ids': reference,
        'predicted_text': '가', 'audio_pcm_sha256': 'a', 'baseline_key': 'k',
        'metrics': {'errors': 0, 'reference_jamo': 2, 'exact': True},
    }})
    stats = pools['valid']['per_consonant']['ㄱ']
    assert stats['reference_total'] == 1 and stats['correct'] == 1
    assert pools['valid']['reference_consonants'] == 1


# --- identity ---------------------------------------------------------

def identity(**overrides):
    base = {'checkpoint_sha256': 'a' * 64, 'labels_contract_sha256': 'b' * 64,
            'labels_version': 'v1', 'decode_version': 'compose-1',
            'input_policy': 'raw float32', 'precision': 'float32',
            'record_version': audit.RECORD_VERSION}
    base.update(overrides)
    return base


@pytest.mark.parametrize('field, value', [
    ('checkpoint_sha256', 'c' * 64),
    ('labels_contract_sha256', 'd' * 64),
    ('decode_version', 'compose-2'),
    ('input_policy', 'normalised'),          # the one that was missing
    ('precision', 'float16'),
    ('record_version', 99),
])
def test_every_part_of_the_computation_changes_the_baseline_key(field, value):
    assert audit.baseline_key(identity()) != \
        audit.baseline_key(identity(**{field: value}))


def row_for(text='책', samples=13920, policy='annotated_span', pcm=None):
    _t, surface, ids = labels.targets(text)
    row = {'samples': samples, 'label_ids': ids, 'surface': surface,
           'text': text, 'span_policy': policy}
    if pcm is not None:
        row['pcm_sha256'] = pcm
    return row


def record_for(row, audio='a' * 64):
    return {**audit.catalog_identity(row), 'audio_pcm_sha256': audio,
            'record_version': audit.RECORD_VERSION,
            'hypothesis_ids': [2, 3], 'baseline_key': 'k', 'id': 'x'}


def test_a_record_still_matching_its_row_is_valid():
    row = row_for()
    assert audit.validate_record(record_for(row), row, 'a' * 64) == []


def test_the_same_id_with_different_audio_is_rejected():
    row = row_for()
    assert 'audio_pcm_sha256' in audit.validate_record(
        record_for(row), row, 'f' * 64)


def test_changed_manifest_audio_is_caught_without_reading_the_file():
    """The default resume path: no `--verify-audio`, and still not fooled.

    Same id, same length, same target, same crop - only the recording
    behind the row was replaced. Nothing but the catalogue's own PCM hash
    says so, so it is compared whether or not the caller paid to read the
    audio back.
    """
    row = row_for(pcm='a' * 64)
    assert audit.validate_record(record_for(row), row) == []
    moved = row_for(pcm='f' * 64)
    assert 'catalog_pcm_sha256' in audit.validate_record(record_for(moved),
                                                         moved)


def test_the_same_audio_with_a_different_reference_is_rejected():
    """The prediction stands; what it was scored against moved."""
    record = record_for(row_for('책'))
    problems = audit.validate_record(record, row_for('사과'), 'a' * 64)
    assert 'reference_ids' in problems and 'transcript' in problems


def test_a_changed_sample_count_or_crop_policy_is_rejected():
    record = record_for(row_for())
    assert 'samples' in audit.validate_record(
        record, row_for(samples=99), 'a' * 64)
    assert 'span_policy' in audit.validate_record(
        record, row_for(policy='whole_file_unparsable_span'), 'a' * 64)


def test_an_older_record_layout_is_rejected():
    row = row_for()
    record = {**record_for(row), 'record_version': 1}
    assert 'record_version' in audit.validate_record(record, row, 'a' * 64)


def test_the_item_binding_covers_audio_reference_and_prediction():
    row = row_for()
    base = record_for(row)
    first = audit.record_binding(base)
    assert first == audit.record_binding(dict(base))
    for field, value in (('audio_pcm_sha256', 'f' * 64),
                         ('reference_ids', [9, 9]),
                         ('hypothesis_ids', [4]),
                         ('baseline_key', 'other')):
        assert audit.record_binding({**base, field: value}) != first


# --- crash recovery ---------------------------------------------------

def write_lines(path, records, torn=None):
    with path.open('w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        if torn is not None:
            f.write(torn)                    # no trailing newline
    return path


def whole(record_id, key='k'):
    return {'id': record_id, 'pool': 'personal', 'baseline_key': key,
            'predicted_jamo': 'ㄱ'}


def test_a_torn_final_line_is_repaired_and_the_rest_survives(tmp_path):
    path = write_lines(tmp_path / 'b.jsonl', [whole('a'), whole('b')],
                       torn='{"id": "c", "pool": "per')
    records, report = audit.load_baseline(path=path, repair=True)
    assert set(records) == {'a', 'b'} and report['torn_tail']
    assert report['repaired_bytes'] > 0
    # And the repaired file can be appended to without joining records.
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(whole('d'), ensure_ascii=False) + '\n')
    again, report2 = audit.load_baseline(path=path)
    assert set(again) == {'a', 'b', 'd'} and not report2['torn_tail']


def test_appending_after_an_unrepaired_torn_tail_would_lose_a_record(tmp_path):
    """Why the repair exists: the join is real if nothing truncates it."""
    path = write_lines(tmp_path / 'b.jsonl', [whole('a')],
                       torn='{"id": "c", "pool": "per')
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(whole('d'), ensure_ascii=False) + '\n')
    records, report = audit.load_baseline(path=path)
    assert 'd' not in records                # glued onto the fragment
    assert report['corrupt_middle'] == [2]   # and reported, not silent


def test_a_corrupt_middle_record_is_reported_not_called_a_torn_tail(tmp_path):
    path = tmp_path / 'b.jsonl'
    with path.open('w', encoding='utf-8') as f:
        f.write(json.dumps(whole('a')) + '\n')
        f.write('{"id": "broken"\n')
        f.write(json.dumps(whole('c')) + '\n')
    records, report = audit.load_baseline(path=path)
    assert set(records) == {'a', 'c'}
    assert report['corrupt_middle'] == [2] and not report['torn_tail']


def test_records_from_another_baseline_are_not_mixed_in(tmp_path):
    path = write_lines(tmp_path / 'b.jsonl',
                       [whole('a', 'k1'), whole('b', 'k2')])
    records, report = audit.load_baseline(expect_key='k1', path=path)
    assert set(records) == {'a'} and report['wrong_key'] == 1


def test_a_complete_final_record_without_its_newline_is_terminated(tmp_path):
    """The other repairable ending: whole record, missing line break.

    Nothing is torn, so there is nothing to truncate - but an append would
    still land on the same line and destroy both records. The repair adds
    the break, and the writer's own append then reads back as two records.
    """
    path = tmp_path / 'b.jsonl'
    path.write_text(json.dumps(whole('a'), ensure_ascii=False) + '\n'
                    + json.dumps(whole('b'), ensure_ascii=False),
                    encoding='utf-8')
    records, report = audit.load_baseline(path=path, repair=True)
    assert set(records) == {'a', 'b'}
    assert report['added_final_newline'] and not report['torn_tail']

    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(whole('c'), ensure_ascii=False) + '\n')
    again, report2 = audit.load_baseline(path=path)
    assert set(again) == {'a', 'b', 'c'}
    assert not report2['corrupt_middle'] and not report2['missing_final_newline']


def test_the_missing_newline_is_reported_even_when_not_repairing(tmp_path):
    path = tmp_path / 'b.jsonl'
    path.write_text(json.dumps(whole('a'), ensure_ascii=False),
                    encoding='utf-8')
    _records, report = audit.load_baseline(path=path)
    assert report['missing_final_newline'] and not report['added_final_newline']
    assert path.read_bytes().endswith(b'}')          # untouched


def test_a_corrupt_middle_stops_the_repair_from_touching_the_file(tmp_path):
    """Damage nobody has explained is not something to rewrite around."""
    path = tmp_path / 'b.jsonl'
    with path.open('w', encoding='utf-8') as f:
        f.write(json.dumps(whole('a')) + '\n')
        f.write('{"id": "broken"\n')
        f.write(json.dumps(whole('c')))              # and no final newline
    before = path.read_bytes()
    _records, report = audit.load_baseline(path=path, repair=True)
    assert report['corrupt_middle'] == [2]
    assert not report['added_final_newline'] and not report['repaired_bytes']
    assert path.read_bytes() == before


def _forbidden(name):
    """A stand-in that fails loudly if anything touches it."""
    class Forbidden:
        def __getattr__(self, attribute):
            raise AssertionError(f'{name}: 여기까지 오면 안 됩니다 ({attribute})')
    return Forbidden()


def test_a_baseline_run_refuses_to_append_after_a_corrupt_record(tmp_path,
                                                                 monkeypatch):
    """And the refusal comes first: no model, no repair, no append."""
    path = tmp_path / 'baseline.jsonl'
    with path.open('w', encoding='utf-8') as f:
        f.write(json.dumps(whole('a')) + '\n')
        f.write('{"id": "broken"\n')
        f.write(json.dumps(whole('c')) + '\n')
    before = path.read_bytes()
    monkeypatch.setattr(audit, 'WORK', tmp_path)
    monkeypatch.setattr(audit, 'BASELINE', path)
    monkeypatch.setattr(audit, 'STATE', tmp_path / 'baseline-state.json')
    # Anything that would load a catalogue or a checkpoint is an error here:
    # the refusal must happen before either is reached.
    monkeypatch.setattr(audit, 'samples_module', _forbidden('Catalog'))
    monkeypatch.setattr(audit, 'engine_module', _forbidden('Engine'))
    with pytest.raises(SystemExit) as raised:
        audit.run_baseline()
    assert '손상' in str(raised.value)
    assert path.read_bytes() == before
    assert not (tmp_path / 'baseline-state.json').exists()


# --- progress, coverage and a failed recomputation --------------------

class StubEngine:
    """Enough of the engine to exercise the writer, with no checkpoint.

    `transcribe` can be made to fail, which is the case that matters: a
    stale record whose replacement could not be computed must not leave the
    pool looking covered.
    """

    identity = {'checkpoint_sha256': 'a' * 64,
                'labels_contract_sha256': 'b' * 64,
                'precision': 'float32'}

    def __init__(self, fail=False):
        self.fail = fail
        self.model = self.head = 'loaded'

    def load(self):
        return None

    def transcribe(self, waveform, pcm_sha256=None, use_cache=False):
        if self.fail:
            raise RuntimeError('추론 실패 (시험용)')
        _t, _s, ids = labels.targets('책')
        return {'token_ids': list(ids), 'text': '책', 'jamo': 'ㅊㅐㄱ',
                'orphan_jamo': []}


def stub_modules(monkeypatch, rows, fail=False):
    """Point the audit at a one-row catalogue and a model-free engine."""
    import types
    # Migration always reports on both pools, so both must exist.
    rows = {pool: rows.get(pool, []) for pool in audit.POOLS}

    class StubCatalog:
        def __init__(self):
            self.rows = rows

        def get(self, record_id):
            for pool_rows in rows.values():
                for row in pool_rows:
                    if row['id'] == record_id:
                        return row
            return None

    engine = StubEngine(fail=fail)
    monkeypatch.setattr(audit, 'samples_module',
                        types.SimpleNamespace(Catalog=StubCatalog))
    monkeypatch.setattr(audit, 'engine_module', types.SimpleNamespace(
        Engine=lambda: engine,
        CHECKPOINT=Path('checkpoint.pt'),
        read_row=lambda row: row['id'],
        pcm_sha256=lambda waveform: 'a' * 64))
    return engine


def catalog_row(record_id='personal:a', text='책', pcm='a' * 64):
    _t, surface, ids = labels.targets(text)
    return {'id': record_id, 'split': 'personal', 'wav': 'a.wav',
            'speaker': 'child1', 'child': 'child1', 'age': '5',
            'style': 'word', 'seconds': 1.0, 'samples': 16000,
            'label_ids': list(ids), 'surface': surface, 'text': text,
            'span_policy': 'annotated_span', 'pcm_sha256': pcm}


def stored_record(row, pool='personal', pcm='a' * 64):
    identity = {'checkpoint_sha256': 'a' * 64,
                'labels_contract_sha256': 'b' * 64,
                'labels_version': labels.LABELS_VERSION,
                'decode_version': audit.compose.DECODE_VERSION,
                'input_policy': audit.INPUT_POLICY,
                'precision': 'float32',
                'record_version': audit.RECORD_VERSION}
    return {'id': row['id'], 'pool': pool, 'audio_pcm_sha256': pcm,
            'hypothesis_ids': list(row['label_ids']),
            'predicted_text': row['text'], 'predicted_jamo': 'ㅊㅐㄱ',
            'baseline_key': audit.baseline_key(identity),
            **audit.catalog_identity(row), **identity}


def test_a_stale_row_that_cannot_be_recomputed_is_not_counted_as_covered(
        tmp_path, monkeypatch):
    """The pool is not covered, and the state file must not claim it is.

    The row's audio was replaced, so the stored record no longer describes
    it; the recomputation then fails. The old record is still in the file -
    counting it would report a finished audit that is not finished.
    """
    row = catalog_row(pcm='f' * 64)                  # audio moved on
    record = stored_record(row)
    record.update(audit.catalog_identity(row))       # target itself unchanged
    baseline = tmp_path / 'baseline.jsonl'
    baseline.write_text(json.dumps(record, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    state_path = tmp_path / 'baseline-state.json'
    monkeypatch.setattr(audit, 'WORK', tmp_path)
    monkeypatch.setattr(audit, 'BASELINE', baseline)
    monkeypatch.setattr(audit, 'STATE', state_path)
    stub_modules(monkeypatch, {'personal': [row]}, fail=True)

    state = audit.run_baseline(pools=('personal',))
    assert state['counts']['stale_recomputed'] == 1
    assert state['counts']['failed'] == 1
    assert state['counts']['job_complete'] is False
    assert state['pool_coverage']['personal'] == {
        'expected': 1, 'have': 0, 'stale_or_unusable': 1, 'complete': False}
    assert state['pools_complete'] is False
    assert json.loads(state_path.read_text(encoding='utf-8'))['status'] \
        == 'complete'


def test_a_run_says_it_is_running_before_it_processes_anything(tmp_path,
                                                               monkeypatch):
    """A crash must not leave the previous run's 'complete' standing."""
    row = catalog_row()
    baseline = tmp_path / 'baseline.jsonl'
    state_path = tmp_path / 'baseline-state.json'
    audit.write_state({'status': 'complete', 'counts': {}}, state_path)
    monkeypatch.setattr(audit, 'WORK', tmp_path)
    monkeypatch.setattr(audit, 'BASELINE', baseline)
    monkeypatch.setattr(audit, 'STATE', state_path)
    engine = stub_modules(monkeypatch, {'personal': [row]})

    seen = []

    def explode(waveform, pcm_sha256=None, use_cache=False):
        seen.append(json.loads(state_path.read_text(encoding='utf-8')))
        raise KeyboardInterrupt

    engine.transcribe = explode
    with pytest.raises(KeyboardInterrupt):
        audit.run_baseline(pools=('personal',))
    # While the job was running, that is what the file said.
    assert seen and seen[0]['status'] == 'running'
    # And afterwards it says how it ended, not that it finished.
    after = json.loads(state_path.read_text(encoding='utf-8'))
    assert after['status'] == 'interrupted'
    assert after['counts']['job_complete'] is False
    assert 'KeyboardInterrupt' in after['stopped_by']


# --- migration --------------------------------------------------------

def test_migration_refuses_records_from_another_computation(tmp_path,
                                                            monkeypatch):
    """Same weights, different input policy: not the same baseline.

    The old check only compared checkpoint hashes, so a record made under
    another policy was restamped with the first row's identity and quietly
    became part of this baseline.
    """
    row = catalog_row()
    other = catalog_row('personal:b')
    mine = stored_record(row)
    theirs = {**stored_record(other),
              'input_policy': 'amplitude-normalised, 8 kHz'}
    baseline = tmp_path / 'baseline.jsonl'
    write_lines(baseline, [mine, theirs])
    before = baseline.read_bytes()
    monkeypatch.setattr(audit, 'WORK', tmp_path)
    monkeypatch.setattr(audit, 'BASELINE', baseline)
    monkeypatch.setattr(audit, 'STATE', tmp_path / 'baseline-state.json')
    monkeypatch.setattr(audit, 'digest', lambda path: 'a' * 64)
    monkeypatch.setattr(audit.labels, 'contract_sha256', lambda: 'b' * 64)
    stub_modules(monkeypatch, {'personal': [row, other]})

    out = audit.migrate_baseline(verify_audio=False)
    assert out['refused'] and out['migrated'] == 0
    assert out['conflicting'][0]['problems'] == ['input_policy']
    assert baseline.read_bytes() == before           # nothing restamped


def test_migration_refuses_a_record_whose_target_moved(tmp_path, monkeypatch):
    """A drifted transcript is reported, not overwritten into agreement."""
    row = catalog_row()
    record = stored_record(row)
    record['transcript'] = '사과'                     # scored against another
    baseline = tmp_path / 'baseline.jsonl'
    write_lines(baseline, [record])
    monkeypatch.setattr(audit, 'WORK', tmp_path)
    monkeypatch.setattr(audit, 'BASELINE', baseline)
    monkeypatch.setattr(audit, 'STATE', tmp_path / 'baseline-state.json')
    monkeypatch.setattr(audit, 'digest', lambda path: 'a' * 64)
    monkeypatch.setattr(audit.labels, 'contract_sha256', lambda: 'b' * 64)
    stub_modules(monkeypatch, {'personal': [row]})

    out = audit.migrate_baseline(verify_audio=False)
    assert out['migrated'] == 0
    assert out['rejected'][0]['problems'] == ['transcript']


def test_migration_fills_a_field_a_legacy_record_simply_lacks(tmp_path,
                                                              monkeypatch):
    """The case migration exists for, still working after the tightening."""
    row = catalog_row()
    record = stored_record(row)
    record.pop('input_policy')                       # written before it existed
    record.pop('binding', None)
    baseline = tmp_path / 'baseline.jsonl'
    write_lines(baseline, [record])
    monkeypatch.setattr(audit, 'WORK', tmp_path)
    monkeypatch.setattr(audit, 'BASELINE', baseline)
    monkeypatch.setattr(audit, 'STATE', tmp_path / 'baseline-state.json')
    monkeypatch.setattr(audit, 'digest', lambda path: 'a' * 64)
    monkeypatch.setattr(audit.labels, 'contract_sha256', lambda: 'b' * 64)
    stub_modules(monkeypatch, {'personal': [row]})

    out = audit.migrate_baseline(verify_audio=False)
    assert out['migrated'] == 1 and not out['rejected']
    written, _ = audit.load_baseline(path=baseline)
    assert written[row['id']]['input_policy'] == audit.INPUT_POLICY
    assert written[row['id']]['binding']
