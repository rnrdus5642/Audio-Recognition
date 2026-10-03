"""The review store and its routes, against a throwaway database.

Every test here writes to a temporary file. The reviewer's real answers in
`consonant-review/reviews.sqlite3` are never opened, let alone written to -
a QA run that left fake annotations behind would corrupt the very thing
this workflow exists to collect.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_web/test_review.py -q
"""
from pathlib import Path
import json
import sys

import pytest

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))
import review as review_module                                 # noqa: E402


@pytest.fixture
def store(tmp_path):
    return review_module.Store(tmp_path / 'reviews.sqlite3')


def annotate(store, record_id='personal:child1_18', **overrides):
    fields = {
        'record_id': record_id, 'pool': 'personal', 'status': 'answered',
        'choice': 'other', 'heard_text': '샙', 'note': None, 'flags': [],
        'audio_pcm_sha256': 'a' * 64, 'baseline_key': 'key',
        'binding': 'bind-1', 'reference_ids': [1, 2, 3],
        'reference_surface': '책', 'predicted_text': '샙',
    }
    fields.update(overrides)
    return store.save(**fields)


# --- the store --------------------------------------------------------

def test_the_real_database_is_never_touched_by_tests(store):
    assert store.path != review_module.DEFAULT_DB
    assert 'reviews.sqlite3' in str(store.path)


def test_an_answer_is_saved_and_read_back(store):
    annotate(store)
    current = store.current('personal:child1_18')
    assert current['status'] == 'answered' and current['heard_text'] == '샙'
    assert current['revisions'] == 1


def test_editing_adds_a_revision_and_keeps_the_old_one(store):
    annotate(store, heard_text='샙')
    annotate(store, heard_text='책')
    current = store.current('personal:child1_18')
    assert current['heard_text'] == '책' and current['revisions'] == 2
    history = store.history('personal:child1_18')
    assert [h['heard_text'] for h in history] == ['샙', '책']


def test_free_text_is_kept_exactly_as_typed(store):
    """No G2P, no normalisation, no correction towards anything."""
    for typed in ('ㅅㅐㅂ', '샙 ', '채ㄱ', 'sae'):
        annotate(store, record_id=f'x:{typed}', heard_text=typed)
        assert store.current(f'x:{typed}')['heard_text'] == typed


def test_flags_survive_a_round_trip(store):
    annotate(store, status='unusable', choice=None, heard_text=None,
             flags=['boundary', 'noise'])
    assert store.current('personal:child1_18')['flags'] == ['boundary', 'noise']


def test_annotations_persist_across_a_reopen(tmp_path):
    path = tmp_path / 'reviews.sqlite3'
    annotate(review_module.Store(path))
    # A new process would do exactly this: open the same file again.
    assert review_module.Store(path).current(
        'personal:child1_18')['heard_text'] == '샙'


# --- export -----------------------------------------------------------

def index_for(record_id='personal:child1_18', **overrides):
    record = {
        'id': record_id, 'pool': 'personal', 'speaker': 'child1',
        'child': 'child1', 'seconds': 0.87, 'transcript': '책',
        'reference_surface': '책', 'predicted_text': '샙',
        'consonant_events': [], 'consonant_mistakes': 2,
        'metrics': {'errors': 2, 'reference_jamo': 3, 'exact': False},
        'roles_trusted': True, 'audio_pcm_sha256': 'a' * 64,
        'baseline_key': 'key', 'binding': 'bind-1',
    }
    record.update(overrides)
    return {'records': {record_id: record}, 'shortlist': [], 'ready': True}


def test_export_separates_answers_from_could_not_tell(store):
    annotate(store, record_id='a', status='answered')
    annotate(store, record_id='b', status='unclear', choice=None,
             heard_text=None)
    annotate(store, record_id='c', status='unusable', choice=None,
             heard_text=None, flags=['noise'])
    index = {'records': {k: index_for(k)['records'][k] for k in 'abc'},
             'shortlist': [], 'ready': True}
    out = review_module.export(index, store)
    assert out['counts'] == {'accepted': 1, 'set_aside': 2, 'stale': 0}
    assert [e['record_id'] for e in out['accepted']] == ['a']
    assert {e['record_id'] for e in out['set_aside']} == {'b', 'c'}


def test_an_annotation_for_changed_audio_is_marked_stale_not_accepted(store):
    annotate(store, record_id='a', audio_pcm_sha256='a' * 64)
    index = index_for('a')
    index['records']['a']['binding'] = 'bind-2'                # audio changed
    out = review_module.export(index, store)
    assert out['counts']['accepted'] == 0 and out['counts']['stale'] == 1
    assert '바뀌었습니다' in out['stale'][0]['stale_reason']


def test_an_annotation_for_a_changed_baseline_is_marked_stale(store):
    annotate(store, record_id='a', binding='old-binding')
    index = index_for('a')                                     # binding bind-1
    out = review_module.export(index, store)
    assert out['counts']['stale'] == 1


def test_an_answer_whose_record_vanished_is_stale_not_accepted(store):
    """The defect: `if record and ...` let a missing record through."""
    annotate(store, record_id='gone', status='answered')
    index = {'records': {}, 'shortlist': [], 'ready': True}
    out = review_module.export(index, store)
    assert out['counts'] == {'accepted': 0, 'set_aside': 0, 'stale': 1}
    assert out['stale'][0]['stale_code'] == 'baseline_record_missing'


def test_every_exported_row_is_marked_not_for_training(store):
    annotate(store, record_id='a')
    annotate(store, record_id='b', status='unclear', choice=None,
             heard_text=None)
    out = review_module.export(index_for('a'), store)
    assert out['training_eligible'] is False
    for bucket in ('accepted', 'set_aside', 'stale'):
        assert all(e['training_eligible'] is False for e in out[bucket])


def test_a_bound_option_exports_the_text_that_was_on_the_button(store):
    annotate(store, record_id='a', choice='expected', heard_text=None)
    out = review_module.export(index_for('a'), store)
    entry = out['accepted'][0]
    assert entry['heard_form'] == '책'
    assert entry['heard_form_source'] == 'bound_option_expected_surface'


def test_a_custom_answer_exports_verbatim_including_spaces(store):
    annotate(store, record_id='a', choice='other', heard_text='  샙 비슷  ')
    entry = review_module.export(index_for('a'), store)['accepted'][0]
    assert entry['heard_form'] == '  샙 비슷  '
    assert entry['heard_form_source'] == 'typed_by_reviewer_verbatim'


# --- concurrent edits -------------------------------------------------

def test_a_stale_tab_cannot_overwrite_a_newer_answer(store):
    first = annotate(store, heard_text='첫 답')
    annotate(store, heard_text='나중 답')                   # someone else saved
    with pytest.raises(review_module.Conflict) as caught:
        # The old tab still believes it is at the first revision.
        annotate(store, heard_text='오래된 탭', expected_revision=first['revision'])
    assert caught.value.current_revision > first['revision']
    assert store.current('personal:child1_18')['heard_text'] == '나중 답'


def test_saving_against_the_current_revision_succeeds(store):
    saved = annotate(store, heard_text='첫 답')
    again = annotate(store, heard_text='고침',
                     expected_revision=saved['revision'])
    assert again['heard_text'] == '고침' and again['revisions'] == 2


def test_the_first_answer_expects_revision_zero(store):
    saved = annotate(store, expected_revision=0)
    assert saved['revisions'] == 1


def test_export_says_what_it_is_not(store):
    annotate(store, record_id='a')
    out = review_module.export(index_for('a'), store)
    assert '전문가' in out['what_this_is']
    assert '그대로' in out['accepted'][0]['heard_text_note']


# --- the shortlist ----------------------------------------------------

def fake_records(count=40):
    records = {}
    pairs = ['ㅊ>ㅅ', 'ㄱ>ㅂ', 'ㄸ>ㄱ', 'ㅅ>ㅈ', 'ㄹ>ㄴ', 'ㅂ>ㄹ']
    for i in range(count):
        pool = 'personal' if i % 2 else 'valid'
        # Alternate the two children across the personal half, so a test
        # about covering both is actually testing something.
        speaker = f'child{1 + (i // 2) % 2}' if pool == 'personal' else f'spk{i}'
        wrong = i % 3 != 0
        pair = pairs[i % len(pairs)]
        reference, hypothesis = pair.split('>')
        records[f'{pool}:{i:03d}'] = {
            'id': f'{pool}:{i:03d}', 'pool': pool, 'speaker': speaker,
            'child': speaker if pool == 'personal' else None,
            'seconds': 1.0, 'transcript': f'말{i}',
            'reference_surface': f'말{i}', 'predicted_text': f'말{i}',
            'consonant_events': ([{'op': 'substitute', 'reference': reference,
                                   'hypothesis': hypothesis, 'role': 'onset'}]
                                 if wrong else []),
            'consonant_mistakes': 1 if wrong else 0,
            'metrics': {'errors': 1 if wrong else 0, 'reference_jamo': 3,
                        'exact': not wrong},
            'roles_trusted': True, 'audio_pcm_sha256': 'a' * 64,
            'baseline_key': 'key',
        }
    return records


def test_the_shortlist_is_small_and_deterministic():
    records = fake_records()
    first = review_module.build_shortlist(records)
    assert 15 <= len(first) <= 20
    assert first == review_module.build_shortlist(records)


def test_the_shortlist_covers_both_sources_and_some_controls():
    shortlist = review_module.build_shortlist(fake_records())
    pools = {item['pool'] for item in shortlist}
    assert pools == {'personal', 'valid'}
    assert any('대조군' in item['reason'] for item in shortlist)
    # Several different confusions, not the same one over and over.
    confusions = {item['reason'] for item in shortlist
                  if item['reason'].startswith('자음 혼동')}
    assert len(confusions) >= 4


def test_both_children_are_represented_in_the_personal_picks():
    """A review that only ever hears one child answers half the question."""
    shortlist = review_module.build_shortlist(fake_records())
    children = {item['speaker'] for item in shortlist
                if item['pool'] == 'personal'}
    assert children == {'child1', 'child2'}


# --- index and shortlist coherence ------------------------------------

def test_a_fixture_index_never_touches_the_live_directory(tmp_path):
    """The defect: load_index(path=...) still wrote the live shortlist."""
    live = review_module.WORK / 'shortlist.json'
    before = live.read_bytes() if live.exists() else None
    per_record = tmp_path / 'per-record.json'
    per_record.write_text(json.dumps(fake_records()), encoding='utf-8')
    index = review_module.load_index(path=per_record, work=tmp_path)
    assert index['ready'] and index['shortlist']
    assert (tmp_path / 'shortlist.json').exists()
    after = live.read_bytes() if live.exists() else None
    assert before == after


def test_a_fixture_index_alone_still_never_touches_the_live_directory(tmp_path):
    """The remaining half of it: `path` given, `work` left out.

    A caller who names an index names where its artefacts live. Falling
    back to the live directory meant a fixture of fake records rebuilt the
    reviewer's real shortlist, over the real one, from six made-up items.
    """
    live_shortlist = review_module.WORK / 'shortlist.json'
    live_index = review_module.WORK / 'per-record.json'
    before = live_shortlist.read_bytes() if live_shortlist.exists() else None
    before_index = live_index.read_bytes() if live_index.exists() else None

    per_record = tmp_path / 'per-record.json'
    per_record.write_text(json.dumps(fake_records()), encoding='utf-8')
    index = review_module.load_index(path=per_record)     # no `work`

    assert index['ready'] and index['shortlist']
    assert (tmp_path / 'shortlist.json').exists()
    after = live_shortlist.read_bytes() if live_shortlist.exists() else None
    after_index = live_index.read_bytes() if live_index.exists() else None
    assert before == after and before_index == after_index


def test_a_shortlist_from_a_smaller_audit_is_rebuilt(tmp_path):
    """A six-record smoke list must not survive the full audit."""
    small = dict(list(fake_records().items())[:6])
    (tmp_path / 'per-record.json').write_text(json.dumps(small),
                                              encoding='utf-8')
    first = review_module.load_index(work=tmp_path)['shortlist']
    saved = json.loads((tmp_path / 'shortlist.json').read_text(encoding='utf-8'))
    assert saved['selection_version'] == review_module.SELECTION_VERSION

    (tmp_path / 'per-record.json').write_text(json.dumps(fake_records()),
                                              encoding='utf-8')
    second = review_module.load_index(work=tmp_path)['shortlist']
    assert [i['id'] for i in second] != [i['id'] for i in first]
    assert len(second) > len(first)


def test_a_shortlist_from_an_older_selection_rule_is_rebuilt(tmp_path):
    records = fake_records()
    (tmp_path / 'per-record.json').write_text(json.dumps(records),
                                              encoding='utf-8')
    review_module.load_index(work=tmp_path)
    path = tmp_path / 'shortlist.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    saved['selection_version'] = 0
    saved['items'] = saved['items'][:1]
    path.write_text(json.dumps(saved), encoding='utf-8')
    assert len(review_module.load_index(work=tmp_path)['shortlist']) > 1


def test_a_half_written_shortlist_is_rebuilt_not_served(tmp_path):
    (tmp_path / 'per-record.json').write_text(json.dumps(fake_records()),
                                              encoding='utf-8')
    (tmp_path / 'shortlist.json').write_text('{"items": [', encoding='utf-8')
    assert review_module.load_index(work=tmp_path)['shortlist']


def test_the_known_cases_come_first_when_present():
    records = fake_records()
    records['personal:child1_18'] = dict(
        records['personal:001'], id='personal:child1_18', pool='personal',
        speaker='child1')
    shortlist = review_module.build_shortlist(records)
    assert shortlist[0]['id'] == 'personal:child1_18'
    assert '이전 분석' in shortlist[0]['reason']
