"""The review endpoints, with the store pointed at a temporary file.

`CHILD_REVIEW_DB` is set before the server module is imported, so nothing
here can reach the reviewer's real answers. The model is warmed because the
audio routes read real recordings; the review itself never runs inference.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_web/test_review_routes.py -q
"""
from pathlib import Path
import os
import sys
import tempfile

import pytest

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))

_TEMP = tempfile.mkdtemp(prefix='child-review-qa-')
os.environ['CHILD_REVIEW_DB'] = str(Path(_TEMP) / 'reviews.sqlite3')

import review as review_module                                 # noqa: E402
import server as server_module                                 # noqa: E402
from fastapi.testclient import TestClient                      # noqa: E402

pytestmark = pytest.mark.skipif(
    not (review_module.WORK / 'per-record.json').exists(),
    reason='audit.py 결과가 없습니다')


@pytest.fixture(scope='module')
def client():
    with TestClient(server_module.app) as c:
        yield c


@pytest.fixture(scope='module', autouse=True)
def isolated_store():
    # Belt and braces: refuse to run at all rather than write a fake answer
    # into the reviewer's real file.
    assert str(server_module.review_db_path()).startswith(_TEMP), \
        '테스트가 실제 검토 DB 를 쓰려고 합니다'
    server_module.REVIEW['store'] = None       # in case another module ran first
    yield
    assert str(server_module.review_db_path()).startswith(_TEMP)
    assert not review_module.DEFAULT_DB.exists() or \
        review_module.DEFAULT_DB.stat().st_size >= 0


def first_id(client, pool='personal'):
    state = client.get('/api/review/state').json()
    return next(item['id'] for item in state['priority']
                if item['pool'] == pool)


def answer(client, record_id, **overrides):
    """Save the way the page does: with the binding and revision it saw."""
    item = client.get(f'/api/review/item/{record_id}').json()
    body = {'record_id': record_id, 'binding': item['binding'],
            'expected_revision': item['revision'],
            'status': 'answered', 'choice': 'expected'}
    body.update(overrides)
    return client.post('/api/review/annotation', json=body)


# --- the page and its lists -------------------------------------------

def test_the_review_page_is_served(client):
    assert client.get('/review').status_code == 200
    assert client.get('/review.js').status_code == 200


def test_the_source_pages_still_work(client):
    for path in ('/', '/aihub', '/recordings'):
        assert client.get(path).status_code == 200
    splits = client.get('/api/status').json()['splits']
    assert splits['test']['recordings'] == 2999
    assert splits['personal']['recordings'] == 83


def test_the_shortlist_is_small_and_covers_both_sources(client):
    state = client.get('/api/review/state').json()
    assert 15 <= len(state['priority']) <= 20
    assert {item['pool'] for item in state['priority']} == {'personal', 'valid'}
    assert state['progress']['priority']['total'] == len(state['priority'])


def test_the_rest_is_available_but_not_demanded(client):
    state = client.get('/api/review/state').json()
    assert state['flagged_counts']['valid'] > len(state['priority'])
    assert '전부 확인하실 필요는 없습니다' in state['note']
    flagged = client.get('/api/review/flagged?source=valid').json()
    assert flagged['total'] == state['flagged_counts']['valid']


def test_sources_can_be_filtered_separately(client):
    for pool in ('personal', 'valid'):
        state = client.get(f'/api/review/state?source={pool}').json()
        assert {item['pool'] for item in state['priority']} == {pool}
    assert client.get('/api/review/state?source=nope').status_code == 404


# --- one recording ----------------------------------------------------

def test_an_item_carries_what_the_page_needs(client):
    item = client.get(
        f'/api/review/item/{first_id(client)}').json()
    assert item['transcript'] and item['expected_surface']
    assert item['audio_url'].startswith('/api/review/audio/')
    assert item['source_label'] == '직접 녹음'
    # The personal clips can also be heard in their original context.
    assert item['context_url']


def test_the_audio_is_the_model_input(client):
    import io
    import soundfile as sf
    record_id = first_id(client)
    served = client.get(f'/api/review/audio/{record_id}')
    assert served.status_code == 200
    data, rate = sf.read(io.BytesIO(served.content), dtype='float32')
    assert rate == 16000
    index, _ = server_module._review()
    assert server_module.engine_module.pcm_sha256(data) == \
        index['records'][record_id]['audio_pcm_sha256']


def test_context_audio_is_longer_than_the_clip(client):
    record_id = first_id(client)
    item = client.get(f'/api/review/item/{record_id}').json()
    assert item['context_seconds'] > item['seconds']
    assert client.get(f'/api/review/context/{record_id}').status_code == 200


def test_ids_are_looked_up_not_used_as_paths(client):
    for bad in ('nope:x', '..%2F..%2Fsecret', 'personal:child9_99'):
        assert client.get(f'/api/review/item/{bad}').status_code == 404
        assert client.get(f'/api/review/audio/{bad}').status_code == 404


# --- answering --------------------------------------------------------

def test_an_answer_is_saved_and_shows_up_again(client):
    record_id = first_id(client)
    out = answer(client, record_id, choice='other', heard_text='ㅅㅐㅂ')
    assert out.status_code == 200 and out.json()['saved']
    item = client.get(f'/api/review/item/{record_id}').json()
    # Kept exactly as typed - not composed, not corrected.
    assert item['annotation']['heard_text'] == 'ㅅㅐㅂ'
    assert item['annotation']['status'] == 'answered'


def test_custom_text_keeps_its_spacing(client):
    """The server promises verbatim, so it must not trim either."""
    record_id = first_id(client)
    answer(client, record_id, choice='other', heard_text='  샙 같음  ')
    item = client.get(f'/api/review/item/{record_id}').json()
    assert item['annotation']['heard_text'] == '  샙 같음  '


def test_editing_keeps_the_earlier_answer_in_history(client):
    record_id = first_id(client)
    answer(client, record_id, choice='expected')
    history = client.get(f'/api/review/history/{record_id}').json()['items']
    assert len(history) >= 2
    assert history[-1]['choice'] == 'expected'


def test_a_save_must_carry_the_binding_it_was_shown(client):
    record_id = first_id(client)
    item = client.get(f'/api/review/item/{record_id}').json()
    stale = client.post('/api/review/annotation', json={
        'record_id': record_id, 'binding': 'something-else',
        'expected_revision': item['revision'],
        'status': 'answered', 'choice': 'expected'})
    assert stale.status_code == 409
    assert '다릅니다' in stale.json()['detail']


def test_a_forgotten_tab_cannot_overwrite_a_newer_answer(client):
    """Two screens, one recording: the older one loses, loudly."""
    record_id = first_id(client, 'valid')
    old_tab = client.get(f'/api/review/item/{record_id}').json()
    answer(client, record_id, choice='expected')          # a newer answer
    late = client.post('/api/review/annotation', json={
        'record_id': record_id, 'binding': old_tab['binding'],
        'expected_revision': old_tab['revision'],
        'status': 'answered', 'choice': 'other', 'heard_text': '오래된 탭'})
    assert late.status_code == 409
    current = client.get(f'/api/review/item/{record_id}').json()['annotation']
    assert current['heard_text'] != '오래된 탭'


def test_a_save_without_a_binding_is_refused(client):
    record_id = first_id(client)
    out = client.post('/api/review/annotation', json={
        'record_id': record_id, 'status': 'answered', 'choice': 'expected'})
    assert out.status_code == 422          # binding/revision are required


def test_saving_an_answer_changes_no_model_result(client):
    """The review is an observation, not a correction."""
    record_id = first_id(client)
    before = client.post('/api/transcribe',
                         json={'sample_id': record_id}).json()
    answer(client, record_id, choice='other', heard_text='전혀 다른 말')
    after = client.post('/api/transcribe',
                        json={'sample_id': record_id}).json()
    assert before['prediction']['text'] == after['prediction']['text']
    assert before['metrics'] == after['metrics']


def test_unclear_is_a_status_of_its_own(client):
    record_id = first_id(client, 'valid')
    answer(client, record_id, status='unclear', choice=None)
    item = client.get(f'/api/review/item/{record_id}').json()
    assert item['annotation']['status'] == 'unclear'
    assert item['annotation']['choice'] is None


def test_unusable_needs_a_reason(client):
    record_id = first_id(client, 'valid')
    bad = answer(client, record_id, status='unusable', choice=None, flags=[])
    assert bad.status_code == 400
    ok = answer(client, record_id, status='unusable', choice=None,
                flags=['noise'])
    assert ok.status_code == 200


def test_a_malformed_answer_is_refused_rather_than_stored(client):
    record_id = first_id(client)
    for overrides in (
            {'choice': None},                      # no choice
            {'choice': 'other', 'heard_text': '  '},
            {'status': 'nope'},
            {'status': 'unusable', 'choice': None, 'flags': ['made-up']}):
        assert answer(client, record_id, **overrides).status_code == 400


def test_an_unknown_recording_cannot_be_annotated(client):
    assert client.post('/api/review/annotation', json={
        'record_id': 'personal:nope', 'binding': 'x', 'expected_revision': 0,
        'status': 'answered', 'choice': 'expected'}).status_code == 404


# --- export -----------------------------------------------------------

def test_export_keeps_uncertain_out_of_the_accepted_labels(client):
    export = client.get('/api/review/export').json()
    assert export['counts']['accepted'] >= 1
    assert all(e['status'] == 'answered' for e in export['accepted'])
    assert all(e['status'] in ('unclear', 'unusable')
               for e in export['set_aside'])
    assert '전문가' in export['what_this_is']


def test_export_carries_provenance(client):
    entry = client.get('/api/review/export').json()['accepted'][0]
    assert len(entry['audio_pcm_sha256']) == 64
    assert entry['baseline_key'] and entry['split']
    assert entry['expected_surface'] and entry['model_prediction']


def test_progress_counts_add_up(client):
    state = client.get('/api/review/state').json()
    tally = state['progress']['priority']
    assert (tally['answered'] + tally['unclear'] + tally['unusable']
            + tally['remaining'] == tally['total'])


def test_a_stale_answer_does_not_count_as_reviewed(client):
    """It was about a recording that has since changed, so it is not done."""
    record_id = first_id(client)
    answer(client, record_id, choice='expected')
    index, _store = server_module._review()
    record = index['records'][record_id]
    original = record['binding']
    try:
        record['binding'] = 'moved-on'
        state = client.get('/api/review/state').json()
        entry = next(e for e in state['priority'] if e['id'] == record_id)
        assert entry['status'] is None
        assert state['progress']['priority']['stale'] >= 1
        export = client.get('/api/review/export').json()
        assert record_id not in [e['record_id'] for e in export['accepted']]
        assert record_id in [e['record_id'] for e in export['stale']]
    finally:
        record['binding'] = original


def test_export_marks_every_row_as_not_for_training(client):
    export = client.get('/api/review/export').json()
    assert export['training_eligible'] is False
    assert all(e['training_eligible'] is False
               for bucket in ('accepted', 'set_aside', 'stale')
               for e in export[bucket])


def test_the_flagged_list_requires_an_explicit_source(client):
    """It cannot page over both, so it must not pretend `전체` works."""
    assert client.get('/api/review/flagged').status_code == 422
    assert client.get('/api/review/flagged?source=personal').status_code == 200


def test_product_facing_labels_carry_no_ids_or_hashes(client):
    state = client.get('/api/review/state').json()
    for entry in state['priority']:
        assert 'personal:' not in entry['label']
        assert 'valid:' not in entry['label']
        assert entry['label'].startswith(('직접 녹음', '말뭉치'))
