"""The personally recorded datasets: provenance, coverage, and no fake score.

These run against the real files on the Desktop and the real checkpoint,
because the things worth checking here are exactly the ones a stub would
paper over: that all 83 clips are present and match their recorded hashes,
that the transcript label is the label (and the elicitation's expected
answer is not), that a whole recording is chunked so that every sample is
included exactly once, and that a recording with no reference produces
"not comparable" rather than a 0% error rate.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_web/test_personal.py -q
"""
from pathlib import Path
import sys

import numpy as np
import pytest

WEB = Path(__file__).resolve().parent
sys.path.insert(0, str(WEB))
sys.path.insert(0, str(WEB.parents[0] / 'child_jamo_finetune'))
import engine as engine_module
import personal
import samples as samples_module
import server as server_module
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not personal.available(),
                                reason='직접 녹음한 원본 폴더가 없습니다')


@pytest.fixture(scope='module')
def client():
    with TestClient(server_module.app) as c:
        c.post('/api/warm').raise_for_status()
        yield c


@pytest.fixture(scope='module')
def clips():
    return personal.clip_rows()


# --- the 83 labelled clips -------------------------------------------

def test_every_labelled_clip_is_present_and_matches_its_hashes(clips):
    """`clip_rows` raises on a mismatch, so reaching here is the check."""
    assert len(clips) == 83
    speakers = {}
    for row in clips:
        speakers[row['speaker']] = speakers.get(row['speaker'], 0) + 1
    assert speakers == {'child1': 48, 'child2': 35}
    assert len({r['id'] for r in clips}) == 83


def test_clip_audio_is_checked_against_the_catalogue_not_just_its_length(clips):
    row = dict(clips[0])
    engine_module.read_row(row)                    # the real one is fine
    row['pcm_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='달라졌습니다'):
        engine_module.read_row(row)


def test_no_clip_path_escapes_the_source_folder(clips):
    root = (personal.SOURCE_ROOT / personal.LABELLED_DIR).resolve()
    for row in clips:
        assert Path(row['path']).resolve().is_relative_to(root)


def test_the_transcript_is_the_label_and_the_expected_answer_is_not(clips):
    by_id = {r['id']: r for r in clips}
    # The child said 야옹이; the question was fishing for 고양이. The label
    # stays 야옹이, and 고양이 is carried alongside as the annotation.
    meow = by_id['personal:child1_06']
    assert meow['text'] == '야옹이'
    assert meow['source']['expected_answer'] == '고양이'
    # The transcript's own spelling is kept, not silently corrected.
    assert by_id['personal:child1_48']['text'] == '두르려'
    # A multi-word utterance stays one utterance.
    assert '북을 쳐요' in {r['text'] for r in clips}


def test_expected_answer_never_reaches_the_reference(clips):
    for row in clips:
        answer = row['source']['expected_answer']
        if answer and answer != row['text']:
            reference = server_module._reference(row)
            assert reference['transcript'] == row['text'] != answer


def test_clips_are_marked_as_not_listened_through(clips):
    assert not any(r['source']['human_reviewed'] for r in clips)
    assert {r['id'] for r in clips if r['source']['priority_review']} == {
        'personal:child1_48', 'personal:child2_06', 'personal:child2_07'}


def test_no_age_is_invented(clips):
    assert {r['age'] for r in clips} == {'unknown'}


# --- the four whole recordings ---------------------------------------

def test_all_four_recordings_are_offered_whole(client):
    found = client.get('/api/samples?split=original&size=24').json()
    assert found['total'] == 4
    names = {i['wav'] for i in found['items']}
    assert len(names) == 4
    assert sum('자유대화' in n for n in names) == 2
    # Whole: each row's duration is the recording's, not a trimmed span.
    for item in found['items']:
        assert item['seconds'] > 55


def test_chunks_cover_every_sample_exactly_once_and_stay_bounded(client):
    for item in client.get('/api/samples?split=original').json()['items']:
        sample = client.get(f'/api/sample/{item["id"]}').json()
        chunks = sample['chunks']
        assert chunks['covers_samples'] == chunks['total_samples']
        segments = chunks['segments']
        assert segments[0]['start_s'] == 0
        assert all(a['end_s'] == b['start_s']
                   for a, b in zip(segments, segments[1:]))
        assert max(s['seconds'] for s in segments) <= personal.CHUNK_MAX_SECONDS


def test_chunk_plan_is_contiguous_for_a_synthetic_waveform():
    """The property, without depending on any particular recording."""
    total = 16000 * 97
    plan = personal.plan_segments(total, np.zeros(total, dtype=np.float32))
    assert plan[0]['start'] == 0 and plan[-1]['end'] == total
    assert sum(s['samples'] for s in plan) == total
    assert all(a['end'] == b['start'] for a, b in zip(plan, plan[1:]))


def test_a_whole_recording_is_refused_by_the_single_shot_route(client):
    item = client.get('/api/samples?split=original').json()['items'][0]
    out = client.post('/api/transcribe', json={'sample_id': item['id']})
    assert out.status_code == 400 and '구간별' in out.json()['detail']


def test_whole_recordings_are_not_offered_to_the_batch(client):
    out = client.get('/api/batch-scope?split=original&count=10')
    assert out.status_code == 400
    assert client.get('/api/status').json()['splits']['original']['batchable'] \
        is False


def test_served_original_audio_is_the_array_the_chunks_are_cut_from(client):
    import io
    import soundfile as sf
    item = client.get('/api/samples?split=original').json()['items'][0]
    served = client.get(f'/api/audio/{item["id"]}')
    assert served.status_code == 200
    data, rate = sf.read(io.BytesIO(served.content), dtype='float32')
    assert rate == 16000
    row = server_module.CATALOG.get(item['id'])
    assert engine_module.pcm_sha256(data) == row['pcm_sha256']
    assert len(data) == row['samples']


# --- no reference means no score -------------------------------------

def test_a_recording_with_no_reference_is_not_given_a_rate(client):
    item = client.get('/api/samples?split=original').json()['items'][0]
    row = server_module.CATALOG.get(item['id'])
    assert server_module._reference(row) is None
    assert server_module._reference_note(row)['state'] == 'none'


def test_segmented_transcription_reports_full_coverage(client):
    """The shortest recording, actually recognised, end to end."""
    items = client.get('/api/samples?split=original').json()['items']
    shortest = min(items, key=lambda i: i['seconds'])
    out = client.post('/api/transcribe-original',
                      json={'sample_id': shortest['id']})
    assert out.status_code == 200, out.text
    body = out.json()
    assert body['reference'] is None
    assert body['coverage']['complete']
    assert body['coverage']['covered_samples'] == body['coverage']['total_samples']
    assert len(body['segments']) == body['coverage']['segments']
    # Timestamps are contiguous and in the recording's own timebase.
    assert body['segments'][0]['start_s'] == 0
    assert abs(body['segments'][-1]['end_s'] - shortest['seconds']) < 0.01
    # Something was actually decoded, and no metric was invented for it.
    assert any(s['prediction']['text'].strip() for s in body['segments'])


def test_batch_does_not_average_in_an_unscored_recording(client):
    ids = [i['id'] for i in
           client.get('/api/samples?split=personal&size=3').json()['items']]
    out = client.post('/api/batch', json={'sample_ids': ids}).json()
    assert out['aggregate']['scored'] == len(out['results'])
    assert out['aggregate']['unscored'] == 0
    assert out['aggregate']['jer'] is not None


# --- the datasets in the picker --------------------------------------

def test_the_two_personal_datasets_are_separate_and_grouped(client):
    splits = client.get('/api/status').json()['splits']
    assert splits['personal']['recordings'] == 83
    assert splits['original']['recordings'] == 4
    assert splits['personal']['group'] == splits['original']['group']
    assert splits['personal']['group'] != splits['test']['group']
    # The corpus counts are untouched by any of this.
    assert splits['test']['recordings'] == 2999
    assert splits['valid']['recordings'] == 1500


def test_the_child_filter_selects_one_child(client):
    for child, count in (('child1', 48), ('child2', 35)):
        found = client.get(f'/api/samples?split=personal&child={child}').json()
        assert found['total'] == count
        assert all(i['child'] == child for i in found['items'])


# --- the two sources are two pages ------------------------------------

def test_each_source_page_is_served_and_owns_its_splits(client):
    for path in ('/', '/aihub', '/recordings'):
        assert client.get(path).status_code == 200
    assert samples_module.splits_of('aihub') == ['test', 'valid', 'train']
    assert samples_module.splits_of('recordings') == ['personal', 'original']
    sources = client.get('/api/status').json()['sources']
    assert sources['aihub']['default'] == 'test'
    assert sources['recordings']['default'] == 'personal'
    # Every split belongs to exactly one page.
    splits = client.get('/api/status').json()['splits']
    assert {s['source'] for s in splits.values()} == {'aihub', 'recordings'}


def test_history_filters_by_source_before_it_takes_the_last_fifty(client):
    """The filter has to come first, or a busy page evicts the quiet one."""
    saved = list(server_module.HISTORY)
    try:
        server_module.HISTORY.clear()
        server_module.HISTORY.append(
            {'sample_id': 'personal:child1_01', 'split': 'personal',
             'metrics': None, 'at': 'x', 'text': '아빠', 'transcript': '아빠'})
        # More corpus runs than the endpoint will ever return, all newer.
        for i in range(60):
            server_module.HISTORY.append(
                {'sample_id': f'test:{i}.wav', 'split': 'test',
                 'metrics': None, 'at': 'x', 'text': 'x', 'transcript': 'x'})

        personal_rows = client.get('/api/history?source=recordings').json()['items']
        # Filtering after slicing would have lost this row entirely.
        assert [r['sample_id'] for r in personal_rows] == ['personal:child1_01']

        corpus_rows = client.get('/api/history?source=aihub').json()['items']
        assert len(corpus_rows) == 50
        assert all(r['split'] == 'test' for r in corpus_rows)
        # Unfiltered stays as it was: the newest fifty of everything.
        assert len(client.get('/api/history').json()['items']) == 50
    finally:
        server_module.HISTORY[:] = saved


def test_an_unknown_source_is_a_404_not_an_empty_list(client):
    assert client.get('/api/history?source=nope').status_code == 404
    assert client.get('/api/history?source=personal').status_code == 404


def test_an_unknown_dataset_name_is_still_a_404(client):
    assert client.get('/api/samples?split=personal2').status_code == 404
    assert client.post('/api/transcribe-original',
                       json={'sample_id': 'personal:nope'}).status_code == 404
    assert client.post(
        '/api/transcribe-original',
        json={'sample_id': samples_module.SPLITS and 'test:missing.wav'}
    ).status_code == 404
