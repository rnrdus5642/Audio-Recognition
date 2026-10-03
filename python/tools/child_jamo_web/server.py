"""Local viewer for the fine-tuned child jamo model.

Binds 127.0.0.1 only. Audio is streamed from the original archives
read-only, by manifest offset, and the bytes the browser plays are
encoded from exactly the array the model was given - so what you hear and
what was transcribed cannot drift apart.

    python python/tools/child_jamo_web/server.py --port 8767
"""
from pathlib import Path
import argparse
import json
import os
import sys
import threading
import time
import uuid

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_jamo_finetune'))
import samples as samples_module
import compose
import engine as engine_module
import labels
import personal
import audit as audit_module
import review as review_module
from common import ROOT

WORK = ROOT / '.venv/child-jamo-web-20260916'
STATIC = Path(__file__).resolve().parent / 'static'
UPLOADS = WORK / 'uploads'
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_UPLOAD_SECONDS = 30.0
ALLOWED_UPLOAD = {'.wav', '.flac', '.ogg'}

app = FastAPI(title='아이 음성 인식 뷰어', docs_url=None, redoc_url=None)
ENGINE = engine_module.Engine()
CATALOG = samples_module.Catalog()
HISTORY = []

#: This process's own identity. `pid` is the process actually serving, not
#: whatever a launcher happened to spawn, and `token` distinguishes it
#: from a different process that a rebooted machine gave the same pid to.
INSTANCE = {
    'pid': os.getpid(),
    'token': uuid.uuid4().hex,
    'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    'app': 'child-jamo-web',
}


def _reference(row):
    """The recorded targets for a sample, used only to score after the fact.

    None when the recording has no reference at all - the whole personal
    recordings have no transcript aligned to them. Everything downstream
    treats that as "not comparable" rather than as a perfect or zero score.
    """
    if not row.get('label_ids'):
        return None
    return {
        'transcript': row['text'],
        'surface': row['surface'],
        'jamo': ''.join(labels.TOKENS[i] for i in row['label_ids']
                        if i != labels.BOUNDARY),
        'label_ids': row['label_ids'],
    }


def _reference_note(row):
    """Why a recording can or cannot be compared, in one short line."""
    state = row.get('reference_state', 'transcript')
    return {
        'state': state,
        'text': {
            'transcript': '',
            'unsupported': '전사 라벨을 목표 발음으로 바꿀 수 없어 비교하지 '
                           '않습니다.',
            'none': '이 녹음 전체에 맞춰진 전사가 없어 비교하지 않습니다. '
                    '인식 결과만 그대로 보여 줍니다.',
        }.get(state, ''),
        'reason': row.get('reference_reason'),
    }


#: One page per source. They are the same document - same model, same
#: server, one resident engine - and the page reads its own path to decide
#: which source it is showing. Real URLs rather than a tab widget, so a
#: link, a refresh and the back button all mean what they usually mean.
@app.get('/')
@app.get('/aihub')
@app.get('/recordings')
def index():
    return FileResponse(STATIC / 'index.html',
                        headers={'Cache-Control': 'no-store'})


#: The listening review is its own page: a different job (answering
#: questions about what was said) from browsing results.
@app.get('/review')
def review_page():
    return FileResponse(STATIC / 'review.html',
                        headers={'Cache-Control': 'no-store'})


@app.get('/review.js')
def review_script():
    return FileResponse(STATIC / 'review.js', media_type='text/javascript')


@app.get('/app.js')
def script():
    return FileResponse(STATIC / 'app.js', media_type='text/javascript')


@app.get('/style.css')
def style():
    return FileResponse(STATIC / 'style.css', media_type='text/css')


#: Browsers ask for this unprompted; answering keeps a 404 out of the console.
FAVICON = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
    "<rect width='64' height='64' rx='14' fill='#2f6df6'/>"
    "<text x='32' y='45' font-size='36' text-anchor='middle' fill='#fff'"
    " font-family='sans-serif'>가</text></svg>"
)


@app.get('/favicon.ico')
def favicon():
    return Response(FAVICON, media_type='image/svg+xml',
                    headers={'Cache-Control': 'max-age=86400'})


@app.get('/api/status')
def status():
    return {
        'model': ENGINE.status(),
        'sources': samples_module.SOURCES,
        'splits': CATALOG.available(),
        'vocabulary': len(labels.TOKENS),
        'uploads_enabled': True,
        'max_upload_seconds': MAX_UPLOAD_SECONDS,
        # Identity of this exact serving process, so a launcher can tell
        # "the server I started" from "something else on this port".
        'instance': INSTANCE,
    }


@app.post('/api/warm')
def warm():
    try:
        ENGINE.load()
    except Exception as exc:
        raise HTTPException(503, f'모델을 불러오지 못했습니다: {exc}')
    return ENGINE.status()


@app.get('/api/splits/{split}')
def split_state(split: str):
    if split not in samples_module.SPLITS:
        raise HTTPException(404, '알 수 없는 데이터셋입니다')
    CATALOG.ensure(split)
    return CATALOG.available()[split]


@app.get('/api/samples')
def samples(split: str = 'test', q: str = '', age: str = '', bucket: str = '',
            style: str = '', child: str = '', page: int = 1, size: int = 24):
    if split not in samples_module.SPLITS:
        raise HTTPException(404, '알 수 없는 데이터셋입니다')
    if not CATALOG.ensure(split):
        return JSONResponse({'loading': True, 'items': [], 'total': 0,
                             'page': 1, 'pages': 1}, status_code=202)
    found = CATALOG.search(split, q, age, bucket, style, child, page, size)
    return {'loading': False, **found}


@app.get('/api/sample/{sample_id:path}')
def sample(sample_id: str):
    row = CATALOG.get(sample_id)
    if row is None:
        raise HTTPException(404, '해당 음성을 찾을 수 없습니다')
    out = {
        **samples_module.summary(row),
        'reference': _reference(row),
        'reference_note': _reference_note(row),
        'audio_url': f'/api/audio/{sample_id}',
        'previous': CATALOG.neighbour(sample_id, -1),
        'next': CATALOG.neighbour(sample_id, 1),
    }
    if row.get('segment_plan'):
        # The chunk boundaries are known before anything is recognised, so
        # the page can show where the recording will be cut - and how many
        # of those cuts did not land in silence - up front.
        plan = row['segment_plan']
        out['chunks'] = {
            'segments': [{k: s[k] for k in
                          ('index', 'start_s', 'end_s', 'seconds',
                           'boundary_dbfs', 'boundary_cuts_speech')}
                         for s in plan],
            'count': len(plan),
            'covers_samples': sum(s['samples'] for s in plan),
            'total_samples': row['samples'],
            'max_seconds': personal.CHUNK_MAX_SECONDS,
            'cuts_speech': sum(s['boundary_cuts_speech'] for s in plan),
        }
    return out


@app.get('/api/random')
def random_sample(split: str = 'test', q: str = '', age: str = '',
                  bucket: str = '', style: str = '', child: str = ''):
    if split not in samples_module.SPLITS:
        raise HTTPException(404, '알 수 없는 데이터셋입니다')
    if not CATALOG.ensure(split):
        raise HTTPException(202, '데이터셋을 불러오는 중입니다')
    picked = CATALOG.pick(split, seed=time.time_ns(), query=q, age=age,
                          bucket=bucket, style=style, child=child)
    if picked is None:
        raise HTTPException(404, '조건에 맞는 음성이 없습니다')
    return picked


@app.get('/api/batch-scope')
def batch_scope(split: str = 'test', sample_id: str = '', count: int = 10,
                q: str = '', age: str = '', bucket: str = '', style: str = '',
                child: str = ''):
    """Exactly which recordings a batch would run, before running them."""
    if split not in samples_module.SPLITS:
        raise HTTPException(404, '알 수 없는 데이터셋입니다')
    if not CATALOG.ensure(split):
        raise HTTPException(202, '데이터셋을 불러오는 중입니다')
    if not samples_module.SPLITS[split].get('batchable', True):
        raise HTTPException(400, '원본 전체 녹음은 묶음 인식 대상이 아닙니다. '
                                 '하나씩 구간별로 인식합니다')
    items = CATALOG.following(split, sample_id, max(1, min(int(count), 10)),
                              query=q, age=age, bucket=bucket, style=style,
                              child=child)
    return {'items': items, 'count': len(items)}


@app.get('/api/audio/{sample_id:path}')
def audio(sample_id: str):
    row = CATALOG.get(sample_id)
    if row is None:
        raise HTTPException(404, '해당 음성을 찾을 수 없습니다')
    try:
        waveform = engine_module.read_row(row)
    except Exception as exc:
        raise HTTPException(500, f'음성을 읽지 못했습니다: {exc}')
    if row.get('kind') == 'personal_original':
        # Two minutes of audio that the page seeks around in, so it is
        # served as a file: the browser gets byte ranges instead of
        # re-downloading the whole recording for every segment click. The
        # bytes are the canonical WAV the chunks were cut from - the file
        # `personal.read` just verified against the catalogue.
        return FileResponse(row['path'], media_type='audio/wav',
                            headers={'Cache-Control': 'no-store'})
    return Response(engine_module.wav_bytes(waveform), media_type='audio/wav',
                    headers={'Cache-Control': 'no-store'})


class TranscribeRequest(BaseModel):
    sample_id: str
    request_id: str | None = None
    use_cache: bool = True


@app.post('/api/transcribe')
def transcribe(body: TranscribeRequest):
    row = CATALOG.get(body.sample_id)
    if row is None:
        raise HTTPException(404, '해당 음성을 찾을 수 없습니다')
    if row.get('kind') == 'personal_original':
        raise HTTPException(
            400, '원본 전체 녹음은 구간별로 인식합니다 (/api/transcribe-original)')
    try:
        waveform = engine_module.read_row(row)
        prediction = ENGINE.transcribe(
            waveform, pcm_sha256=row.get('pcm_sha256'),
            use_cache=body.use_cache)
    except Exception as exc:
        raise HTTPException(500, f'인식에 실패했습니다: {exc}')

    reference = _reference(row)
    # Scoring happens strictly after the prediction exists, and only when
    # there is something recorded to score against.
    metrics = alignment = None
    if reference:
        metrics = compose.compare(reference['label_ids'],
                                  prediction['token_ids'])
        alignment = compose.alignment(reference['label_ids'],
                                      prediction['token_ids'])
    record = {
        'request_id': body.request_id,
        'sample_id': body.sample_id,
        'sample': samples_module.summary(row),
        'reference': reference,
        'reference_note': _reference_note(row),
        'prediction': prediction,
        'metrics': metrics,
        'alignment': alignment,
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    HISTORY.append({k: record[k] for k in
                    ('sample_id', 'metrics', 'at')} |
                   {'text': prediction['text'],
                    'split': row['split'],
                    'transcript': reference['transcript'] if reference
                    else None})
    del HISTORY[:-200]
    return record


class OriginalRequest(BaseModel):
    sample_id: str
    request_id: str | None = None


@app.post('/api/transcribe-original')
def transcribe_original(body: OriginalRequest):
    """One whole recording, in bounded chunks, every sample included.

    The chunks are cut from the same array `/api/audio` serves, so the
    timestamps shown beside each piece of text point at exactly that part
    of the recording the browser is playing. The chunks are recognised
    independently and reported as they came out: nothing is stitched,
    merged or de-duplicated across a boundary, because that would be this
    page inventing words the model did not produce.
    """
    row = CATALOG.get(body.sample_id)
    if row is None:
        raise HTTPException(404, '해당 음성을 찾을 수 없습니다')
    if row.get('kind') != 'personal_original':
        raise HTTPException(400, '원본 전체 녹음에만 쓰는 경로입니다')
    try:
        waveform = engine_module.read_row(row)
        plan = row['segment_plan']
        if sum(s['samples'] for s in plan) != len(waveform):
            raise ValueError('분할이 음성 전체를 덮지 않습니다')
        segments = []
        for piece in plan:
            chunk = waveform[piece['start']:piece['end']]
            prediction = ENGINE.transcribe(
                chunk, pcm_sha256=engine_module.pcm_sha256(chunk))
            segments.append({k: piece[k] for k in
                             ('index', 'start_s', 'end_s', 'seconds',
                              'samples', 'boundary_dbfs',
                              'boundary_cuts_speech')}
                            | {'prediction': prediction})
    except Exception as exc:
        raise HTTPException(500, f'인식에 실패했습니다: {exc}')

    covered = sum(s['samples'] for s in segments)
    HISTORY.append({'sample_id': body.sample_id, 'metrics': None,
                    'split': row['split'],
                    'at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'text': ' '.join(s['prediction']['text']
                                     for s in segments)[:120],
                    'transcript': None})
    del HISTORY[:-200]
    return {
        'request_id': body.request_id,
        'sample_id': body.sample_id,
        'sample': samples_module.summary(row),
        'reference': None,
        'reference_note': _reference_note(row),
        'segments': segments,
        'audio_url': f'/api/audio/{body.sample_id}',
        'coverage': {
            'total_samples': len(waveform),
            'covered_samples': covered,
            'complete': covered == len(waveform),
            'segments': len(segments),
            'max_seconds': max(s['seconds'] for s in segments),
            'inference_seconds': sum(s['prediction']['inference_seconds']
                                     for s in segments),
            'note': '모든 샘플이 한 번씩 포함됩니다. 구간은 잘라 보여 주기 '
                    '위한 것이고, 원본은 통째로 재생됩니다.',
        },
        'at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


class BatchRequest(BaseModel):
    sample_ids: list[str]
    request_id: str | None = None


@app.post('/api/batch')
def batch(body: BatchRequest):
    ids = body.sample_ids[:10]
    if not ids:
        raise HTTPException(400, '음성을 선택해 주세요')
    results, failures = [], []
    for sample_id in ids:
        row = CATALOG.get(sample_id)
        if row is None:
            failures.append({'sample_id': sample_id,
                             'reason': '음성을 찾을 수 없습니다'})
            continue
        if row.get('kind') == 'personal_original':
            failures.append({'sample_id': sample_id,
                             'reason': '원본 전체 녹음은 구간별로 인식합니다'})
            continue
        try:
            waveform = engine_module.read_row(row)
            prediction = ENGINE.transcribe(waveform,
                                           pcm_sha256=row.get('pcm_sha256'))
            reference = _reference(row)
            results.append({
                'sample_id': sample_id,
                'sample': samples_module.summary(row),
                'reference': reference,
                'reference_note': _reference_note(row),
                'prediction': prediction,
                'metrics': compose.compare(reference['label_ids'],
                                           prediction['token_ids'])
                if reference else None,
            })
        except Exception as exc:
            failures.append({'sample_id': sample_id,
                             'reason': f'{type(exc).__name__}: {exc}'})
    # Only the recordings that actually have a reference enter the rate.
    # A recognised recording with nothing to compare against is counted
    # and named, not folded in as if it had scored perfectly.
    scored = [r for r in results if r['metrics']]
    errs = sum(r['metrics']['errors'] for r in scored)
    ref = sum(r['metrics']['reference_jamo'] for r in scored)
    return {
        'request_id': body.request_id,
        'results': results,
        'requested': len(ids),
        'failures': failures,
        'all_failed': bool(ids) and not results,
        'aggregate': {
            'recordings': len(results),
            'scored': len(scored),
            'unscored': len(results) - len(scored),
            'errors': errs, 'reference_jamo': ref,
            # None, not 0, when nothing was scored - a 0% error rate that
            # no recording produced would be an invented result.
            'jer': errs / ref if ref else None,
            'exact': sum(1 for r in scored if r['metrics']['exact']),
            'scope': '이번 선택분에 대한 집계입니다 (전체 테스트 점수와 별개).',
        },
    }


@app.get('/api/history')
def history(source: str = ''):
    """The recent runs, optionally only one source's.

    The filter is applied before the limit, not after. Slicing first and
    filtering in the browser would mean a page showing the last few of
    whatever the *other* page happened to run - an AIHub run could push
    the personal recordings' own history off the end of a list that never
    had room for it.

    Membership comes from the split each run recorded, which is a name
    this server defines. Nothing is guessed from a filename.
    """
    if source:
        if source not in samples_module.SOURCES:
            raise HTTPException(404, '알 수 없는 출처입니다')
        allowed = set(samples_module.splits_of(source))
        items = [row for row in HISTORY if row.get('split') in allowed]
    else:
        items = HISTORY
    return {'items': items[-50:][::-1], 'source': source or None}


# --- listening review -------------------------------------------------
#
# The review answers a question the model cannot: what was actually said.
# So nothing here feeds back into inference - saving an annotation changes
# no prediction, no score and no page the model results are shown on.

REVIEW = {'index': None, 'store': None}
REVIEW_LOCK = threading.Lock()


def review_db_path():
    """Where answers are kept. Read when first needed, not at import.

    `CHILD_REVIEW_DB` points this at a throwaway file for tests. Resolving
    it lazily matters: a test module that sets the variable after this one
    was already imported would otherwise be handed the reviewer's real
    answers to write into.
    """
    return os.environ.get('CHILD_REVIEW_DB') or review_module.DEFAULT_DB


def _review():
    with REVIEW_LOCK:
        if REVIEW['index'] is None or not REVIEW['index'].get('ready'):
            # Retried while not ready: the artefacts may have been built
            # after this process started, and a cached "not ready" would
            # otherwise need a restart to clear.
            REVIEW['index'] = review_module.load_index()
        if REVIEW['store'] is None:
            REVIEW['store'] = review_module.Store(review_db_path())
        return REVIEW['index'], REVIEW['store']


def _review_record(record_id):
    index, _ = _review()
    record = index['records'].get(record_id)
    if record is None:
        # Ids come from the audit, never from a path or a free-form string.
        raise HTTPException(404, '검토 목록에 없는 음성입니다')
    return record


def _verified_record(record_id):
    """The record, checked against the recording the catalogue holds now.

    The baseline index alone does not prove the audio about to be played is
    the audio the prediction was made from, so the current row is read and
    hashed before anything is served or saved. A recording that has moved
    on is a 409, not a silently different clip under an old question.
    """
    record = _review_record(record_id)
    row = CATALOG.get(record_id)
    if row is None:
        raise HTTPException(404, '해당 음성을 찾을 수 없습니다')
    try:
        waveform = engine_module.read_row(row)
    except Exception as exc:
        raise HTTPException(500, f'음성을 읽지 못했습니다: {exc}')
    problems = audit_module.validate_record(
        record, row, engine_module.pcm_sha256(waveform))
    if problems:
        raise HTTPException(
            409, '이 음성이나 목표가 기준 기록과 달라졌습니다 '
                 f'({", ".join(problems)}). 검토를 다시 만들어 주세요')
    return record, row, waveform


def _review_label(record, index=None):
    """A human-facing name. Never a raw id or a speaker hash."""
    source = '직접 녹음' if record['pool'] == 'personal' else '말뭉치'
    child = {'child1': '아동 1', 'child2': '아동 2'}.get(record.get('child'))
    who = child or ('화자 ' + str(index) if index else '')
    return f'{source} {who}'.strip()


def _review_item(record_id, record, annotation, row=None):
    row = row if row is not None else CATALOG.get(record_id)
    context = review_context(record_id)
    stale = review_module.is_stale(annotation, record)
    return {
        'id': record_id,
        # Handed out with the item and required back on save, so an answer
        # can only be applied to the thing it was given for.
        'binding': record.get('binding'),
        'revision': annotation['revision'] if annotation else 0,
        'annotation_stale': stale,
        'pool': record['pool'],
        'source_label': '직접 녹음' if record['pool'] == 'personal'
                        else '말뭉치 (AIHub)',
        'speaker': record['speaker'],
        'child': record.get('child'),
        'child_label': {'child1': '아동 1', 'child2': '아동 2'}.get(
            record.get('child')),
        'age': row['age'] if row else None,
        'seconds': record['seconds'],
        'transcript': record['transcript'],
        'expected_surface': record['reference_surface'],
        'predicted_text': record['predicted_text'],
        'prediction_differs': record['predicted_text'] != record['reference_surface'],
        'consonant_events': record['consonant_events'],
        'audio_url': f'/api/review/audio/{record_id}',
        'context_url': f'/api/review/context/{record_id}' if context else None,
        'context_seconds': context['seconds'] if context else None,
        'annotation': annotation,
    }


def review_context(record_id):
    """A window of the original recording around a personal clip.

    For listening only, and labelled as such on the page: it is the source
    recording either side of the clip, not a new cut of it. Nothing is
    written, and the original file is opened read-only.
    """
    row = CATALOG.get(record_id)
    if row is None or row.get('kind') != 'personal_clip':
        return None
    return personal.clip_context(row, seconds=1.0)


@app.get('/api/review/state')
def review_state(source: str = ''):
    index, store = _review()
    if not index['ready']:
        raise HTTPException(
            503, '검토 목록이 아직 준비되지 않았습니다 (audit.py 를 먼저 실행)')
    current = store.all_current()
    shortlist = index['shortlist']
    if source:
        if source not in ('personal', 'valid'):
            raise HTTPException(404, '알 수 없는 출처입니다')
        shortlist = [s for s in shortlist if s['pool'] == source]

    def live(record_id):
        """The current answer, or None if it no longer describes the audio.

        A stale annotation is not a finished item: it was about a recording
        or a prediction that has since changed, so it is counted as still
        needing an answer rather than as done.
        """
        annotation = current.get(record_id)
        if annotation is None:
            return None
        if review_module.is_stale(annotation,
                                  index['records'].get(record_id)):
            return None
        return annotation

    def tally(ids):
        statuses = [live(i) for i in ids]
        answered = sum(1 for a in statuses if a and a['status'] == 'answered')
        unclear = sum(1 for a in statuses if a and a['status'] == 'unclear')
        unusable = sum(1 for a in statuses if a and a['status'] == 'unusable')
        stale = sum(1 for i in ids
                    if current.get(i) and live(i) is None)
        return {'total': len(ids), 'answered': answered, 'unclear': unclear,
                'unusable': unusable, 'stale': stale,
                'remaining': len(ids) - answered - unclear - unusable}

    flagged = {pool: [r['id'] for r in index['records'].values()
                      if r['pool'] == pool and r['consonant_mistakes']]
               for pool in ('personal', 'valid')}
    return {
        'ready': True,
        'priority': [{**item,
                      'transcript': index['records'][item['id']]['transcript'],
                      'label': _review_label(index['records'][item['id']],
                                             position),
                      'status': (live(item['id']) or {}).get('status')}
                     for position, item in enumerate(shortlist, 1)
                     if item['id'] in index['records']],
        'progress': {
            'priority': tally([s['id'] for s in shortlist]),
            'by_source': {pool: tally([s['id'] for s in index['shortlist']
                                       if s['pool'] == pool])
                          for pool in ('personal', 'valid')},
        },
        'flagged_counts': {pool: len(ids) for pool, ids in flagged.items()},
        'reviewed_total': len(current),
        'choices': review_module.CHOICES,
        'flags': review_module.FLAGS,
        'note': '우선 확인 목록만 답해 주셔도 됩니다. 나머지는 참고용으로 '
                '열어 둔 것이고 전부 확인하실 필요는 없습니다.',
    }


@app.get('/api/review/flagged')
def review_flagged(source: str, page: int = 1, size: int = 30):
    """The rest of the disagreements, browsable but never required.

    `source` is required here on purpose. Defaulting it silently turned an
    "all sources" choice on screen into one source's list.
    """
    index, store = _review()
    if source not in ('personal', 'valid'):
        raise HTTPException(404, '알 수 없는 출처입니다')
    current = store.all_current()
    rows = sorted((r for r in index['records'].values()
                   if r['pool'] == source and r['consonant_mistakes']),
                  key=lambda r: (-r['consonant_mistakes'], r['id']))
    size = max(1, min(int(size), 100))
    page = max(1, int(page))
    start = (page - 1) * size
    return {
        'total': len(rows), 'page': page, 'size': size,
        'pages': max(1, (len(rows) + size - 1) // size),
        'items': [{'id': r['id'], 'pool': r['pool'],
                   'label': _review_label(r, start + offset + 1),
                   'transcript': r['transcript'],
                   'expected_surface': r['reference_surface'],
                   'predicted_text': r['predicted_text'],
                   'consonant_mistakes': r['consonant_mistakes'],
                   'status': None if review_module.is_stale(
                       current.get(r['id']), r)
                   else (current.get(r['id']) or {}).get('status')}
                  for offset, r in enumerate(rows[start:start + size])],
    }


@app.get('/api/review/item/{record_id:path}')
def review_item(record_id: str):
    record, row, _waveform = _verified_record(record_id)
    _, store = _review()
    return _review_item(record_id, record, store.current(record_id), row)


@app.get('/api/review/audio/{record_id:path}')
def review_audio(record_id: str):
    """Exactly the signal the model was given for this recording."""
    _record, _row, waveform = _verified_record(record_id)
    return Response(engine_module.wav_bytes(waveform), media_type='audio/wav',
                    headers={'Cache-Control': 'no-store'})


@app.get('/api/review/context/{record_id:path}')
def review_audio_context(record_id: str):
    _review_record(record_id)
    context = review_context(record_id)
    if context is None:
        raise HTTPException(404, '이 음성에는 원본 앞뒤가 없습니다')
    return Response(engine_module.wav_bytes(context['waveform']),
                    media_type='audio/wav',
                    headers={'Cache-Control': 'no-store'})


@app.get('/api/review/history/{record_id:path}')
def review_history(record_id: str):
    _review_record(record_id)
    _, store = _review()
    return {'items': store.history(record_id)}


class ReviewSave(BaseModel):
    record_id: str
    status: str
    #: The identity of the item the page was showing, and the revision it
    #: had. Both are required: without them a forgotten tab can answer a
    #: question that has since changed, or overwrite a newer answer.
    binding: str
    expected_revision: int
    choice: str | None = None
    heard_text: str | None = None
    note: str | None = None
    flags: list[str] = []


@app.post('/api/review/annotation')
def review_save(body: ReviewSave):
    record, _row, _waveform = _verified_record(body.record_id)
    _, store = _review()
    if body.binding != record.get('binding'):
        raise HTTPException(
            409, '화면에 있던 음성과 지금 저장하려는 음성이 다릅니다. '
                 '새로 고친 뒤 다시 들어 주세요')
    if body.status not in review_module.STATUSES:
        raise HTTPException(400, '알 수 없는 상태입니다')
    if body.status == 'answered':
        if body.choice not in review_module.CHOICES:
            raise HTTPException(400, '무엇으로 들렸는지 골라 주세요')
        if body.choice == 'unclear':
            raise HTTPException(
                400, "'판단하기 어려움' 은 별도 상태로 저장합니다")
        if body.choice == 'other' and not (body.heard_text or '').strip():
            raise HTTPException(400, '들린 대로 적어 주세요')
    for flag in body.flags:
        if flag not in review_module.FLAGS:
            raise HTTPException(400, f'알 수 없는 표시입니다: {flag}')
    if body.status == 'unusable' and not body.flags:
        raise HTTPException(400, '쓸 수 없는 이유를 하나 이상 골라 주세요')

    try:
        saved = store.save(
            expected_revision=body.expected_revision,
            record_id=body.record_id, pool=record['pool'], status=body.status,
            choice=body.choice if body.status == 'answered' else None,
            # Stored exactly as submitted. Trimming is only used above to
            # decide whether anything was typed at all; what gets written is
            # the reviewer's own characters, spaces included. No G2P, no
            # correction towards the transcript or the model's guess.
            heard_text=body.heard_text if (body.heard_text or '').strip() else None,
            note=body.note if (body.note or '').strip() else None,
            flags=body.flags,
            audio_pcm_sha256=record['audio_pcm_sha256'],
            baseline_key=record['baseline_key'],
            binding=record['binding'],
            reference_ids=record.get('reference_ids'),
            reference_surface=record['reference_surface'],
            predicted_text=record['predicted_text'])
    except review_module.Conflict as exc:
        raise HTTPException(409, str(exc))
    return {'saved': True, 'annotation': saved}


@app.get('/api/review/export')
def review_export():
    index, store = _review()
    return review_module.export(index, store)


@app.post('/api/upload')
def upload(file: UploadFile = File(...), reference: str = Form('')):
    # A plain `def` endpoint: FastAPI runs it on a worker thread, so a
    # multi-second inference does not stall the event loop and every
    # other request with it.
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in ALLOWED_UPLOAD:
        raise HTTPException(
            400, f'{" / ".join(sorted(ALLOWED_UPLOAD))} 형식만 올릴 수 있습니다')
    blob = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(blob) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, '파일이 너무 큽니다 (최대 20MB)')
    try:
        # One canonical waveform: the bytes served back are the bytes the
        # model saw, and the duration limit is enforced from the header
        # before anything is decoded or resampled.
        data, encoded, original_rate = engine_module.canonical_upload(blob)
    except engine_module.DurationTooLong as exc:
        raise HTTPException(413, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    UPLOADS.mkdir(parents=True, exist_ok=True)
    checksum = engine_module.pcm_sha256(data)
    (UPLOADS / f'{checksum[:16]}.wav').write_bytes(encoded)
    seconds = len(data) / engine_module.SAMPLE_RATE

    try:
        prediction = ENGINE.transcribe(data, pcm_sha256=checksum)
    except Exception as exc:
        raise HTTPException(500, f'인식에 실패했습니다: {exc}')
    out = {'kind': 'upload', 'filename': Path(file.filename or 'audio').name,
           'seconds': seconds, 'audio_url': f'/api/upload-audio/{checksum[:16]}',
           'prediction': prediction, 'original_sample_rate': original_rate,
           'served_audio_is_model_input': True}
    reference = (reference or '').strip()
    if reference:
        text, surface, ids = labels.targets(reference)
        if ids is None:
            out['reference_error'] = (
                '입력한 문장에서 목표 발음을 만들 수 없습니다 '
                f'({text}). 인식 결과는 그대로 표시합니다.')
        else:
            out['reference'] = {'transcript': text, 'surface': surface,
                                'jamo': ''.join(labels.TOKENS[i] for i in ids
                                                if i != labels.BOUNDARY),
                                'label_ids': ids}
            out['metrics'] = compose.compare(ids, prediction['token_ids'])
            out['alignment'] = compose.alignment(ids, prediction['token_ids'])
    return out


@app.get('/api/upload-audio/{token}')
def upload_audio(token: str):
    if not token.isalnum() or len(token) != 16:
        raise HTTPException(404, '잘못된 요청입니다')
    stored = UPLOADS / f'{token}.wav'
    if not stored.exists():
        raise HTTPException(404, '파일을 찾을 수 없습니다')
    return FileResponse(stored, media_type='audio/wav')


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--warm', action='store_true')
    #: Which file this process claims ownership in. A throwaway instance on
    #: a spare port - a QA run, say - must not overwrite the record the
    #: launcher uses to identify the real server, or `-Stop` stops being
    #: able to recognise its own process at all.
    parser.add_argument('--pid-file', default=None)
    args = parser.parse_args()
    if args.host != '127.0.0.1':
        raise SystemExit('이 뷰어는 127.0.0.1 에서만 동작합니다')
    WORK.mkdir(parents=True, exist_ok=True)
    pid_file = (Path(args.pid_file) if args.pid_file
                else WORK / ('server.pid.json' if args.port == 8767
                             else f'server-{args.port}.pid.json'))
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    # The serving process records itself. A launcher that only knows the
    # pid it spawned can be holding a shim, and stopping that would leave
    # this process - and its GPU memory - behind.
    pid_file.write_text(json.dumps({
        **INSTANCE, 'port': args.port,
        'url': f'http://{args.host}:{args.port}/',
        'python': sys.executable, 'script': str(Path(__file__).resolve()),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.warm:
        ENGINE.load()
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
