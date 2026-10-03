"""Local microphone transcription with the unchanged, current checkpoint.

Run: .venv-gpu/Scripts/python.exe -X utf8 python/tools/child_jamo_web/mic_server.py
"""
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import argparse
import io
import json
import logging
import os
import re
import threading
import uuid

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import engine
from common import ROOT

EXPECTED_CHECKPOINT = '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'
STATIC = Path(__file__).resolve().parent / 'static/microphone'
RECORDINGS = ROOT / '.venv/child-jamo-web-20260916/microphone/recordings'
MAX_SECONDS = 15
MAX_BYTES = 4 * 1024 * 1024
ENGINE = engine.Engine()
INSTANCE = {'app': 'child-jamo-microphone', 'pid': os.getpid(), 'token': uuid.uuid4().hex}


def load_model():
    try:
        ENGINE.load()
        if ENGINE.identity['checkpoint_sha256'] != EXPECTED_CHECKPOINT:
            ENGINE.ready = False
            ENGINE.error = '현재 테스트 대상으로 확인한 모델과 체크포인트가 다릅니다.'
    except Exception:
        logging.exception('Microphone model loading failed')


@asynccontextmanager
async def lifespan(_app):
    torch.set_num_threads(4)
    threading.Thread(target=load_model, daemon=True).start()
    yield


app = FastAPI(title='마이크 음성 인식', lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount('/assets', StaticFiles(directory=STATIC), name='assets')


@app.middleware('http')
async def local_requests(request, call_next):
    # This application has no remote API clients or cross-origin microphone UI.
    host = request.headers.get('host', '').split(':')[0]
    if host not in ('127.0.0.1', 'localhost', 'testserver'):
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': '로컬 주소로 접속해 주세요.'}, status_code=403)
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD') and origin and origin != f'http://{request.headers.get("host")}':
        from fastapi.responses import JSONResponse
        return JSONResponse({'detail': '같은 페이지에서 녹음해 주세요.'}, status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/status')
@app.get('/health')
def status():
    info = ENGINE.status()
    return {'instance': INSTANCE, 'model': {'ready': info['ready'], 'error': info['error'],
            'checkpoint_sha256': info.get('checkpoint_sha256'), 'device': info.get('device'),
            'label': '현재 모델'}, 'max_seconds': MAX_SECONDS}


def record_path(record_id, suffix):
    if not re.fullmatch(r'[0-9a-f]{32}', record_id):
        raise HTTPException(404, '녹음을 찾을 수 없습니다.')
    return RECORDINGS / (record_id + suffix)


@app.get('/api/history')
def history():
    paths = sorted(RECORDINGS.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:50]
    records = []
    for path in paths:
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
            if record_path(item['id'], '.wav').is_file():
                records.append(item)
        except (OSError, ValueError, KeyError):
            logging.warning('Skipping incomplete microphone record: %s', path.name)
    return {'items': records}


@app.get('/api/audio/{record_id}')
def audio(record_id: str):
    path = record_path(record_id, '.wav')
    if not path.is_file():
        raise HTTPException(404, '녹음을 찾을 수 없습니다.')
    return FileResponse(path, media_type='audio/wav')


@app.post('/api/transcribe')
def transcribe(file: UploadFile = File(...)):
    if not ENGINE.ready or ENGINE.identity.get('checkpoint_sha256') != EXPECTED_CHECKPOINT:
        raise HTTPException(503, ENGINE.error or '모델을 준비하고 있습니다. 잠시 후 다시 시도해 주세요.')
    blob = file.file.read(MAX_BYTES + 1)
    if len(blob) > MAX_BYTES:
        raise HTTPException(413, '녹음이 너무 큽니다. 15초 이하로 녹음해 주세요.')
    try:
        info = sf.info(io.BytesIO(blob))
        if info.format != 'WAV' or info.channels not in (1, 2):
            raise ValueError('모노 또는 스테레오 WAV 녹음이 필요합니다.')
        if info.duration > MAX_SECONDS + .2:
            raise engine.DurationTooLong(info.duration)
        wave, encoded, original_rate = engine.canonical_upload(blob)
    except engine.DurationTooLong:
        raise HTTPException(413, '15초 이하로 녹음해 주세요.')
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, f'녹음을 읽지 못했습니다: {exc}')
    try:
        # No expected word, dictionary, gain normalization, trimming, or new padding.
        prediction = ENGINE.transcribe(wave, use_cache=False)
    except Exception:
        logging.exception('Microphone inference failed')
        raise HTTPException(500, '인식 중 문제가 생겼습니다. 다시 녹음해 주세요.')
    record_id = uuid.uuid4().hex
    record = {'id': record_id, 'created_at': datetime.now().astimezone().isoformat(),
              'prediction': prediction, 'seconds': len(wave) / engine.SAMPLE_RATE,
              'audio_url': f'/api/audio/{record_id}', 'source_sample_rate': original_rate,
              'pcm_sha256': engine.pcm_sha256(wave), 'checkpoint_sha256': EXPECTED_CHECKPOINT,
              'served_audio_is_model_input': True}
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    record_path(record_id, '.wav').write_bytes(encoded)
    temporary = record_path(record_id, '.json.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(record_path(record_id, '.json'))
    return record


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8776)
    parser.add_argument('--recordings-dir', type=Path, help='Separate directory for browser QA recordings')
    args = parser.parse_args()
    if args.recordings_dir:
        RECORDINGS = args.recordings_dir.resolve()
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='info')
