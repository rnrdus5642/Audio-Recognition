"""Local microphone/file evaluation UI for the exported spelling-jamo model."""
import argparse
from contextlib import asynccontextmanager
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import threading
import unicodedata
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from rapidfuzz.distance import Levenshtein

RELEASE_CODE = Path(__file__).resolve().parents[1] / 'child_orthographic_release'
sys.path.insert(0, str(RELEASE_CODE))
from inference import Engine, MAX_BYTES, MAX_SECONDS, prepare_wav
from word_game import WordGames, StartWordGame, FinishWordGame

ROOT = Path(__file__).resolve().parents[4]
STATIC = Path(__file__).resolve().parent / 'static'
DEFAULT_BUNDLE = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/releases/epoch3-final-web-20261031'
DEFAULT_RECORDINGS = ROOT / '03_Recordings/원문자모_성능테스트'


def score(reference, predicted):
    ref = ''.join(unicodedata.normalize('NFC', reference).split())
    hyp = ''.join(unicodedata.normalize('NFC', predicted).split())
    if not ref:
        return None
    errors = Levenshtein.distance(ref, hyp)
    return {'reference': ref, 'hypothesis': hyp, 'errors': errors, 'reference_characters': len(ref),
            'cer': errors / len(ref), 'exact': ref == hyp,
            'metric': 'NFC_characters_without_whitespace'}


def create_app(bundle=DEFAULT_BUNDLE, recordings=DEFAULT_RECORDINGS, device='cpu', threads=2):
    engine = Engine(bundle, device, threads)
    recordings = Path(recordings)
    recordings.mkdir(parents=True, exist_ok=True)

    def load():
        try:
            engine.load()
        except Exception:
            logging.exception('Model load failed')

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=load, daemon=True).start()
        yield

    app = FastAPI(title='원문 자모 모델 성능 테스트', lifespan=lifespan, docs_url=None, redoc_url=None)
    app.state.engine = engine
    app.mount('/assets', StaticFiles(directory=STATIC), name='assets')

    @app.middleware('http')
    async def local_only(request, call_next):
        host = request.headers.get('host', '')
        if host.split(':')[0] not in ('127.0.0.1', 'localhost', 'testserver'):
            return JSONResponse({'detail': '이 PC의 로컬 주소로 접속해 주세요.'}, status_code=403)
        origin = request.headers.get('origin')
        if request.method not in ('GET', 'HEAD') and origin and origin != f'http://{host}':
            return JSONResponse({'detail': '같은 테스트 페이지에서 요청해 주세요.'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/health')
    @app.get('/api/status')
    def status():
        return engine.status()

    def audio_path(record_id):
        if not re.fullmatch('[0-9a-f]{32}', record_id):
            raise HTTPException(404, '녹음을 찾을 수 없습니다.')
        return recordings / f'{record_id}.wav'

    @app.get('/api/audio/{record_id}')
    def audio(record_id: str):
        path = audio_path(record_id)
        if not path.is_file():
            raise HTTPException(404, '녹음을 찾을 수 없습니다.')
        return FileResponse(path, media_type='audio/wav')

    @app.get('/api/history')
    def history():
        items = []
        for path in sorted(recordings.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                item = json.loads(path.read_text(encoding='utf-8'))
                if item['model_id'] == engine.status()['model_id'] and audio_path(item['id']).is_file():
                    items.append(item)
            except (ValueError, KeyError, OSError):
                logging.warning('Skipping incomplete test record: %s', path.name)
        scored = [item['score'] for item in items if item.get('score') is not None]
        references = sum(item['reference_characters'] for item in scored)
        errors = sum(item['errors'] for item in scored)
        return {'items': items[:100], 'total': len(items), 'scored_count': len(scored),
                'exact_count': sum(item['exact'] for item in scored),
                'corpus_cer': errors / references if references else None,
                'note': '현재 모델의 직접 테스트 기록이며 전체 개발 검증 성적이 아닙니다.'}

    def infer(waveform):
        if not engine.ready:
            raise HTTPException(503, engine.error or '모델을 준비하고 있습니다.')
        try:
            # Reference, speaker age and session metadata never enter the model.
            return engine.transcribe(waveform)
        except BlockingIOError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            logging.exception('Inference failed')
            raise HTTPException(500, '인식 중 오류가 발생했습니다. 서버 기록을 확인해 주세요.') from exc

    def read_audio(file):
        try:
            return prepare_wav(file.file.read(MAX_BYTES + 1))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    def save_record(result, canonical):
        record_id = result['id']
        temporary_audio = recordings / f'{record_id}.{uuid.uuid4().hex}.wav.tmp'
        temporary_record = recordings / f'{record_id}.{uuid.uuid4().hex}.json.tmp'
        try:
            temporary_audio.write_bytes(canonical)
            temporary_audio.replace(audio_path(record_id))
            temporary_record.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            temporary_record.replace(recordings / f'{record_id}.json')
        finally:
            temporary_audio.unlink(missing_ok=True)
            temporary_record.unlink(missing_ok=True)
        return result

    def save_game(state, frame):
        model_id = engine.status()['model_id']
        record_id = uuid.uuid5(uuid.NAMESPACE_URL, f"orthographic-game:{model_id}:{state['id']}").hex
        result = {'id': record_id, 'created_at': datetime.now().astimezone().isoformat(),
                  'model_id': model_id, 'epoch_equivalents': engine.release['epoch_equivalents'],
                  'device': engine.device, 'reference': state['settings']['reference'],
                  'age': state['settings']['age'], 'source': 'game',
                  'source_sample_rate': frame['source_sample_rate'], 'prediction': frame['prediction'],
                  'score': score(state['settings']['reference'], frame['prediction']['text']),
                  'audio_url': f'/api/audio/{record_id}', 'session_id': state['id'],
                  'game': {'outcome': state['outcome'], 'settings': state['settings'],
                           # Multiple candidates do not provide a known reference
                           # transcript. Never count a model-selected target as CER ground truth.
                           'matched_target': state['streak_target'] if state['outcome'] == 'accepted' else None,
                           'candidate': frame['candidate'], 'candidate_scores': frame['candidate_scores'],
                           'matched_text': frame['matched_text'],
                           'comparison_text': frame['comparison_text'], 'normalization': frame['normalization'],
                           'blocked_by': frame['blocked_by'], 'decision_reason': frame['decision_reason'],
                           'selection_reason': frame.get('selection_reason'),
                           'ambiguous': frame['ambiguous'], 'tied_targets': frame['tied_targets'],
                           'similarity': frame['similarity'], 'streak': state['streak'],
                           'last_checked_seconds': state['elapsed'],
                           'saved_window_ends_at': frame['elapsed_seconds'],
                           'audio_role': 'confirming_window' if state['outcome'] == 'accepted' else 'best_matching_window',
                           'trace_dropped_frames': state['trace_dropped_frames'],
                           'trace': state['trace']}}
        return save_record(result, frame['canonical'])

    games = WordGames(engine, infer, save_game)
    app.state.games = games

    @app.post('/api/game/start')
    def game_start(params: StartWordGame):
        return games.start(params)

    @app.post('/api/game/{session_id}/frame')
    def game_frame(session_id: str, file: UploadFile = File(...),
                   frame_index: int = Form(...), elapsed_seconds: float = Form(...)):
        waveform, canonical, rate = read_audio(file)
        return games.frame(session_id, frame_index, elapsed_seconds, waveform, canonical, rate)

    @app.post('/api/game/{session_id}/finish')
    def game_finish(session_id: str, params: FinishWordGame):
        return games.finish(session_id, params.reason)

    @app.post('/api/preview')
    def preview(file: UploadFile = File(...)):
        waveform, _canonical, _original_rate = read_audio(file)
        return {'prediction': infer(waveform), 'device': engine.device,
                'model_id': engine.status()['model_id'], 'provisional': True}

    @app.post('/api/transcribe')
    def transcribe(file: UploadFile = File(...), reference: str = Form(''),
                   age: str = Form('unknown'), source: str = Form('upload'),
                   session_id: str = Form(''), segment_index: int = Form(0)):
        if len(reference) > 300 or age not in ('unknown', '3', '4', '5', '6', 'adult') or source not in ('microphone', 'upload', 'live'):
            raise HTTPException(400, '입력한 정답이나 녹음 정보를 확인해 주세요.')
        if source == 'live' and (not re.fullmatch('[0-9a-f]{32}', session_id) or not 1 <= segment_index <= 1000):
            raise HTTPException(400, '실시간 녹음 세션 정보가 올바르지 않습니다.')
        waveform, canonical, original_rate = read_audio(file)
        request_hash = hashlib.sha256(canonical + json.dumps(
            [reference.strip(), age], ensure_ascii=False).encode('utf-8')).hexdigest()
        # Replaying a completed live segment after a connection failure must not
        # create duplicate audio/history entries. Partial previews are never saved.
        record_id = (uuid.uuid5(uuid.NAMESPACE_URL,
                     f"orthographic-live:{engine.status()['model_id']}:{session_id}:{segment_index}").hex
                     if source == 'live' else uuid.uuid4().hex)
        record_path = recordings / f'{record_id}.json'
        if source == 'live' and record_path.is_file():
            existing = json.loads(record_path.read_text(encoding='utf-8'))
            if existing.get('request_sha256') != request_hash:
                raise HTTPException(409, '이미 저장된 구간과 녹음 내용이 다릅니다.')
            if audio_path(record_id).is_file():
                return existing
        prediction = infer(waveform)
        result = {'id': record_id, 'created_at': datetime.now().astimezone().isoformat(),
                  'model_id': engine.status()['model_id'], 'epoch_equivalents': engine.release['epoch_equivalents'],
                  'device': engine.device, 'reference': reference.strip(), 'age': age, 'source': source,
                  'source_sample_rate': original_rate, 'prediction': prediction,
                  'score': score(reference, prediction['text']), 'audio_url': f'/api/audio/{record_id}'}
        if source == 'live':
            result.update(session_id=session_id, segment_index=segment_index, request_sha256=request_hash)
        return save_record(result, canonical)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument('--recordings', type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument('--port', type=int, default=8782)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.bundle, args.recordings, args.device, args.threads), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
