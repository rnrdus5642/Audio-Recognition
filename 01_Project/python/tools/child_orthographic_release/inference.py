"""Portable, reference-independent inference for orthographic-hubert-ctc-v1."""
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import threading
import time
import unicodedata

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
MAX_SECONDS = 15
MAX_BYTES = 16 * 1024 * 1024


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decode(frame_ids, tokens):
    output, collapsed, previous = [], [], None
    for value in frame_ids:
        value = int(value)
        if value != previous and value != 0:
            collapsed.append(value)
            output.append(' ' if value == 1 else tokens[value])
        previous = value
    jamo = ''.join(output)
    return unicodedata.normalize('NFC', jamo), jamo, collapsed


def prepare_wav(blob):
    if not blob or len(blob) > MAX_BYTES:
        raise ValueError('16MB 이하의 WAV 파일을 선택해 주세요.')
    try:
        info = sf.info(io.BytesIO(blob))
        if info.format not in ('WAV', 'WAVEX') or info.channels not in (1, 2):
            raise ValueError('모노 또는 스테레오 WAV 파일이 필요합니다.')
        if not 8000 <= info.samplerate <= 192000:
            raise ValueError('8~192kHz WAV 파일이 필요합니다.')
        if not .1 <= info.duration <= MAX_SECONDS + .05:
            raise ValueError(f'0.1초 이상 {MAX_SECONDS}초 이하의 녹음을 사용해 주세요.')
        samples, rate = sf.read(io.BytesIO(blob), dtype='float32', always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as exc:
        raise ValueError('WAV 파일을 읽을 수 없습니다.') from exc
    wave = samples.mean(axis=1)
    if not np.isfinite(wave).all() or float(np.abs(wave).max(initial=0)) > 1.01:
        raise ValueError('오디오에 유효하지 않은 샘플 값이 있습니다.')
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        divisor = gcd(rate, SAMPLE_RATE)
        wave = resample_poly(wave, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
    wave = np.ascontiguousarray(wave, dtype=np.float32)
    # Quantize once; the saved/playable WAV then exactly represents model input.
    encoded = io.BytesIO()
    sf.write(encoded, wave, SAMPLE_RATE, format='WAV', subtype='PCM_16')
    canonical = encoded.getvalue()
    wave, _ = sf.read(io.BytesIO(canonical), dtype='float32')
    return np.ascontiguousarray(wave), canonical, rate


class Engine:
    def __init__(self, bundle, device='cpu', threads=2):
        self.bundle = Path(bundle).resolve()
        self.device = device
        self.threads = threads
        self.ready = False
        self.error = None
        self.busy = False
        self.lock = threading.Lock()
        self.release = read(self.bundle / 'release.json')
        self.model = self.head = None

    def load(self):
        started = time.perf_counter()
        try:
            manifest = self.release
            if manifest['format'] != 'orthographic-hubert-ctc-v1':
                raise ValueError('Unsupported inference bundle')
            for filename, expected in manifest['files'].items():
                path = (self.bundle / filename).resolve()
                if not path.is_relative_to(self.bundle) or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
                    raise ValueError(f'Bundle integrity check failed: {filename}')
            if manifest['normalize_waveform'] or manifest['blank_id'] != 0 or manifest['space_id'] != 1:
                raise ValueError('Unexpected decoder/preprocessing contract')
            vocab = read(self.bundle / 'vocab.json')
            if len(vocab) != 69 or sorted(vocab.values()) != list(range(69)):
                raise ValueError('Unexpected vocabulary')
            self.tokens = {value: token for token, value in vocab.items()}
            from transformers import HubertConfig, HubertModel
            torch.set_num_threads(self.threads)
            torch.backends.cuda.matmul.allow_tf32 = False
            config = HubertConfig.from_dict(read(self.bundle / 'config.json'))
            config._attn_implementation = 'sdpa'
            config.apply_spec_augment = False
            config.layerdrop = 0
            saved = torch.load(self.bundle / manifest['weights_file'], map_location='cpu', weights_only=True, mmap=True)
            with torch.device('meta'):
                model = HubertModel(config)
                head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(config.hidden_size, 69))
            model.load_state_dict(saved['model'], strict=True, assign=True)
            head.load_state_dict(saved['head'], strict=True, assign=True)
            self.model = model.to(self.device).eval().requires_grad_(False)
            self.head = head.to(self.device).eval().requires_grad_(False)
            self.load_seconds = time.perf_counter() - started
            self.ready = True
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            raise

    def status(self):
        return {'ready': self.ready, 'busy': self.busy, 'error': self.error,
                'label': self.release['label'], 'epoch_equivalents': self.release['epoch_equivalents'],
                'provisional': self.release.get('provisional', True),
                'device': self.device, 'precision': 'float32' if self.device == 'cpu' else 'bfloat16 encoder / float32 head',
                'model_id': self.release['files']['model.pt']['sha256'],
                'weights_id': self.release['source_receipt']['weights_id'],
                'max_seconds': MAX_SECONDS, 'sample_rate': SAMPLE_RATE}

    @torch.inference_mode()
    def transcribe(self, waveform):
        if not self.ready:
            raise RuntimeError('Model is not ready')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('다른 녹음을 분석 중입니다. 완료 후 다시 시도해 주세요.')
        self.busy = True
        try:
            started = time.perf_counter()
            values = torch.from_numpy(np.ascontiguousarray(waveform)[None, :]).to(self.device)
            mask = torch.ones(values.shape, dtype=torch.long, device=self.device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                hidden = self.model(input_values=values, attention_mask=mask).last_hidden_state
            logits = self.head(hidden.float())
            length = int(self.model._get_feat_extract_output_lengths(mask.sum(-1))[0])
            if not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite model output')
            text, jamo, ids = decode(logits[0, :length].argmax(-1).cpu().tolist(), self.tokens)
            elapsed = time.perf_counter() - started
            return {'text': text, 'jamo': jamo, 'token_ids': ids, 'frames': length,
                    'audio_seconds': len(waveform) / SAMPLE_RATE, 'inference_seconds': elapsed,
                    'uncomposed_jamo': sum(0x1100 <= ord(ch) <= 0x11ff for ch in text)}
        finally:
            self.busy = False
            self.lock.release()
