"""The trained model, resident once, answering one request at a time.

Inference here is the same computation training measured with: FP32, no
autocast, an explicit attention mask, true output lengths, and the same
greedy CTC decode. If this drifted from `train.py`, the page would be
showing numbers that no recorded score supports.

The reference transcript never reaches this module's prediction path. It
is used afterwards, by the caller, only to score what was already
decided.
"""
from pathlib import Path
import hashlib
import io
import sys
import threading
import time

import numpy as np
import soundfile as sf
import torch

FINETUNE = Path(__file__).resolve().parents[1] / 'child_jamo_finetune'
sys.path.insert(0, str(FINETUNE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio as source_audio
import compose
import labels
from common import ENCODER, WORK as TRAIN_WORK, digest, load

CHECKPOINT = TRAIN_WORK / 'main/best.pt'
RESULT = TRAIN_WORK / 'main/result.json'
SAMPLE_RATE = source_audio.SAMPLE_RATE
MAX_UPLOAD_SECONDS = 30.0
#: Below the encoder's own receptive field there is nothing to decode.
MIN_UPLOAD_SAMPLES = 1600


class DurationTooLong(ValueError):
    def __init__(self, seconds):
        super().__init__(f'{MAX_UPLOAD_SECONDS:.0f}초 이하만 인식합니다 '
                         f'(올린 파일 {seconds:.1f}초)')
        self.seconds = seconds


class Engine:
    """One resident model. Concurrent callers queue rather than duplicate it."""

    def __init__(self, device=None):
        # A repaired or fresh environment may only have the CPU PyTorch wheel.
        # Keep the viewer usable there instead of failing during model load.
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.ready = False
        self.error = None
        self.identity = {}
        self.model = None
        self.head = None
        self._cache = {}
        self._loading = threading.Event()

    # --- loading -----------------------------------------------------

    def load(self):
        """Idempotent; the first caller loads and the rest wait for it."""
        with self.lock:
            if self.ready or self.error:
                return self.ready
            try:
                self._load_locked()
                self.ready = True
            except Exception as exc:                  # surfaced through /status
                self.error = f'{type(exc).__name__}: {exc}'
                raise
        return self.ready

    def _load_locked(self):
        from transformers import HubertModel
        started = time.monotonic()
        # Match the training run's matmul policy explicitly rather than
        # inheriting whatever this process happened to default to.
        torch.backends.cuda.matmul.allow_tf32 = False

        checkpoint_sha = digest(CHECKPOINT)
        saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        contract = saved['config']['data_contract']

        # The head only means anything against the label list it was
        # trained on, so a changed contract is refused rather than
        # rendered with today's token order.
        if contract['labels_contract_sha256'] != labels.contract_sha256():
            raise RuntimeError(
                'label contract has changed since this checkpoint was '
                f'trained ({contract["labels_contract_sha256"][:16]} vs '
                f'{labels.contract_sha256()[:16]})')
        # The whole encoder identity from the frozen contract, not just the
        # weights: preprocessor_config.json is what declares raw
        # unnormalised input, so a swapped one changes the computation
        # without touching a single weight.
        encoder_files = {}
        for name in ['model.safetensors', 'config.json',
                     'preprocessor_config.json']:
            actual = digest(ENCODER / name)
            if contract['encoder'].get(name) != actual:
                raise RuntimeError(
                    f'encoder {name} differs from the one this checkpoint was '
                    f'trained against')
            encoder_files[name] = actual

        # The page claims a specific recorded test score. Only show it if
        # this checkpoint is the one that score was produced from.
        recorded = load(RESULT) if RESULT.exists() else {}
        historical = None
        provenance = 'result.json 이 없어 과거 성적을 표시하지 않습니다'
        if recorded:
            same_checkpoint = recorded.get('selected_checkpoint') == saved.get('valid')
            same_contract = (recorded.get('config', {}).get('data_contract')
                             == contract)
            if same_checkpoint and same_contract:
                historical = recorded.get('test')
                provenance = ('result.json 의 selected_checkpoint 및 '
                              'data_contract 가 이 체크포인트와 일치')
            else:
                provenance = (
                    'result.json 이 이 체크포인트를 가리키지 않아 과거 성적을 '
                    f'숨깁니다 (checkpoint 일치={same_checkpoint}, '
                    f'contract 일치={same_contract})')

        model = HubertModel.from_pretrained(
            ENCODER, local_files_only=True, attn_implementation='sdpa',
            dtype=torch.float32)
        model.config.layerdrop = 0.0
        # eval() already disables masking, but saying so explicitly keeps
        # the served computation from depending on that detail.
        model.config.apply_spec_augment = False
        head = torch.nn.Linear(model.config.hidden_size, len(labels.TOKENS))

        model.load_state_dict(saved['model'], strict=True)
        head_state = {k.replace('linear.', ''): v
                      for k, v in saved['head'].items()}
        head.load_state_dict(head_state, strict=True)

        self.model = model.to(self.device).eval()
        self.head = head.to(self.device).eval().float()
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(False)

        self.identity = {
            'checkpoint': str(CHECKPOINT),
            'checkpoint_bytes': CHECKPOINT.stat().st_size,
            # The fine-tuned weights, which is what this page is showing.
            'checkpoint_sha256': checkpoint_sha,
            'selected_by_validation': saved.get('valid'),
            'labels_version': labels.LABELS_VERSION,
            'labels_contract_sha256': labels.contract_sha256(),
            'encoder': contract['encoder']['path'],
            'encoder_files_sha256': encoder_files,
            'decode_version': compose.DECODE_VERSION,
            'precision': 'float32, no autocast',
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'attention': model.config._attn_implementation,
            'device': torch.cuda.get_device_name(0)
            if self.device == 'cuda' else 'cpu',
            'parameters': sum(p.numel() for p in model.parameters()),
            'load_seconds': round(time.monotonic() - started, 1),
            'historical_test': historical,
            'historical_provenance': provenance,
            'historical_note': 'Aggregate over the whole 2,999-recording test '
                               'split for this fixed checkpoint. It is not the '
                               'score of whatever sample is on screen.',
        }
        del saved

    def status(self):
        return {'ready': self.ready, 'error': self.error, **self.identity}

    # --- inference ---------------------------------------------------

    def cache_key(self, pcm_sha256):
        """Identity of a result: these weights, this audio, this decoder.

        Keyed on the fine-tuned checkpoint rather than the base encoder -
        two different fine-tunes share a base and would otherwise collide
        on the same audio.
        """
        return f'{self.identity.get("checkpoint_sha256", "")[:16]}:' \
               f'{self.identity.get("labels_contract_sha256", "")[:12]}:' \
               f'{compose.DECODE_VERSION}:{pcm_sha256}'

    @torch.inference_mode()
    def transcribe(self, waveform, pcm_sha256=None, use_cache=True):
        """Greedy CTC over one waveform. No reference is accepted here."""
        if not self.ready:
            self.load()
        key = self.cache_key(pcm_sha256) if pcm_sha256 else None
        if key and use_cache and key in self._cache:
            return dict(self._cache[key], cached=True)

        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        with self.lock:                       # one inference at a time
            started = time.monotonic()
            values = torch.from_numpy(waveform[None, :]).to(self.device)
            mask = torch.ones(values.shape, dtype=torch.long,
                              device=self.device)
            hidden = self.model(input_values=values,
                                attention_mask=mask).last_hidden_state
            logits = self.head(hidden.float())
            lengths = self.model._get_feat_extract_output_lengths(
                mask.sum(-1)).to(torch.long)
            if not torch.isfinite(logits).all():
                raise RuntimeError('non-finite logits')
            width = int(lengths[0])
            frames = logits[0, :width]
            argmax = frames.argmax(-1).cpu().tolist()
            probability = frames.softmax(-1).max(-1).values.mean().item()
            elapsed = time.monotonic() - started

        ids = compose.collapse(argmax)
        rendered = compose.compose(ids)
        out = {
            'token_ids': ids,
            'jamo': rendered['jamo'],
            'jamo_tokens': rendered['jamo_tokens'],
            'text': rendered['text'],
            'segments': rendered['segments'],
            'syllables': rendered['syllables'],
            'orphan_jamo': rendered['orphan_jamo'],
            'fully_composed': rendered['fully_composed'],
            'frames': width,
            'samples': int(len(waveform)),
            'seconds': len(waveform) / SAMPLE_RATE,
            'mean_frame_probability': probability,
            'inference_seconds': elapsed,
            'realtime_factor': (len(waveform) / SAMPLE_RATE) / elapsed
            if elapsed else None,
            'cached': False,
            'decode_version': compose.DECODE_VERSION,
        }
        if key and use_cache:
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = out
        return out


# --- audio -----------------------------------------------------------

def read_row(row):
    """The exact waveform training would have fed for this manifest row.

    The audit recorded a hash of the cropped PCM, so that is what is
    checked - not merely the sample count. Two different recordings of
    the same length would otherwise pass, and the cached prediction is
    keyed on that hash, so a changed waveform could be served with text
    that belongs to the audio it replaced.

    Personally recorded rows are whole files rather than archive members,
    so they are read by `personal`, which applies the same contract: the
    catalogued sample count and PCM hash both have to match.
    """
    if row.get('reader') == 'personal':
        # Imported inside the call: `personal` is what builds those rows,
        # and a module-level import here would be a cycle.
        import personal
        return personal.read(row)
    blob = source_audio.payload(row['archive'], row['offset'], row['size'])
    waveform, policy = source_audio.crop(source_audio.decode(blob),
                                         row['speech_start'], row['speech_end'])
    if len(waveform) != row['samples'] or policy != row['span_policy']:
        raise ValueError(
            f'{row["wav"]}: read {len(waveform)} samples / {policy}, manifest '
            f'says {row["samples"]} / {row["span_policy"]}')
    recorded = row.get('pcm_sha256')
    if recorded:
        actual = pcm_sha256(waveform)
        if actual != recorded:
            raise ValueError(
                f'{row["wav"]}: audio no longer matches the audited recording '
                f'(pcm {actual[:16]} vs {recorded[:16]}); the archive or the '
                f'manifest has changed')
    return waveform


def canonical_upload(blob):
    """One waveform for both inference and playback, for uploaded files.

    The browser is served 16-bit PCM, so if inference ran on the
    unquantised float the page would be playing a slightly different
    signal from the one it describes. Encoding and decoding once here
    makes the served bytes and the model's input the same array, and any
    clipping of out-of-range input happens once, visibly, before either.

    Duration is checked from the header before anything is decoded, so an
    oversized file is refused rather than resampled first.
    """
    stream = io.BytesIO(blob)
    try:
        info = sf.info(stream)
    except Exception as exc:
        raise ValueError(f'음성 헤더를 읽지 못했습니다: {exc}')
    if not info.frames or not info.samplerate:
        raise ValueError('음성 데이터가 비어 있습니다')
    header_seconds = info.frames / info.samplerate
    if header_seconds > MAX_UPLOAD_SECONDS:
        raise DurationTooLong(header_seconds)

    stream.seek(0)
    data, rate = sf.read(stream, dtype='float32', always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        divisor = gcd(int(rate), SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // divisor,
                             int(rate) // divisor).astype('float32')
    if not np.isfinite(data).all():
        raise ValueError('음성에 유효하지 않은 값이 있습니다')
    seconds = len(data) / SAMPLE_RATE
    if seconds > MAX_UPLOAD_SECONDS:
        raise DurationTooLong(seconds)
    if len(data) < MIN_UPLOAD_SAMPLES:
        raise ValueError(f'너무 짧습니다 ({seconds:.2f}초). '
                         f'{MIN_UPLOAD_SAMPLES / SAMPLE_RATE:.2f}초 이상 필요합니다')

    # Round-trip through the exact bytes the browser will play.
    encoded = wav_bytes(np.ascontiguousarray(data, dtype=np.float32))
    final, final_rate = sf.read(io.BytesIO(encoded), dtype='float32',
                                always_2d=False)
    assert final_rate == SAMPLE_RATE
    return np.ascontiguousarray(final, dtype=np.float32), encoded, int(rate)


def wav_bytes(waveform):
    """16-bit PCM WAV of exactly the signal that went into the model."""
    buffer = io.BytesIO()
    sf.write(buffer, waveform, SAMPLE_RATE, subtype='PCM_16', format='WAV')
    return buffer.getvalue()


def pcm_sha256(waveform):
    return hashlib.sha256(
        np.ascontiguousarray(waveform, dtype=np.float32).tobytes()).hexdigest()
