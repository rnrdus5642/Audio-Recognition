"""Isolated pretrained-encoder screening; never imports the production ASR model."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.venv/encoder-screen-0914'
WORK.mkdir(parents=True, exist_ok=True)
MODELS = {
    'xlsr300': {'repo': 'facebook/wav2vec2-xls-r-300m', 'family': 'wav2vec2', 'tier': 'primary'},
    'omni300': {'url': 'https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt', 'family': 'omni', 'tier': 'primary'},
    'w2vbert': {'repo': 'facebook/w2v-bert-2.0', 'family': 'wav2vec2-bert', 'tier': 'primary'},
    'hubertko': {'repo': 'team-lucid/hubert-large-korean', 'family': 'hubert', 'tier': 'primary'},
    'wavlm': {'repo': 'microsoft/wavlm-large', 'family': 'wavlm', 'tier': 'control'},
    'huberten': {'repo': 'facebook/hubert-large-ll60k', 'family': 'hubert', 'tier': 'control'},
    'xlsr1b': {'repo': 'facebook/wav2vec2-xls-r-1b', 'family': 'wav2vec2', 'tier': 'large'},
    'omni1b': {'url': 'https://dl.fbaipublicfiles.com/mms/omniASR-W2V-1B.pt', 'family': 'omni', 'tier': 'large'},
    'hubertko1b': {'repo': 'team-lucid/hubert-xlarge-korean', 'family': 'hubert', 'tier': 'large'},
    'mms1b': {'repo': 'facebook/mms-1b', 'family': 'wav2vec2', 'tier': 'restricted-license'},
    'xeus': {'repo': 'espnet/xeus', 'family': 'xeus', 'tier': 'restricted-license', 'weight': 'model/xeus_checkpoint_new.pth'},
}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
