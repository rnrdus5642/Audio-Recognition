"""Frozen spelling/jamo contract and paths for the new child ASR run."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata as ud

ROOT = Path(__file__).resolve().parents[4]
CODE = Path(__file__).resolve().parent
PREP = ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세'
WORK = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027'
MANIFESTS = PREP / 'manifests-20261027'
PLAN = ROOT / '01_Project/docs/data/orthographic-jamo-finetune-plan-20261027.json'
ENCODER = ROOT / '01_Project/.venv/encoder-screen-0914/models/hubertko1b'
VOCAB = json.loads((PREP / 'vocab.json').read_text(encoding='utf-8'))
TOKENS = {i:t for t,i in VOCAB.items()}
POLICY = json.loads((PREP / 'label-policy.json').read_text(encoding='utf-8'))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(40):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def normalize(original):
    text = ud.normalize('NFC', original).replace('\ufeff', '')
    for a,b in POLICY['normalization']['fullwidth_punctuation'].items():
        text = text.replace(a,b)
    for ch in POLICY['normalization']['replace_with_space']:
        text = text.replace(ch, ' ')
    text = ' '.join(text.split())
    if not text or not re.fullmatch('[가-힣 ]+', text):
        raise ValueError('Empty or unresolved orthographic label')
    return text


def encode(text):
    return [VOCAB['|' if ch == ' ' else ch] for ch in ud.normalize('NFD', text)]


def decode(ids):
    units, previous = [], None
    for i in ids:
        if i != previous and i != 0:
            units.append(' ' if i == 1 else TOKENS[i])
        previous = i
    return ud.normalize('NFC', ''.join(units))


def frames(samples):
    for k,s in zip([10,3,3,3,3,2,2], [5,2,2,2,2,2,2]):
        samples = (samples-k)//s+1
    return samples


def ctc_width(ids):
    return len(ids) + sum(a == b for a,b in zip(ids,ids[1:]))


def rows(split):
    with gzip.open(MANIFESTS / f'{split}.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def safe_audio(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or path.drive.upper() != 'C:':
        raise ValueError(f'Audio outside project C-drive root: {relative}')
    return path


def log(run, event, **values):
    record = {'at':time.strftime('%Y-%m-%d %H:%M:%S'), 'event':event, **values}
    with (Path(run)/'events.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)
