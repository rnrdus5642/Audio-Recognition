"""Read-only deployment evaluation through the existing local GPU preview API.

Writes only evaluation artifacts, never training state or user recording history.
Replays the actual WordGames policy using cached, target-independent predictions.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import sys
import time
from types import SimpleNamespace
import unicodedata

import httpx
import numpy as np
from rapidfuzz.distance import Levenshtein
import soundfile as sf

from word_game import WordGames, StartWordGame

ROOT = Path(__file__).resolve().parents[4]
MANIFESTS = ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세/manifests-20261027'
BUNDLE = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/releases/epoch2-preview-20261029'
BASE = 'http://127.0.0.1:8782'
RATE = 16000
SEED = 20261030


def safe(path):
    path = (ROOT / path).resolve()
    if path.drive.upper() != 'C:' or not path.is_relative_to(ROOT):
        raise ValueError(f'Path outside C-drive project: {path}')
    return path


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = safe(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_rows(split):
    with gzip.open(MANIFESTS / f'{split}.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def normalize(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def encoded(wave):
    stream = io.BytesIO()
    sf.write(stream, wave, RATE, format='WAV', subtype='PCM_16')
    return stream.getvalue()


def waveform(row):
    path = safe(row['audio_path'])
    blob = path.read_bytes()
    if row.get('audio_sha256') and hashlib.sha256(blob).hexdigest() != row['audio_sha256']:
        raise ValueError(f'Audio changed: {path}')
    wave, rate = sf.read(io.BytesIO(blob), dtype='float32')
    if rate != RATE or wave.ndim != 1:
        raise ValueError(f'Unexpected audio contract: {path}')
    return wave


def local_rows():
    rows = []
    music = ROOT / '03_Recordings/악기놀이녹음본/분할_라벨링_2026-09-11'
    for row in read(music / '라벨목록.json')['clips']:
        rows.append({'record_id': 'music:' + row['id'], 'text': row['label'], 'age': 'unknown',
                     'speaker_key': 'music:' + row['speaker_id'], 'group': 'music',
                     'audio_path': str((music / row['audio_file']).relative_to(ROOT)),
                     'audio_sha256': row['audio_sha256'], 'seconds': row['duration_seconds'],
                     'label_note': 'intended word / sequence label, listening audit incomplete'})
    child = ROOT / '03_Recordings/아동오디오/단어별_라벨링_20260906'
    for line in (child / 'labels.jsonl').read_text(encoding='utf-8-sig').splitlines():
        row = json.loads(line)
        rows.append({'record_id': 'child:' + row['id'], 'text': row['text'], 'age': 'unknown',
                     'speaker_key': row['speaker'], 'group': 'local_child',
                     'audio_path': str((child / row['audio']).relative_to(ROOT)),
                     'audio_sha256': row['audio_sha256'], 'seconds': row['duration_s'],
                     'expected_answer': row.get('expected_answer'),
                     'label_note': 'provided transcript, segmentation listening audit incomplete'})
    for number in (44, 45, 46, 47, 48, 50, 51):
        path = ROOT / f'01_Project/.venv/grab-real-0910/recording-{number}.wav'
        rows.append({'record_id': f'user:{number}', 'text': '잡아' if number < 50 else '작아',
                     'age': 'unknown', 'speaker_key': 'user', 'group': 'user_grab_control',
                     'audio_path': str(path.relative_to(ROOT)), 'seconds': sf.info(path).duration,
                     'label_note': 'user intended-word label'})
    return rows


def sample_words(rows, per_age):
    chosen = []
    for age in (3, 4, 5, 6):
        groups = defaultdict(list)
        for row in rows:
            if row['age'] == age and row['seconds'] <= 4.5 and re.fullmatch('[가-힣]{1,20}', row['text'].strip()):
                groups[row['speaker_key']].append(row)
        rng = random.Random(SEED + age)
        keys = sorted(groups)
        rng.shuffle(keys)
        for values in groups.values():
            values.sort(key=lambda row: row['record_id'])
            rng.shuffle(values)
        selected = []
        while len(selected) < per_age:
            added = 0
            for key in keys:
                if groups[key] and len(selected) < per_age:
                    selected.append({**groups[key].pop(), 'group': 'dev_word'})
                    added += 1
            if not added:
                break
        chosen.extend(selected)
    return chosen


def stats(rows):
    if not rows:
        return {'clips': 0}
    errors = sum(row['errors'] for row in rows)
    characters = sum(row['reference_characters'] for row in rows)
    return {'clips': len(rows), 'speakers': len({row['speaker_key'] for row in rows}),
            'seconds': sum(row['seconds'] for row in rows), 'cer': errors / characters,
            'errors': errors, 'reference_characters': characters,
            'exact': sum(row['exact'] for row in rows), 'exact_rate': sum(row['exact'] for row in rows) / len(rows),
            'nonblank_rate': sum(bool(row['prediction']['text']) for row in rows) / len(rows),
            'uncomposed_clips': sum(row['prediction'].get('uncomposed_jamo', 0) > 0 for row in rows)}


class Evaluator:
    def __init__(self, output):
        self.output = safe(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(base_url=BASE, timeout=60)
        self.status = self.client.get('/api/status').raise_for_status().json()
        assert self.status['ready'] and self.status['device'] == 'cuda'
        release = read(BUNDLE / 'release.json')
        assert self.status['model_id'] == release['files']['model.pt']['sha256']
        self.cache = {}
        self.cache_file = self.output / 'predictions.jsonl'
        if self.cache_file.exists():
            for line in self.cache_file.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                assert item['model_id'] == self.status['model_id']
                self.cache[item['audio_hash']] = item['prediction']
        self.calls, self.hit_count = 0, 0
        self.games = WordGames(SimpleNamespace(ready=True), lambda wave: self.pending,
                               lambda state, frame: {'prediction': frame['prediction'],
                                                     'similarity': frame['similarity'],
                                                     'candidate': frame['candidate']})

    def infer(self, wave):
        blob = encoded(wave)
        digest = hashlib.sha256(blob).hexdigest()
        if digest in self.cache:
            self.hit_count += 1
            return self.cache[digest]
        for attempt in range(30):
            response = self.client.post('/api/preview', files={'file': ('evaluation.wav', blob, 'audio/wav')})
            if response.status_code not in (409, 503):
                response.raise_for_status()
                break
            time.sleep(.25)
        else:
            raise RuntimeError('Web model remained busy')
        result = response.json()
        assert result['model_id'] == self.status['model_id']
        prediction = result['prediction']
        self.cache[digest] = prediction
        with self.cache_file.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'model_id': result['model_id'], 'audio_hash': digest,
                                     'prediction': prediction}, ensure_ascii=False) + '\n')
        self.calls += 1
        time.sleep(.015)  # Leave gaps for interactive use; only one inference at a time.
        return prediction

    def asr(self, row):
        prediction = self.infer(waveform(row))
        ref, hyp = normalize(row['text']), normalize(prediction['text'])
        return {**row, 'prediction': prediction, 'reference_characters': len(ref),
                'errors': Levenshtein.distance(ref, hyp), 'exact': ref == hyp}

    def replay(self, row, bank):
        wave = waveform(row)
        target = row['text'].strip()
        rng = random.Random(f'{SEED}:{row["record_id"]}')
        others = sorted(set(bank) - {target})
        random_candidates = [target] + rng.sample(others, min(9, len(others)))
        rng.shuffle(random_candidates)
        phonemes = self.games.g2p.to_ipa(target)
        # Stress condition: a different word below the pass threshold on ideal
        # text. This avoids calling an intentional >=85-point allowance an ASR error.
        distances = [(self.games.matcher.score_against(phonemes, self.phonemes[word])[1], word) for word in others]
        below = sorted((pair for pair in distances if pair[0] < .85), reverse=True)
        wrong = below[0][1]
        configs = {'single': [target], 'multi10': random_candidates, 'wrong_near_target': [wrong]}
        if row.get('expected_answer') and row['expected_answer'] != target and re.fullmatch('[가-힣]{1,20}', row['expected_answer']):
            configs['annotated_wrong_answer'] = [row['expected_answer']]
        if row['group'] == 'user_grab_control' and target == '작아':
            configs['jaba_control'] = ['잡아']
        if row['group'] == 'music':
            configs['music12'] = self.music_bank
        sessions = {name: self.games.start(StartWordGame(targets=words))['session_id'] for name, words in configs.items()}
        results, frames = {}, []
        signal = np.zeros(RATE * 10, dtype=np.float32)
        count = min(len(wave), len(signal) - int(.4 * RATE))
        signal[int(.4 * RATE):int(.4 * RATE) + count] = wave[:count]
        for index in range(1, 21):
            end = index * 8000
            chunk = signal[max(0, end - 40000):end]
            chunk = np.pad(chunk, (40000 - len(chunk), 0))
            prediction = self.infer(chunk)
            self.pending = prediction
            canonical = encoded(chunk)
            frames.append({'elapsed_seconds': index * .5, 'prediction': prediction})
            for name, session in sessions.items():
                if name in results:
                    continue
                answer = self.games.frame(session, index, index * .5, chunk, canonical, RATE)
                if answer['state'] != 'listening':
                    results[name] = answer
            if len(results) == len(sessions):
                break
        for name, session in sessions.items():
            if name not in results:
                results[name] = self.games.finish(session, 'timeout')
            results[name]['trace'] = self.games.get(session)['trace']
        return {**row, 'trials': results, 'frames': frames,
                'wrong_near_target': wrong, 'ideal_wrong_target_score': below[0][0],
                'any_exact_window': any(normalize(frame['prediction']['text']) == normalize(target) for frame in frames)}

    def noise(self, bank):
        rows = []
        for kind, level in [('silence', 0), ('white_noise_quiet', .005), ('white_noise', .03)]:
            for seed in range(3):
                rng = np.random.default_rng(SEED + seed)
                signal = rng.normal(0, level, RATE * 10).astype(np.float32) if level else np.zeros(RATE * 10, dtype=np.float32)
                session = self.games.start(StartWordGame(targets=bank))['session_id']
                for index in range(1, 21):
                    chunk = signal[max(0, index * 8000 - 40000):index * 8000]
                    chunk = np.pad(chunk, (40000 - len(chunk), 0))
                    blob = encoded(chunk)
                    # Match server PCM quantization exactly for the energy gate.
                    chunk, _ = sf.read(io.BytesIO(blob), dtype='float32')
                    self.pending = self.infer(chunk)
                    answer = self.games.frame(session, index, index * .5, chunk, blob, RATE)
                    if answer['state'] != 'listening':
                        break
                answer = self.games.finish(session, 'timeout')
                rows.append({'kind': kind, 'seed': seed, 'result': answer})
        return rows


def game_stats(rows, trial):
    valid = [row for row in rows if trial in row['trials']]
    accepted = [row for row in valid if row['trials'][trial]['state'] == 'accepted']
    correct = [row for row in accepted if row['trials'][trial]['matched_target'] == row['text'].strip()]
    times = [row['trials'][trial]['elapsed_seconds'] for row in correct]
    return {'clips': len(valid), 'speakers': len({r['speaker_key'] for r in valid}), 'accepted': len(accepted),
            'correct': len(correct), 'other_candidate': len(accepted) - len(correct),
            'retry': len(valid) - len(accepted), 'accept_rate': len(accepted) / len(valid) if valid else None,
            'correct_rate': len(correct) / len(valid) if valid else None,
            'correct_decision_seconds_median': float(np.median(times)) if times else None,
            'correct_decision_seconds_p90': float(np.percentile(times, 90)) if times else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--words-per-age', type=int, default=80)
    args = parser.parse_args()
    output = args.output or ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test' / datetime.now().strftime('performance-%Y%m%d-%H%M%S')
    ev = Evaluator(output)
    started = time.monotonic()
    monitor = [{**r, 'group': 'monitor'} for r in load_rows('monitor')]
    dev = load_rows('dev')
    local = local_rows()
    selected = sample_words(dev, args.words_per_age)
    game_rows = selected + [r for r in local if re.fullmatch('[가-힣]{1,20}', r['text'].strip()) and r['seconds'] <= 9.6]
    bank = sorted({r['text'].strip() for r in game_rows})
    ev.phonemes = {word: ev.games.g2p.to_ipa(word) for word in bank}
    ev.music_bank = sorted({r['text'] for r in local if r['group'] == 'music'})
    words = sorted(set(bank) | {'놔요','놔','내려놔','잡아','작아','책','짹','택','실로폰채','들어요','들어야지','들거야','들자','들거에요','들어야죠','들까','들까요','물음표','읽다','앉다','값'})
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_orthographic_release'))
    from inference import decode
    vocab = read(BUNDLE / 'vocab.json')
    tokens = {v:k for k,v in vocab.items()}
    failures = []
    for number in range(0xac00, 0xd7a4):
        text = chr(number)
        decomposed = unicodedata.normalize('NFD', text)
        if any(char not in vocab for char in decomposed) or decode([vocab[char] for char in decomposed], tokens)[0] != text:
            failures.append(text)
    coverage = {'hangul_syllables_checked': 11172, 'hangul_failures': failures,
                'word_missing_tokens': {word:[char for char in unicodedata.normalize('NFD',word) if char not in vocab] for word in words},
                'ui_limits': '1-20 candidates; each Hangul word 1-20 syllables; terminal ? ignored; spaces/digits/Latin rejected'}
    plan = {'created_at': datetime.now().astimezone().isoformat(), 'model': ev.status,
            'monitor_clips': len(monitor), 'local_clips': len(local), 'word_clips': len(game_rows),
            'word_clips_by_age': dict(Counter(str(r['age']) for r in selected)),
            'game_policy': {'window_seconds':2.5,'hop_seconds':.5,'threshold':.85,'consecutive':2,'timeout_seconds':10},
            'replay': '0.4s leading silence + unchanged full clip + silence up to 10s; 0.5s virtual hops; no wall-clock/network latency simulation',
            'selection': 'dev single-token Hangul labels <=4.5s; fixed seed; speaker round-robin within age; plus preserved local recordings',
            'candidate_policy': 'single reference; 10 random candidates including reference; closest distinct target with ideal similarity below 0.85; actual 12-word music bank',
            'label_limits': 'AIHub intended spelling, not phonetic ground truth; local labels have incomplete human listening review; local ages unknown',
            'data_splits': 'monitor is a subset of dev; used during training selection, NOT final untouched test',
            'seed':SEED, 'coverage':coverage, 'candidate_bank':bank}
    write(ev.output/'plan.json', plan)
    write(ev.output/'selected-word-records.json',game_rows)
    print(json.dumps({'event':'start','output':str(ev.output),'monitor':len(monitor),'local':len(local),'game':len(game_rows)},ensure_ascii=False),flush=True)

    asr = []
    for index, row in enumerate(monitor + local + selected, 1):
        asr.append(ev.asr(row))
        if index % 100 == 0:
            write(ev.output/'progress.json',{'phase':'asr','done':index,'total':len(monitor)+len(local)+len(selected),'seconds':time.monotonic()-started})
            print(json.dumps({'phase':'asr','done':index,'seconds':round(time.monotonic()-started,1)}),flush=True)
    write(ev.output/'asr-results.json',asr)
    asr_groups = {group: stats([r for r in asr if r['group']==group]) for group in sorted({r['group'] for r in asr})}
    asr_ages = {str(age):stats([r for r in asr if r['group']=='monitor' and r['age']==age]) for age in (3,4,5,6)}
    print(json.dumps({'phase':'asr_complete','monitor_by_age':asr_ages,'groups':asr_groups},ensure_ascii=False),flush=True)

    games = []
    for index, row in enumerate(game_rows, 1):
        games.append(ev.replay(row,bank))
        with (ev.output/'game-results.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(games[-1],ensure_ascii=False)+'\n')
        if index % 20 == 0:
            write(ev.output/'progress.json',{'phase':'games','done':index,'total':len(game_rows),'seconds':time.monotonic()-started})
            print(json.dumps({'phase':'games','done':index,'total':len(game_rows),'seconds':round(time.monotonic()-started,1)}),flush=True)
    noise = ev.noise(ev.music_bank)
    write(ev.output/'noise-results.json',noise)
    per_word = []
    for word in sorted({r['text'] for r in games}):
        group=[r for r in games if r['text']==word]
        per_word.append({'word':word,'groups':dict(Counter(r['group'] for r in group)),
                         **game_stats(group,'single'),
                         'any_exact_window':sum(r['any_exact_window'] for r in group),
                         'examples':[{'reference':r['text'],'source':r['audio_path'],
                                      'best_prediction':r['trials']['single']['record']['prediction']['text'],
                                      'best_score':r['trials']['single']['record']['similarity'],
                                      'outcome':r['trials']['single']['state']} for r in group if r['trials']['single']['state']!='accepted'][:3]})
    write(ev.output/'word-results.json',per_word)
    timing=[r['inference_seconds'] for r in ev.cache.values()]
    summary = {'status':'complete','plan':plan,'asr_groups':asr_groups,'monitor_by_age':asr_ages,
               'monitor_macro_cer':sum(v['cer'] for v in asr_ages.values())/4,
               'game_by_group':{group:{name:game_stats([r for r in games if r['group']==group],name) for name in ('single','multi10','wrong_near_target','music12','jaba_control','annotated_wrong_answer')} for group in sorted({r['group'] for r in games})},
               'game_dev_by_age':{str(age):{name:game_stats([r for r in games if r['group']=='dev_word' and r['age']==age],name) for name in ('single','multi10','wrong_near_target')} for age in (3,4,5,6)},
               'zero_pass_words':[r for r in per_word if r['accepted']==0],
               'noise_trials':len(noise),'noise_false_accepts':sum(r['result']['state']=='accepted' for r in noise),
               'inference_calls':ev.calls,'cache_hits':ev.hit_count,
               'inference_seconds_p50':float(np.median(timing)),'inference_seconds_p95':float(np.percentile(timing,95)),
               'elapsed_seconds':time.monotonic()-started}
    write(ev.output/'summary.json',summary)
    write(ev.output/'progress.json',{'status':'complete','seconds':summary['elapsed_seconds']})
    print(json.dumps({'event':'complete','output':str(ev.output),'monitor_macro_cer':summary['monitor_macro_cer'],'game_by_group':summary['game_by_group'],'noise_false_accepts':summary['noise_false_accepts'],'seconds':summary['elapsed_seconds']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
