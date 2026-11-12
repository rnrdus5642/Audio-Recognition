"""Read-only v3/v4 stress evaluation: ideal text, real audio, and distractors.

Writes reproducible local artifacts only. No training, model, web configuration,
or user history changes. Actual inference goes through /api/preview, one at a time.
"""
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf

from answer_policy import spelling_key
from evaluate_performance import ROOT, RATE, Evaluator, encoded, read, safe, waveform, write
from verify_postprocessing import Replay
from word_game import WordGames, StartWordGame

OUT = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/stress-20261030-044247'
WEB = OUT.parent
CURRICULUM = WEB / 'curriculum-20261030-033454'
CONDITIONS = ('alone', 'speech_before', 'speech_before_gap', 'speech_after', 'repeat_joined',
              'repeat_gap', 'noise_before', 'noise_20db', 'noise_10db')
SEED = 202610300442


def dump_line(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def exact_present(text, targets):
    """Diagnostic only: a complete spelling anywhere, including suffix speech.

    Not the production suffix restriction; distinguishes correct text missed by
    that restriction. Does not call a syllable inside another word a correct word.
    """
    words = text.split()
    keys = {spelling_key(target) for target in targets}
    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            span = spelling_key(''.join(words[start:end]))
            if span in keys:
                return True
            if any(len(word) >= 2 and span == word * n for word in keys for n in (2, 3, 4)):
                return True
    return False


class Audit:
    def __init__(self):
        self.stages = read(CURRICULUM / 'catalog.json')
        namespace = {'__file__': str(Path(__file__).with_name('word_game.py')), '__name__': 'baseline_word_game'}
        exec(compile((WEB / 'postprocessing-20261030/before/word_game.py').read_text(encoding='utf-8'),
                     namespace['__file__'], 'exec'), namespace)
        self.before = Replay(namespace['WordGames'], namespace['StartWordGame'])
        self.after = Replay(WordGames, StartWordGame)
        self.ev = None
        self.results = []
        self.done = set()
        self.path = OUT / 'trials.jsonl'
        if self.path.exists():
            self.results = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
            self.done = {row['id'] for row in self.results}

    def compare(self, identity, phase, condition, targets, frames, positive, **meta):
        if identity in self.done:
            return
        old = self.before.run(targets, frames)
        new = self.after.run(targets, frames)
        exact = [f['seconds'] for f in frames if f['voice_fraction'] >= .04 and exact_present(f['text'], targets)]
        raw = [f['seconds'] for f in frames if f['voice_fraction'] >= .04 and
               any(spelling_key(word) in spelling_key(f['text']) for word in targets)]
        result = {'id': identity, 'phase': phase, 'condition': condition, 'targets': targets, 'positive': positive,
                  'exact_word_checks': len(exact), 'exact_word_times': exact, 'raw_substring_checks': len(raw),
                  'before': old, 'after': new, **meta}
        if positive and meta.get('target_start') is not None:
            for label in ('before', 'after'):
                result[label]['before_target_started'] = result[label]['accepted'] and result[label]['seconds'] <= meta['target_start']
        dump_line(self.path, result)
        self.results.append(result)
        self.done.add(identity)

    def ideal(self):
        all_words = sorted({word for stage in self.stages for word in stage['targets']})
        for stage in self.stages:
            for word in stage['targets']:
                sequences = {'clean': [word] * 2, 'prefix_spaced': ['안녕하세요 ' + word] * 2,
                             'prefix_joined': ['안녕하세요' + word] * 2,
                             'suffix_spaced': [word + ' 안녕하세요'] * 2,
                             'suffix_joined': [word + '안녕하세요'] * 2,
                             'repeat_joined': [word * 2] * 2, 'repeat_spaced': [word + ' ' + word] * 2,
                             'blank_gap': [word, '', word], 'jamo_gap': [word, 'ᄋ', word],
                             'single_correct_check': [word, '', '', '']}
                for condition, texts in sequences.items():
                    frames = [{'seconds': (i + 1) * .5, 'text': text, 'voice_fraction': .3 if text else 0.}
                              for i, text in enumerate(texts)]
                    self.compare(f'ideal:{stage["id"]}:{word}:{condition}', 'ideal_positive', condition,
                                 stage['targets'], frames, True, reference=word, stage=stage['id'])
            allowed_families = {self.after.games.action_guard.families.get(spelling_key(w)) for w in stage['targets']} - {None}
            for word in all_words:
                if word in stage['targets']:
                    continue
                family = self.after.games.action_guard.families.get(spelling_key(word))
                # A deliberately omitted same-action inflection is a separate
                # category, not a clear wrong-action false acceptance.
                condition = 'same_action_unlisted' if family in allowed_families else 'different_word'
                frames = [{'seconds': .5, 'text': word, 'voice_fraction': .3},
                          {'seconds': 1., 'text': word, 'voice_fraction': .3}]
                self.compare(f'ideal-negative:{stage["id"]}:{word}', 'ideal_negative', condition,
                             stage['targets'], frames, False, reference=word, stage=stage['id'])
        for heard, target in [('책상', '책'), ('물고기', '물'), ('알파벳', '알'), ('흙탕물', '흙'),
                              ('집게', '집'), ('북극', '북'), ('책상책상', '책'), ('하하', '하')]:
            frames = [{'seconds': .5, 'text': heard, 'voice_fraction': .3},
                      {'seconds': 1., 'text': heard, 'voice_fraction': .3}]
            self.compare('compound:' + heard, 'ideal_negative', 'internal_substring', [target], frames, False, reference=heard)
        print(json.dumps({'phase': 'ideal_complete', 'trials': len(self.results)}, ensure_ascii=False), flush=True)
        self.summarize()

    def setup_audio(self):
        self.ev = Evaluator(OUT)
        sources = [WEB / 'performance-20261030-025230/predictions.jsonl',
                   CURRICULUM / 'audio-evaluation/predictions.jsonl', WEB / 'context-fix-20261030/predictions.jsonl']
        for path in sources:
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    item = json.loads(line)
                    assert item['model_id'] == self.ev.status['model_id']
                    self.ev.cache[item['audio_hash']] = item['prediction']
        source_results = read(WEB / 'performance-20261030-025230/asr-results.json')
        self.prefixes = [next(row for row in source_results if row['text'].strip() == text and row['exact'])
                         for text in ('고양이', '사과', '강아지')]
        write(OUT / 'prefix-sources.json', self.prefixes)

    def frames(self, identity, signal):
        path = OUT / 'audio' / (identity + '.wav')
        path.parent.mkdir(exist_ok=True)
        blob = encoded(signal.astype(np.float32))
        if not path.exists():
            path.write_bytes(blob)
        else:
            assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(blob).digest()
        quantized, rate = sf.read(io.BytesIO(blob), dtype='float32')
        assert rate == RATE
        frames = []
        for index in range(1, math.ceil(len(quantized) / 8000) + 1):
            end = index * 8000
            segment = quantized[max(0, end - 40000):min(end, len(quantized))]
            right = max(0, end - len(quantized))
            segment = np.pad(segment, (max(0, 40000 - len(segment) - right), right))
            assert len(segment) == 40000
            prediction = self.ev.infer(segment)
            frames.append({'seconds': index * .5, 'text': prediction['text'],
                           'voice_fraction': WordGames.voice_fraction(segment), 'prediction': prediction})
        dump_line(OUT / 'audio-frames.jsonl', {'id': identity, 'audio_path': str(path.relative_to(ROOT)),
                                            'audio_sha256': hashlib.sha256(blob).hexdigest(), 'frames': frames})
        return frames, str(path.relative_to(ROOT))

    def paired_audio(self):
        rows = read(CURRICULUM / 'selected-audio.json')
        for number, row in enumerate(rows):
            audio = waveform(row)
            word = row['curriculum_word']
            prefix_row = self.prefixes[number % len(self.prefixes)]
            prefix = waveform(prefix_row)
            active = audio[np.abs(audio) >= .006]
            rms = float(np.sqrt(np.mean(active ** 2))) if len(active) else .01
            rng = np.random.default_rng(SEED + number)
            noise = rng.normal(size=len(audio)).astype(np.float32)
            for condition in CONDITIONS:
                identity = f'pair-{number:03d}-{condition}'
                if f'{identity}:single' in self.done:
                    continue
                start = .4
                if condition == 'alone': signal = audio
                elif condition.startswith('speech_before'):
                    gap = np.zeros(4000) if condition.endswith('_gap') else np.zeros(0)
                    signal = np.concatenate([prefix, gap, audio]); start += (len(prefix) + len(gap)) / RATE
                elif condition == 'speech_after': signal = np.concatenate([audio, prefix])
                elif condition.startswith('repeat_'):
                    gap = np.zeros(4000) if condition.endswith('_gap') else np.zeros(0)
                    signal = np.concatenate([audio, gap, audio])
                elif condition == 'noise_before':
                    lead = rng.normal(0, .025, 12800); signal = np.concatenate([lead, audio]); start += .8
                else:
                    db = 20 if condition == 'noise_20db' else 10
                    signal = np.clip(audio + noise * rms / 10 ** (db / 20), -.999, .999)
                signal = np.concatenate([np.zeros(6400), signal, np.zeros(41600)])
                frames, audio_path = self.frames(identity, signal)
                meta = {'reference': word, 'record_id': row['record_id'], 'source_path': row['audio_path'],
                        'speaker_key': row['speaker_key'], 'age': row['age'], 'audio_path': audio_path,
                        'prefix_source': prefix_row['audio_path'], 'target_start': start}
                for stage in self.stages:
                    if word in stage['targets']:
                        self.compare(identity + ':' + stage['id'], 'paired_stage', condition, stage['targets'], frames, True,
                                     stage=stage['id'], **meta)
                self.compare(identity + ':single', 'paired_single', condition, [word], frames, True, **meta)
            if (number + 1) % 4 == 0 or number + 1 == len(rows):
                print(json.dumps({'phase': 'paired_audio', 'clips_done': number + 1, 'clips_total': len(rows),
                                  'new_inference_calls': self.ev.calls, 'cache_hits': self.ev.hit_count}, ensure_ascii=False), flush=True)
                self.summarize()

    def contexts(self):
        rows = read(CURRICULUM / 'selected-context-audio.json')
        for number, row in enumerate(rows):
            identity = f'context-{number:03d}'
            relevant = [s for s in self.stages if set(s['targets']) & set(row['curriculum_words'])]
            if all(identity + ':' + s['id'] in self.done for s in relevant): continue
            signal = np.concatenate([np.zeros(6400), waveform(row), np.zeros(41600)])
            frames, audio_path = self.frames(identity, signal)
            for stage in relevant:
                self.compare(identity + ':' + stage['id'], 'natural_context', 'sentence', stage['targets'], frames, True,
                             reference=row['text'], record_id=row['record_id'], source_path=row['audio_path'],
                             age=row['age'], speaker_key=row['speaker_key'], audio_path=audio_path, stage=stage['id'],
                             note='Reference contains the requested expression, sometimes with a particle; no new listening audit.')
            if (number + 1) % 20 == 0:
                print(json.dumps({'phase': 'natural_context', 'clips_done': number + 1, 'clips_total': len(rows),
                                  'new_inference_calls': self.ev.calls}, ensure_ascii=False), flush=True)
                self.summarize()

    def select_distractors(self):
        path = OUT / 'distractor-sources.json'
        if path.exists(): return read(path)
        words = {spelling_key(w) for s in self.stages for w in s['targets']}
        excluded = {r['record_id'] for name in ('selected-audio.json', 'selected-context-audio.json') for r in read(CURRICULUM / name)}
        excluded.update(r['record_id'] for r in read(WEB / 'performance-20261030-025230/selected-word-records.json'))
        pool = defaultdict(list)
        with gzip.open(ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세/manifests-20261027/dev.jsonl.gz', 'rt', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                text = spelling_key(row['text'])
                if row['record_id'] in excluded or not 1.5 <= row['seconds'] <= 6 or len(text) < 5: continue
                if any(word in text for word in words): continue
                key = hashlib.sha256((str(SEED) + row['record_id']).encode()).hexdigest()
                pool[row['age']].append((key, row))
        chosen = []
        for age in (3, 4, 5, 6):
            candidates = sorted(pool[age], key=lambda pair: pair[0])
            seen, selected = set(), []
            for _key, row in candidates:
                if row['speaker_key'] not in seen:
                    seen.add(row['speaker_key']); selected.append(row)
                    if len(selected) == 10: break
            selected_ids = {r['record_id'] for r in selected}
            selected += [row for _key, row in candidates if row['record_id'] not in selected_ids][:10 - len(selected)]
            assert len(selected) == 10
            chosen.extend(selected)
        write(path, chosen)
        return chosen

    def negatives(self):
        rows = self.select_distractors()
        for number, row in enumerate(rows):
            identity = f'distractor-{number:03d}'
            if all(identity + ':' + s['id'] in self.done for s in self.stages): continue
            frames, audio_path = self.frames(identity, np.concatenate([np.zeros(6400), waveform(row), np.zeros(41600)]))
            for stage in self.stages:
                self.compare(identity + ':' + stage['id'], 'new_distractor', 'unrelated_sentence', stage['targets'], frames, False,
                             reference=row['text'], record_id=row['record_id'], age=row['age'], speaker_key=row['speaker_key'],
                             source_path=row['audio_path'], audio_path=audio_path, stage=stage['id'])
            if (number + 1) % 10 == 0:
                print(json.dumps({'phase': 'distractors', 'clips_done': number + 1, 'clips_total': 40}, ensure_ascii=False), flush=True)
                self.summarize()
        rng = np.random.default_rng(SEED)
        count = RATE * 60
        parts = [part for row in rows for part in (waveform(row), np.zeros(1600))]
        speech = np.concatenate(parts)[:count]
        noise = rng.normal(0, .025, count)
        impulses = .008 * np.sin(2 * np.pi * 60 * np.arange(count) / RATE)
        for start in range(0, count - 160, 8000): impulses[start:start + 160] += rng.normal(0, .09, 160)
        for condition, signal in [('silence_60s', np.zeros(count)), ('white_noise_60s', noise),
                                  ('hum_clicks_60s', impulses), ('unrelated_speech_60s', speech)]:
            identity = 'long-' + condition
            if all(identity + ':' + s['id'] in self.done for s in self.stages): continue
            frames, audio_path = self.frames(identity, signal)
            for stage in self.stages:
                self.compare(identity + ':' + stage['id'], 'long_negative', condition, stage['targets'], frames, False,
                             audio_path=audio_path, stage=stage['id'], note='Synthetic sounds or concatenated speech, 60 seconds per stream.')
            print(json.dumps({'phase': 'long_negative', 'condition': condition, 'new_inference_calls': self.ev.calls}, ensure_ascii=False), flush=True)
            self.summarize()

    def summarize(self):
        totals = defaultdict(lambda: Counter())
        failures = []
        for r in self.results:
            count = totals[r['phase'] + '/' + r['condition']]
            count['trials'] += 1
            for label in ('before', 'after'):
                count[label + '_accepted'] += r[label]['accepted']
                count[label + '_premature'] += r[label].get('before_target_started', False)
            count['exact_word_appeared'] += r['exact_word_checks'] > 0
            count['exact_word_at_least_twice'] += r['exact_word_checks'] >= 2
            count['after_missed_with_exact_word'] += r['positive'] and not r['after']['accepted'] and r['exact_word_checks'] > 0
            count['after_missed_with_two_exact_checks'] += r['positive'] and not r['after']['accepted'] and r['exact_word_checks'] >= 2
            if (r['positive'] and not r['after']['accepted'] and r['exact_word_checks'] > 0) or (not r['positive'] and r['after']['accepted']):
                failures.append({k: v for k, v in r.items() if k not in ('before', 'after')} |
                                {'decision': {k: v for k, v in r['after'].items() if k != 'trace'}})
        write(OUT / 'summary.json', {'completed_trials': len(self.results), 'totals': totals,
                                    'new_inference_calls_this_run': self.ev.calls if self.ev else 0,
                                    'limits': 'Controlled conditions reuse recordings. Labels are dataset/intended-word labels, not a new listening audit. Ideal text trials are not audio accuracy. Deliberate negatives are not natural-use false-acceptance rates. These data are development data, not an untouched test set.'})
        write(OUT / 'failure-index.json', failures)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(__file__).with_name('answer_policy.py'), Path(__file__).with_name('answer_families.json'),
             Path(__file__).with_name('word_game.py'), Path(__file__).with_name('verify_postprocessing.py')]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    pause = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/main/pause.request'
    pause_hash = hashlib.sha256(pause.read_bytes()).hexdigest()
    plan = {'created_at': datetime.now().astimezone().isoformat(), 'seed': SEED, 'conditions': CONDITIONS,
            'source_hashes': hashes, 'pause_hash': pause_hash, 'threshold': .85, 'confirmations': 2,
            'isolated_recordings': 58, 'context_recordings': 140, 'new_distractor_recordings': 40,
            'long_negative_streams': 4, 'no_product_or_training_changes': True}
    if (OUT / 'plan.json').exists():
        previous = read(OUT / 'plan.json')
        assert previous['source_hashes'] == hashes and previous['pause_hash'] == pause_hash
    else: write(OUT / 'plan.json', plan)
    audit = Audit()
    audit.ideal()
    audit.setup_audio()
    audit.paired_audio()
    audit.contexts()
    audit.negatives()
    audit.summarize()
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
    assert hashlib.sha256(pause.read_bytes()).hexdigest() == pause_hash
    write(OUT / 'complete.json', {'finished_at': datetime.now().astimezone().isoformat(), 'model': audit.ev.status,
                                'trials': len(audit.results), 'source_unchanged': True, 'training_pause_unchanged': True})
    audit.ev.client.close()
    print(json.dumps({'phase': 'complete', 'trials': len(audit.results), 'output': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
