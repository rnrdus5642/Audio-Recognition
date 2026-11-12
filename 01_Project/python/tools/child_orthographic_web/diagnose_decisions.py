"""Counterfactual decision audit; never modifies the deployed game or training.

Use the previously measured, target-independent ASR predictions. Complete missing
tail windows through the same deployed API, then recompute actual matcher scores.
Validate the 85-point / two-consecutive baseline against every saved decision.
One-frame and two-of-three policies are analysis-only (the API requires >=2).
"""
import argparse
from collections import Counter
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np
import soundfile as sf

from evaluate_performance import Evaluator, ROOT, RATE, SEED, encoded, safe, waveform, write


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normal(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def decide(frames, threshold, count=2, mode='consecutive'):
    streak, last_target, last_time, recent = 0, None, 0., []
    for frame in frames:
        scores = frame['scores']
        top = max(scores.values())
        leaders = [word for word, score in scores.items() if math.isclose(score, top, rel_tol=0, abs_tol=1e-9)]
        target = leaders[0] if top >= threshold and len(leaders) == 1 else None
        time = frame['seconds']
        continuous = target == last_target and time - last_time <= 1.25
        streak = (streak + 1 if continuous else 1) if target else 0
        if time - last_time > 1.25:
            recent = []
        recent = (recent + [target])[-3:]
        passed = streak >= count
        if mode == 'two_of_three':
            passed = bool(target and recent.count(target) >= 2)
        elif mode == 'exact_once_or_two':
            passed = passed or bool(target and normal(frame['text']) == normal(target))
        if passed:
            return {'state': 'accepted', 'target': target, 'seconds': time, 'text': frame['text'], 'score': top}
        last_target, last_time = target, time
    return {'state': 'retry', 'target': None, 'seconds': 10., 'text': '', 'score': None}


def metrics(trials, policy):
    outcomes = [(row, row['outcomes'][policy]) for row in trials]
    accepted = [(row, result) for row, result in outcomes if result['state'] == 'accepted']
    correct = sum(row['reference'] == result['target'] for row, result in accepted)
    return {'clips': len(trials), 'accepted': len(accepted), 'correct': correct,
            'wrong_candidate': len(accepted) - correct, 'retry': len(trials) - len(accepted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = safe(args.source)
    previous = load(source / 'summary.json')
    audit = load(source / 'audit.json')
    policy_path = Path(__file__).with_name('word_game.py')
    assert hashlib.sha256(policy_path.read_bytes()).hexdigest() == audit['source_sha256'][str(policy_path.relative_to(ROOT))]
    rows = [json.loads(line) for line in (source / 'game-results.jsonl').read_text(encoding='utf-8').splitlines()]
    asr = {(r['group'], r['record_id']): r for r in load(source / 'asr-results.json')}
    extras = load(source / 'similar-target-results.json')
    output = source / datetime.now().strftime('decision-diagnosis-%Y%m%d-%H%M%S')
    ev = Evaluator(output)
    assert ev.status['model_id'] == previous['plan']['model']['model_id']
    for line in (source / 'predictions.jsonl').read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        assert item['model_id'] == ev.status['model_id']
        ev.cache[item['audio_hash']] = item['prediction']
    policies = {f'{int(threshold*100)}_x{count}': {'threshold': threshold, 'count': count, 'mode': 'consecutive'}
                for threshold in (.75, .80, .85, .90, .95, 1.) for count in (1, 2, 3)}
    policies['85_two_of_three'] = {'threshold': .85, 'count': 2, 'mode': 'two_of_three'}
    policies['85_exact_once_or_two'] = {'threshold': .85, 'count': 2, 'mode': 'exact_once_or_two'}
    text_cache, score_cache, trials = {}, {}, []
    validated = 0

    def scored_frames(frames, targets):
        result = []
        for frame in frames:
            prediction = frame['prediction']
            text = prediction['text']
            clean = ''.join(re.findall(r'[가-힣\s]', unicodedata.normalize('NFC', text))).strip()
            valid = bool(clean) and not prediction.get('uncomposed_jamo', 0) and frame['voice_fraction'] >= .04
            scores = {}
            for target in targets:
                if not valid:
                    scores[target] = 0.
                    continue
                key = (clean, target)
                if key not in score_cache:
                    for word in (clean, target):
                        if word not in text_cache:
                            text_cache[word] = ev.games.g2p.to_ipa(word)
                    score_cache[key] = ev.games.matcher.score_against(text_cache[clean], text_cache[target])[1]
                scores[target] = score_cache[key]
            result.append({'seconds': frame['seconds'], 'text': text, 'scores': scores,
                           'voice_fraction': frame['voice_fraction'], 'uncomposed_jamo': prediction.get('uncomposed_jamo', 0)})
        return result

    def append_trial(row, name, targets, frames, saved):
        nonlocal validated
        scored = scored_frames(frames, targets)
        outcomes = {name: decide(scored, **params) for name, params in policies.items()}
        baseline = outcomes['85_x2']
        assert baseline['state'] == saved['state'], (row['record_id'], name, baseline, saved['state'])
        assert baseline['target'] == saved['matched_target'], (row['record_id'], name, baseline, saved['matched_target'])
        if baseline['state'] == 'accepted':
            assert abs(baseline['seconds'] - saved['elapsed_seconds']) < 1e-9
        # Also validate numeric scores against each previously observed frame.
        for old in saved.get('trace', []):
            now = scored[old['frame_index'] - 1]
            for score in old['candidate_scores']:
                assert abs(now['scores'][score['target']] - score['similarity']) < 1e-9
        validated += 1
        trial = {'record_id': row['record_id'], 'group': row['group'], 'age': row['age'],
                 'reference': row['text'], 'scenario': name, 'targets': targets,
                 'frames': scored, 'outcomes': outcomes}
        trials.append(trial)
        return trial

    print(json.dumps({'phase': 'start', 'output': str(output), 'clips': len(rows)}, ensure_ascii=False), flush=True)
    failures = []
    for index, row in enumerate(rows, 1):
        wave = waveform(row)
        signal = np.zeros(RATE * 10, dtype=np.float32)
        signal[6400:6400 + len(wave)] = wave
        frames = []
        for step in range(1, 21):
            end = step * 8000
            chunk = signal[max(0, end - 40000):end]
            chunk = np.pad(chunk, (40000 - len(chunk), 0))
            prediction = row['frames'][step - 1]['prediction'] if step <= len(row['frames']) else ev.infer(chunk)
            frames.append({'seconds': step * .5, 'prediction': prediction, 'voice_fraction': ev.games.voice_fraction(chunk)})
        for name, saved in row['trials'].items():
            trial = append_trial(row, name, saved['settings']['targets'], frames, saved)
            if name == 'single' and trial['outcomes']['85_x2']['state'] != 'accepted':
                high = [f for f in trial['frames'] if f['scores'][row['text']] >= .85]
                exact = [f for f in trial['frames'] if normal(f['text']) == normal(row['text'])]
                best = max(trial['frames'], key=lambda f: f['scores'][row['text']])
                failures.append({'record_id': row['record_id'], 'group': row['group'], 'age': row['age'],
                                 'reference': row['text'], 'audio_path': row['audio_path'], 'clip_seconds': row['seconds'],
                                 'category': 'passed_frames_not_consecutive' if high else 'never_reached_85',
                                 'full_clip_text': asr[row['group'], row['record_id']]['prediction']['text'],
                                 'full_clip_exact': asr[row['group'], row['record_id']]['exact'],
                                 'any_exact_window': bool(exact), 'best_text': best['text'],
                                 'best_score': best['scores'][row['text']], 'best_seconds': best['seconds'],
                                 'passing_frames': [{'seconds': f['seconds'], 'text': f['text'], 'score': f['scores'][row['text']]} for f in high],
                                 'outcomes': trial['outcomes'], 'frames': trial['frames']})
        for extra in extras:
            if extra['record_id'] == row['record_id']:
                append_trial(row, 'permitted_near_negative', [extra['target']], frames, extra['result'])
        if index % 100 == 0:
            print(json.dumps({'phase': 'replay', 'clips': index, 'validated_trials': validated}), flush=True)

    for item in load(source / 'noise-results.json'):
        level = {'silence': 0, 'white_noise_quiet': .005, 'white_noise': .03}[item['kind']]
        rng = np.random.default_rng(SEED + item['seed'])
        signal = rng.normal(0, level, RATE * 10).astype(np.float32) if level else np.zeros(RATE * 10, dtype=np.float32)
        frames = []
        for step in range(1, 21):
            end = step * 8000
            chunk = signal[max(0, end - 40000):end]
            chunk = np.pad(chunk, (40000 - len(chunk), 0))
            chunk, _ = sf.read(io.BytesIO(encoded(chunk)), dtype='float32')
            frames.append({'seconds': step*.5, 'prediction': ev.infer(chunk), 'voice_fraction': ev.games.voice_fraction(chunk)})
        row = {'record_id': f'{item["kind"]}:{item["seed"]}', 'text': '', 'group': 'synthetic_noise', 'age': 'unknown'}
        append_trial(row, 'noise', item['result']['settings']['targets'], frames, item['result'])

    policy_stats = {}
    groups = sorted({r['group'] for r in trials})
    scenarios = sorted({r['scenario'] for r in trials})
    for policy in policies:
        policy_stats[policy] = {scenario: metrics([r for r in trials if r['scenario'] == scenario], policy) for scenario in scenarios}
        policy_stats[policy]['by_group'] = {group: {scenario: metrics([r for r in trials if r['group'] == group and r['scenario'] == scenario], policy)
                                                 for scenario in scenarios} for group in groups}
        policy_stats[policy]['rescued_baseline_failures'] = sum(r['outcomes'][policy]['state'] == 'accepted' for r in failures)
        policy_stats[policy]['regressed_baseline_successes'] = sum(r['scenario'] == 'single' and r['outcomes']['85_x2']['state'] == 'accepted'
                                                                  and r['outcomes'][policy]['state'] != 'accepted' for r in trials)
    report = {'source': str(source.relative_to(ROOT)), 'model_id': ev.status['model_id'], 'policies': policies,
              'validated_saved_decisions': validated, 'additional_gpu_calls': ev.calls,
              'method': 'Same measured predictions and original 2.5s/0.5s audio windows; only missing tails inferred. Full 10s timelines. Baseline decisions and saved scores validated. No live settings changes.',
              'limits': ['Development data already used during model selection, not an untouched final test.',
                         'Local labels have incomplete listening review; ages unknown.',
                         'Negative scenarios are designed stress cases, not a natural-use false-positive rate.',
                         'Two-of-three and one-frame policies are offline counterfactuals, not deployed changes.',
                         '9 synthetic noise runs include 3 identical silence runs, not 9 independent real-room noises.'],
              'baseline_failure_categories': dict(Counter(r['category'] for r in failures)),
              'baseline_failures': len(failures), 'baseline_failures_full_clip_exact': sum(r['full_clip_exact'] for r in failures),
              'baseline_failures_any_exact_window': sum(r['any_exact_window'] for r in failures),
              'policy_stats': policy_stats,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), policy_path, ROOT / '01_Project/shared/confusion_matrices/ko_child_v2.json']}}
    write(output / 'summary.json', report)
    write(output / 'failure-details.json', failures)
    with gzip.open(output / 'scored-trials.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(trials, stream, ensure_ascii=False)
    print(json.dumps({'phase': 'complete', 'output': str(output), 'validated': validated, 'gpu_calls': ev.calls,
                      'categories': report['baseline_failure_categories'], 'failed_full_exact': report['baseline_failures_full_clip_exact'],
                      'failed_any_exact': report['baseline_failures_any_exact_window'],
                      'comparison': {p: {s: policy_stats[p][s] for s in ['single','multi10','wrong_near_target','permitted_near_negative','noise']} for p in ['85_x2','80_x2','85_x1','90_x2','85_two_of_three','85_exact_once_or_two']}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
