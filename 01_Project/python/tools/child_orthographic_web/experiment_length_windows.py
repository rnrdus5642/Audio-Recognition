"""Offline length-window experiment on saved ASR outputs, without deployment.

All variants use the real WordGames confirmation, thresholds, ambiguity, energy
gate and action guard. Only score_candidates' eligible text spans change.
No model loads, audio inference, network calls or user-history writes occur.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import unicodedata

from answer_policy import comparison_spans
from verify_postprocessing import Replay
from word_game import WordGames, StartWordGame

ROOT = Path(__file__).resolve().parents[4]
WEB = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test'
STRESS = WEB / 'stress-20261030-044247'
GENERAL = WEB / 'performance-20261030-025230/decision-diagnosis-20261030-032424/scored-trials.json.gz'
CURRICULUM = WEB / 'curriculum-20261030-033454/audio-evaluation/stage-trials.json.gz'
RATIOS = (12, 13, 14, 15)  # Integer tenths avoid floating-point rounding.
CONFIGS = {
    'current_v4': {'position': 'suffix', 'unit': None, 'ratio': None},
    'suffix_phoneme_1.2': {'position': 'suffix', 'unit': 'phoneme', 'ratio': 12},
    'suffix_phoneme_1.5': {'position': 'suffix', 'unit': 'phoneme', 'ratio': 15},
    'whole_any_unlimited': {'position': 'any', 'unit': None, 'ratio': None},
    **{f'whole_any_{unit}_{ratio / 10:.1f}': {'position': 'any', 'unit': unit, 'ratio': ratio}
       for unit in ('phoneme', 'syllable') for ratio in RATIOS},
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frames_only(frames):
    return [{'seconds': f.get('seconds', f.get('elapsed_seconds')), 'text': f['text'],
             'voice_fraction': f['voice_fraction']} for f in frames]


def load_trials():
    audio = {}
    with (STRESS / 'audio-frames.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            audio[item['audio_path']] = frames_only(item['frames'])
    trials = []
    with (STRESS / 'trials.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            frames = audio[row['audio_path']] if 'audio_path' in row else frames_only(row['after']['trace'])
            meta = {k: v for k, v in row.items() if k not in ('before', 'after')}
            meta.update(dataset='stress', frames=frames,
                        expected_v4={k: row['after'][k] for k in ('accepted', 'target', 'seconds')})
            trials.append(meta)
    with gzip.open(GENERAL, 'rt', encoding='utf-8') as stream:
        for row in json.load(stream):
            trials.append({'id': 'general:' + row['record_id'] + ':' + row['scenario'],
                           'dataset': 'general', 'phase': 'general', 'condition': row['scenario'],
                           'targets': row['targets'], 'positive': row['reference'] in row['targets'],
                           'reference': row['reference'], 'record_id': row['record_id'],
                           'age': row['age'], 'frames': frames_only(row['frames'])})
    with gzip.open(CURRICULUM, 'rt', encoding='utf-8') as stream:
        curriculum = json.load(stream)
    complete = defaultdict(dict)
    for row in curriculum:
        for f in frames_only(row['current']['trace']):
            previous = complete[row['record_id']].get(f['seconds'])
            assert previous is None or previous == f, 'Inconsistent cached ASR frame'
            complete[row['record_id']][f['seconds']] = f
    for row in curriculum:
        frames = [f for _, f in sorted(complete[row['record_id']].items())]
        assert len(frames) == 20
        trials.append({'id': 'curriculum:' + row['record_id'] + ':' + row['stage'],
                       'dataset': 'curriculum', 'phase': 'curriculum',
                       'condition': 'positive' if row['is_positive'] else 'negative',
                       'targets': row['current']['settings']['targets'], 'positive': row['is_positive'],
                       'reference': row['curriculum_word'], 'record_id': row['record_id'],
                       'age': row['age'], 'stage': row['stage'], 'frames': frames})
    for identity in ('130b0218bcd1583ba1aaf720377d7998', '35ec27b8ae315273aaefeb9af8001ddb',
                     '4cec6998f8ea5b4c8df4b33259cb432a', '7bee3e531c205424a8996a56d459ce80'):
        path = ROOT / '03_Recordings/원문자모_성능테스트' / (identity + '.json')
        row = read(path)['game']
        trials.append({'id': 'user:' + identity, 'dataset': 'user', 'phase': 'user', 'condition': 'saved_trace',
                       'targets': row['settings']['targets'], 'positive': None,
                       'record_id': identity, 'frames': frames_only(row['trace'])})
    assert len({r['id'] for r in trials}) == len(trials)
    return trials


class ScoreBank:
    def __init__(self, game):
        self.game = game
        original = game.g2p.to_ipa
        self.ipa = lru_cache(maxsize=None)(lambda word: tuple(original(word)))

    @lru_cache(maxsize=None)
    def word_spans(self, text, targets):
        """All whitespace-bounded, contiguous Korean phrases, rightmost first.

        No word-internal slicing; a malformed token is not silently repaired.
        Repetition normalization remains exactly the current production rule.
        """
        words = unicodedata.normalize('NFC', text).split()
        seen, out = set(), []
        for end in range(len(words), 0, -1):
            for start in range(end - 1, -1, -1):
                if not re.fullmatch('[가-힣]+', words[start]):
                    break
                raw, joined = ' '.join(words[start:end]), ''.join(words[start:end])
                variations = [(raw, joined, None)]
                for count in (2, 3, 4):
                    width, remainder = divmod(len(joined), count)
                    root = joined[:width]
                    if not remainder and width >= 2 and root in targets and root * count == joined:
                        variations.append((raw, root, 'repeated_word'))
                        break
                for span in variations:
                    if span not in seen:
                        seen.add(span)
                        out.append(span)
        return tuple(out)

    @lru_cache(maxsize=None)
    def distance_score(self, heard, target):
        return self.game.matcher.score_against(self.ipa(heard), self.ipa(target))[1]

    @lru_cache(maxsize=None)
    def scores(self, text, targets, voiced):
        results = {key: [] for key in CONFIGS if key != 'current_v4'}
        suffix = tuple(comparison_spans(text, set(targets)))
        any_spans = self.word_spans(text, targets)
        for target in targets:
            target_lengths = {'syllable': len(target), 'phoneme': len(self.ipa(target))}
            # Each actual phrase/target distance is calculated once and reused.
            pools = {}
            for position, spans in [('suffix', suffix), ('any', any_spans)]:
                pool = []
                if voiced:
                    for raw, compared, reason in spans:
                        pool.append((self.distance_score(compared, target), raw, compared, reason,
                                     {'syllable': len(compared), 'phoneme': len(self.ipa(compared))}))
                pools[position] = pool
            for key, cfg in CONFIGS.items():
                if key == 'current_v4':
                    continue
                best, matched, compared, normalization, compared_length = 0., None, None, None, None
                for score, raw, normalized, reason, lengths in pools[cfg['position']]:
                    if cfg['ratio'] is not None and lengths[cfg['unit']] * 10 > target_lengths[cfg['unit']] * cfg['ratio']:
                        continue
                    if score > best:
                        best, matched, compared, normalization = score, raw, normalized, reason
                        compared_length = lengths.get(cfg['unit'])
                results[key].append({'target': target, 'similarity': best, 'matched_text': matched,
                                     'comparison_text': compared, 'normalization': normalization,
                                     'blocked_by': self.game.action_guard.conflict(compared or '', target),
                                     'length_unit': cfg['unit'], 'compared_length': compared_length,
                                     'target_length': target_lengths.get(cfg['unit']),
                                     'max_length': (target_lengths[cfg['unit']] * cfg['ratio'] // 10
                                                    if cfg['ratio'] is not None else None)})
        for scores in results.values():
            scores.sort(key=lambda item: item['similarity'], reverse=True)
        return results


def brief(result):
    frame = result['trace'][-1]
    return {k: result[k] for k in ('accepted', 'target', 'seconds', 'text')} | {
        k: frame.get(k) for k in ('candidate', 'similarity', 'comparison_text', 'matched_text',
                                 'normalization', 'blocked_by', 'decision_reason', 'streak')}


def changed(a, b):
    return (a['accepted'], a['target'], a['seconds']) != (b['accepted'], b['target'], b['seconds'])


def summarize(rows):
    grouped = defaultdict(lambda: {k: Counter() for k in CONFIGS})
    delays = defaultdict(lambda: defaultdict(list))
    for row in rows:
        keys = [row['phase'] + '/' + row['condition']]
        if row['phase'] in ('paired_single', 'paired_stage'):
            keys.append(row['phase'] + '/ALL')
        for key in keys:
            baseline = row['results']['current_v4']
            for variant, value in row['results'].items():
                n = grouped[key][variant]
                n['trials'] += 1
                n['accepted'] += int(value['accepted'])
                n['gained'] += int(not baseline['accepted'] and value['accepted'])
                n['lost'] += int(baseline['accepted'] and not value['accepted'])
                n['changed_target'] += int(baseline['accepted'] and value['accepted'] and baseline['target'] != value['target'])
                n['premature'] += int(value['accepted'] and value['seconds'] <= row.get('target_start', -1))
                n['reference_exact'] += int(value['accepted'] and value['target'] == row.get('reference'))
                if value['accepted'] and baseline['accepted']:
                    delays[key][variant].append(value['seconds'] - baseline['seconds'])
    output = {}
    for group, variants in grouped.items():
        output[group] = {}
        for variant, counts in variants.items():
            output[group][variant] = dict(counts) | {
                'median_confirmation_change_seconds': statistics.median(delays[group][variant])
                if delays[group][variant] else None}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    out = args.output or WEB / ('length-window-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    out = out.resolve()
    if not out.is_relative_to(WEB.resolve()) or out.drive.upper() != 'C:':
        raise ValueError('Output must remain in this C-drive experiment folder')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    protected = [Path(__file__).parent / n for n in ['word_game.py', 'answer_policy.py', 'answer_families.json',
                                                     'static/game.js', 'server.py']]
    protected += [ROOT / '01_Project/python/runtime/matching/matcher.py',
                  ROOT / '01_Project/shared/confusion_matrices/ko_child_v2.json',
                  ROOT / '01_Project/python/tools/child_orthographic_release/inference.py',
                  ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/main/pause.request']
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    source_paths = [STRESS / 'audio-frames.jsonl', STRESS / 'trials.jsonl', GENERAL, CURRICULUM, Path(__file__)]
    trials = load_trials()
    if args.limit:
        trials = trials[:args.limit]
    write(out / 'plan.json', {'started_at': datetime.now().astimezone().isoformat(),
                             'configs': CONFIGS, 'trials': len(trials), 'threshold': .85, 'confirmation': 2,
                             'max_length': 'floor(target_length * multiplier); no new lower bound',
                             'scope': 'All complete whitespace-bounded phrases, after existing repetition normalization',
                             'model': read(STRESS / 'complete.json')['model'],
                             'protected_hashes': original_hashes,
                             'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
                             'limits': ['Saved ASR replay, no fresh inference or microphone recordings.',
                                        'Reused recordings and stages are not independent trials.',
                                        'Development data, not an untouched test set.',
                                        'No thresholds, action dictionary, model or production policy changes.']})
    print(json.dumps({'phase': 'start', 'output': str(out), 'trials': len(trials), 'variants': len(CONFIGS)}), flush=True)
    replays = {key: Replay(WordGames, StartWordGame) for key in CONFIGS}
    bank = ScoreBank(replays['current_v4'].games)
    for key, replay in replays.items():
        replay.games.g2p.to_ipa = lambda text: list(bank.ipa(text))
        if key != 'current_v4':
            replay.games.score_candidates = lambda text, candidates, fraction, key=key: bank.scores(
                text, tuple(w for w, _ in candidates), fraction >= .04)[key]
    results, parity_count, changed_examples = [], 0, []
    with gzip.open(out / 'inputs.jsonl.gz', 'wt', encoding='utf-8') as inputs, \
            gzip.open(out / 'results.jsonl.gz', 'wt', encoding='utf-8') as saved, \
            gzip.open(out / 'changes.jsonl.gz', 'wt', encoding='utf-8') as changes:
        for index, row in enumerate(trials, 1):
            inputs.write(json.dumps(row, ensure_ascii=False) + '\n')
            meta = {k: v for k, v in row.items() if k not in ('frames', 'expected_v4')}
            runs = {key: replay.run(row['targets'], row['frames']) for key, replay in replays.items()}
            baseline = runs['current_v4']
            if 'expected_v4' in row:
                assert {k: baseline[k] for k in row['expected_v4']} == row['expected_v4'], row['id']
                parity_count += 1
            item = meta | {'results': {key: brief(value) for key, value in runs.items()}}
            results.append(item)
            saved.write(json.dumps(item, ensure_ascii=False) + '\n')
            changed_keys = [key for key in CONFIGS if changed(baseline, runs[key])]
            if changed_keys:
                detail = meta | {'variants_changed': changed_keys, 'frames': row['frames'],
                                 'results': {key: runs[key] for key in ['current_v4', *changed_keys]}}
                changes.write(json.dumps(detail, ensure_ascii=False) + '\n')
                if row['phase'] not in ('ideal_positive', 'ideal_negative'):
                    changed_examples.append(item)
            # No real session or history storage is involved; release replay state.
            for replay in replays.values():
                replay.games.sessions.clear()
            if index % 500 == 0 or index == len(trials):
                print(json.dumps({'phase': 'replay', 'done': index, 'total': len(trials),
                                  'elapsed_seconds': round(time.monotonic() - started, 1),
                                  'baseline_parity': parity_count,
                                  'unique_distances': bank.distance_score.cache_info().currsize}), flush=True)
    write(out / 'summary.json', {'trials': len(results), 'variants': len(CONFIGS),
                                'baseline_parity': parity_count, 'groups': summarize(results),
                                'phase_counts': dict(Counter(row['phase'] for row in trials))})
    write(out / 'changed-examples.json', changed_examples)
    final_hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert final_hashes == original_hashes, 'Protected production/training files changed during experiment'
    write(out / 'complete.json', {'finished_at': datetime.now().astimezone().isoformat(),
                                 'duration_seconds': time.monotonic() - started,
                                 'trials': len(results), 'policy_runs': len(results) * len(CONFIGS),
                                 'baseline_parity': parity_count, 'production_and_pause_unchanged': True,
                                 'distance_cache': bank.distance_score.cache_info()._asdict()})
    print(json.dumps({'phase': 'complete', 'output': str(out), 'trials': len(results)}), flush=True)


if __name__ == '__main__':
    main()
