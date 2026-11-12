"""Replay saved decisions and controlled audio pairs without adding user history."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from diagnose_decisions import decide
from evaluate_performance import ROOT, Evaluator, encoded, read, waveform, write
from word_game import WordGames, StartWordGame

OUTPUT = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/context-fix-20261030'
SOURCE = OUTPUT.parent / 'performance-20261030-025230'


def main():
    games = WordGames(SimpleNamespace(ready=True), None, None)
    cached = {}

    def rescore(frames, targets):
        candidates = [(word, games.g2p.to_ipa(word)) for word in targets]
        result = []
        for frame in frames:
            key = (frame['text'], tuple(targets), frame['voice_fraction'] >= .04)
            if key not in cached:
                cached[key] = games.score_candidates(frame['text'], candidates, frame['voice_fraction'])
            scores = cached[key]
            result.append({**frame, 'scores': {item['target']: item['similarity'] for item in scores},
                           'matched_texts': {item['target']: item['matched_text'] for item in scores}})
        return result

    with gzip.open(SOURCE / 'decision-diagnosis-20261030-032424/scored-trials.json.gz', 'rt', encoding='utf-8') as stream:
        trials = json.load(stream)
    totals, changes = defaultdict(lambda: defaultdict(int)), []
    for trial in trials:
        before = trial['outcomes']['85_x2']
        after = decide(rescore(trial['frames'], trial['targets']), .85)
        counts = totals[trial['scenario']]
        counts['trials'] += 1
        for name, decision in [('before', before), ('after', after)]:
            counts[name + '_accepted'] += decision['state'] == 'accepted'
            counts[name + '_correct'] += decision['target'] == trial['reference']
            counts[name + '_wrong'] += decision['state'] == 'accepted' and decision['target'] != trial['reference']
        if (before['state'], before['target']) != (after['state'], after['target']):
            changes.append({key: trial[key] for key in ('record_id', 'reference', 'scenario', 'targets')} |
                           {'before': before, 'after': after})
    write(OUTPUT / 'regression.json', {'trials': len(trials), 'totals': totals, 'changes': changes,
                                     'note': 'Saved ASR replay, not new audio inference or independent validation.'})
    print(json.dumps({'regression': totals}, ensure_ascii=False), flush=True)

    trace_results = []
    for record_id in ('4cec6998f8ea5b4c8df4b33259cb432a', '130b0218bcd1583ba1aaf720377d7998',
                      '49588409177a50329093c093260dea5c', 'cf094b976f4a5dbe96d08b0a6342dbbb'):
        record = read(ROOT / '03_Recordings/원문자모_성능테스트' / (record_id + '.json'))
        frames = [{**frame, 'seconds': frame['elapsed_seconds'],
                   'scores': {item['target']: item['similarity'] for item in frame['candidate_scores']}}
                  for frame in record['game']['trace']]
        after_frames = rescore(frames, record['game']['settings']['targets'])
        trace_results.append({'record_id': record_id, 'before': decide(frames, .85),
                              'after': decide(after_frames, .85), 'frames': after_frames})
    write(OUTPUT / 'user-trace-replay.json', trace_results)
    print(json.dumps({'user_replays': [{k: v for k, v in row.items() if k != 'frames'} for row in trace_results]}, ensure_ascii=False), flush=True)

    # Import the pre-change policy with its original source location so its
    # project-relative confusion-matrix path remains correct.
    namespace = {'__file__': str(Path(__file__).with_name('word_game.py')), '__name__': 'baseline_word_game'}
    exec(compile((OUTPUT / 'word_game_before.py').read_text(encoding='utf-8'), namespace['__file__'], 'exec'), namespace)
    ev = Evaluator(OUTPUT)
    before_games = namespace['WordGames'](SimpleNamespace(ready=True), lambda wave: ev.pending,
                                         lambda state, frame: {})
    rows = read(SOURCE / 'selected-word-records.json')
    prefix_row = next(row for row in read(SOURCE / 'asr-results.json') if row['text'].strip() == '고양이' and row['exact'])
    prefix = waveform(prefix_row)
    targets = [(row['text'].strip(), waveform(row), row['audio_path']) for row in rows
               if row['text'].strip() in ('출발', '흔들어')][:3]
    user_path = ROOT / '03_Recordings/원문자모_성능테스트/8b98bea5753f57a4946c8cf29d60a52d.wav'
    user_wave, rate = sf.read(user_path, dtype='float32')
    assert rate == 16000 and user_wave.ndim == 1
    targets.append(('들어야지', user_wave, str(user_path.relative_to(ROOT))))
    rng = np.random.default_rng(42)
    noises = rng.normal(0, .02, 12800).astype(np.float32)
    results = []
    for number, (target, audio, source) in enumerate(targets):
        for condition, lead in [('alone', np.zeros(0)), ('speech', prefix),
                                ('speech_with_gap', np.concatenate([prefix, np.zeros(3200)])),
                                ('noise', noises), ('prefix_only', prefix)]:
            signal = np.concatenate([np.zeros(6400), lead, audio if condition != 'prefix_only' else np.zeros(0), np.zeros(48000)]).astype(np.float32)
            sf.write(OUTPUT / f'pair-{number}-{condition}.wav', signal, 16000, subtype='PCM_16')
            old = before_games.start(namespace['StartWordGame'](reference=target))['session_id']
            new = ev.games.start(StartWordGame(reference=target))['session_id']
            timelines = []
            old_answer, new_answer = None, None
            for index in range(1, min(20, len(signal) // 8000) + 1):
                end = index * 8000
                wave = np.pad(signal[max(0, end - 40000):end], (max(0, 40000 - end), 0))
                ev.pending = ev.infer(wave)
                blob = encoded(wave)
                old_answer = before_games.frame(old, index, index * .5, wave, blob, 16000)
                new_answer = ev.games.frame(new, index, index * .5, wave, blob, 16000)
                timelines.append({'seconds': index * .5, 'prediction': ev.pending,
                                  'before': old_answer['state'], 'after': new_answer['state'],
                                  'after_score': new_answer['similarity'], 'matched_text': new_answer['matched_text']})
            results.append({'target': target, 'source': source, 'prefix_source': prefix_row['audio_path'],
                            'condition': condition, 'before': old_answer['state'], 'after': new_answer['state'],
                            'frames': timelines})
            print(json.dumps({k: v for k, v in results[-1].items() if k not in ('frames', 'source', 'prefix_source')}, ensure_ascii=False), flush=True)
    write(OUTPUT / 'audio-pairs.json', {'model': ev.status, 'inference_calls': ev.calls, 'results': results,
          'note': 'Controlled concatenations of existing recordings; synthetic white noise; not live microphone accuracy.'})
    ev.client.close()


if __name__ == '__main__':
    main()
