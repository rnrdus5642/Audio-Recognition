"""Compare v3/v4 decisions on unchanged ASR recordings, including wrong actions."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import unicodedata

import numpy as np

from evaluate_performance import ROOT, read, write
from word_game import WordGames, StartWordGame

OUTPUT = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/postprocessing-20261030'


class Replay:
    def __init__(self, game_class, param_class):
        self.params = param_class
        self.wave = np.full(40000, .03, dtype=np.float32)
        self.games = game_class(SimpleNamespace(ready=True), lambda wave: self.prediction, lambda state, frame: {})
        self.games.voice_fraction = lambda wave: self.fraction
        original = self.games.score_candidates
        cache = {}

        def score(text, candidates, fraction):
            key = (text, tuple(word for word, _ in candidates), fraction >= .04)
            if key not in cache:
                cache[key] = original(text, candidates, fraction)
            return cache[key]
        self.games.score_candidates = score

    def run(self, targets, frames):
        start = self.games.start(self.params(targets=targets))
        session = start['session_id']
        for index, frame in enumerate(frames, 1):
            text = frame['text']
            self.prediction = {'text': text, 'uncomposed_jamo': sum(0x1100 <= ord(ch) <= 0x11ff for ch in unicodedata.normalize('NFC', text))}
            self.fraction = frame['voice_fraction']
            answer = self.games.frame(session, index, frame.get('seconds', frame.get('elapsed_seconds')),
                                      self.wave, b'replay', 16000)
            if answer['state'] == 'accepted':
                break
        else:
            answer = self.games.finish(session, 'stopped')
        trace = self.games.get(session)['trace']
        return {'accepted': answer['state'] == 'accepted', 'target': answer['matched_target'],
                'seconds': answer['elapsed_seconds'], 'text': answer['prediction']['text'],
                'trace': trace}


def main():
    namespace = {'__file__': str(Path(__file__).with_name('word_game.py')), '__name__': 'baseline_word_game'}
    exec(compile((OUTPUT / 'before/word_game.py').read_text(encoding='utf-8'), namespace['__file__'], 'exec'), namespace)
    before = Replay(namespace['WordGames'], namespace['StartWordGame'])
    after = Replay(WordGames, StartWordGame)
    totals, changes = defaultdict(lambda: defaultdict(int)), []

    def compare(group, identity, targets, frames, positive, reference=None):
        old, new = before.run(targets, frames), after.run(targets, frames)
        count = totals[group]
        count['trials'] += 1
        for label, result in [('before', old), ('after', new)]:
            count[label + '_accepted'] += result['accepted']
            count[label + '_correct'] += result['accepted'] and positive and (reference is None or result['target'] == reference)
            count[label + '_wrong'] += result['accepted'] and (not positive or (reference is not None and result['target'] != reference))
        if (old['accepted'], old['target']) != (new['accepted'], new['target']):
            changes.append({'group': group, 'identity': identity, 'targets': targets, 'positive': positive,
                            'reference': reference, 'before': old, 'after': new})
        return old, new

    source = OUTPUT.parent / 'performance-20261030-025230/decision-diagnosis-20261030-032424/scored-trials.json.gz'
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for trial in json.load(stream):
            old, _new = compare(trial['scenario'], trial['record_id'], trial['targets'], trial['frames'],
                                trial['reference'] in trial['targets'], trial['reference'])
            baseline = trial['outcomes']['85_x2']
            assert old['accepted'] == (baseline['state'] == 'accepted') and old['target'] == baseline['target']
    print(json.dumps({'general': totals}, ensure_ascii=False), flush=True)

    source = OUTPUT.parent / 'curriculum-20261030-033454/audio-evaluation/stage-trials.json.gz'
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        stages = json.load(stream)
    full = defaultdict(dict)
    for trial in stages:
        for frame in trial['current']['trace']:
            time = frame['elapsed_seconds']
            old = full[trial['record_id']].get(time)
            if old:
                assert old['text'] == frame['text'] and old['voice_fraction'] == frame['voice_fraction']
            full[trial['record_id']][time] = frame
    for trial in stages:
        frames = [frame for _time, frame in sorted(full[trial['record_id']].items())]
        assert len(frames) == 20
        compare('curriculum_positive' if trial['is_positive'] else 'curriculum_negative',
                trial['record_id'] + '/' + trial['stage'], trial['current']['settings']['targets'], frames,
                trial['is_positive'])

    user = []
    for record_id in ('130b0218bcd1583ba1aaf720377d7998', '35ec27b8ae315273aaefeb9af8001ddb',
                      '4cec6998f8ea5b4c8df4b33259cb432a', '7bee3e531c205424a8996a56d459ce80'):
        record = read(ROOT / '03_Recordings/원문자모_성능테스트' / (record_id + '.json'))
        game = record['game']
        user.append({'record_id': record_id, 'before': before.run(game['settings']['targets'], game['trace']),
                     'after': after.run(game['settings']['targets'], game['trace'])})
    write(OUTPUT / 'user-replay.json', user)
    write(OUTPUT / 'comparison.json', {'totals': totals, 'changes': changes,
          'note': 'Same saved ASR outputs; no fresh microphone data. Many tests share audio and are not independent. Negative sets are intentional confusions, not natural-use false-acceptance estimates.'})
    print(json.dumps({'totals': totals, 'changed': len(changes),
                      'user': [{'id': r['record_id'], 'before': r['before']['accepted'], 'after': r['after']['accepted']} for r in user]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
