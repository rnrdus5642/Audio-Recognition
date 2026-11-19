"""Replay existing Unity fixture inputs through the current web single-word policy.

No ASR/model or microphone is used. Expected values come from WordGames.frame,
not a second implementation of the scoring/confirmation algorithm. Preserve the
previous fixture separately and pass its path with --baseline.

Requires the web v7 source snapshot recorded in the generated fixture. The Unity
package tests consume the committed fixture without running this exporter.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controls():
    examples = [
        (['실로폰채'], '양쪽 손으로 실로펀 체를 들어'),
        (['실로폰채'], 'ᅵᆯ로폰채 들어'),
        (['실로폰채'], 'ᅵᆯ로폰채를 두 손으로 들어'),
        (['실로폰채'], 'ᄉᆯ로폰채를 두 손으로 들어'),
        (['실로폰채'], 'ᄉ'),
        (['실로폰낫', '실로폰낮'], 'ᅵᆯ로폰낟'),
        (['사과'], '사과ᄀ 같아요'),
        (['책'], '책ᄉ'),
        (['책'], 'ᅢᆨ'),
        (['할머니'], '우리 할머니를 불러'),
        (['할아버지'], '할아버찌에게 인사'),
        (['실로폰채', '실로펀채'], '실로펀채를 들어'),
        (['실로폰채', '실로펀채를'], '실로펀채를 들어'),
        (['실로폰채'], '실로펀채는 아니에요'),
        (['실로폰채'], '실로펀채 말고 북채'),
        (['실로폰채'], '실로펀채장난감'),
        (['놓아야죠'], '넣어야죠'),
        (['덮어야죠'], '덥어야죠를'),
        (['실로폰채', '들어'], '실로폰채를 들어'),
        (['화분'], '화분을 골라요'),
        (['흙'], '이건 흙인 거 같아요'),
        (['씨앗'], '여기에 씨앗을 심자'),
        (['물'], '물감'),
        (['알'], '여기 알이 있어요'),
        (['먹이', '주자'], '먹이를 주자'),
        (['집'], '집에 보내요'),
        (['사과'], '이건 사과인거 같아요'),
        (['사과'], '사과주스'),
        (['사과'], '사과가 아니에요'),
        (['사과'], '사과 말고 배 주세요'),
        (['낫', '낮'], '낫'),
        (['실로폰채'], 'ᄉᆯ로폰채를'),
    ]
    for index, (targets, text) in enumerate(examples):
        yield {'id': f'v7-control:{index}', 'targets': targets, 'checks': [
            {'text': text, 'seconds': time, 'voice_fraction': .3} for time in [.5, 1.]]}
    for index, (text, threshold) in enumerate([
            ('양쪽 손으로 실로펀 체를 들어', .96),
            ('ᅵᆯ로폰채를 두 손으로 들어', .95),
            ('ᄉᆯ로폰채를 두 손으로 들어', .95)]):
        yield {'id': f'v7-threshold:{index}', 'targets': ['실로폰채'], 'threshold': threshold,
               'checks': [{'text': text, 'seconds': t, 'voice_fraction': .3} for t in [.5, 1.]]}
    for index, (texts, fractions, times) in enumerate([
            (['실로폰채', '', 'ᅵᆯ로폰채를'], [1, 0, 1], [.5, 1, 1.5]),
            (['실로폰채', 'ᄉ', 'ᄉ', '실로폰채'], [1]*4, [.5, 1, 1.5, 2]),
            (['ᅵᆯ로폰채를']*2, [0, 0], [.5, 1]),
            (['ᅵᆯ로폰채를']*3, [1]*3, [.5, 3, 3.5]),
            (['실로폰채', '다른말', '실로폰채'], [1]*3, [.5, 1, 1.5]),
    ]):
        yield {'id': f'v7-evidence:{index}', 'targets': ['실로폰채'], 'checks': [
            {'text': text, 'seconds': t, 'voice_fraction': f} for text, t, f in zip(texts, times, fractions)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    from word_game import WordGames, StartWordGame, POLICY, PROJECT
    if POLICY != 'orthographic-word-context-v7':
        parser.error('This exporter requires the recorded web v7 source snapshot; '
                     'see docs/UNITY_WORD_POLICY_V7_2026-11-19.md. '
                     'Unity tests can use the committed fixture directly.')
    import numpy as np
    from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes
    from python.build.g2p.ko import KoreanG2P

    assert args.baseline.resolve() != args.output.resolve(), 'Keep the original fixture'
    with gzip.open(args.baseline, 'rt', encoding='utf-8') as stream:
        baseline = json.load(stream)
    assert baseline['policy'] == 'orthographic-word-context-v5'
    assert len(baseline['cases']) == 11804
    state = SimpleNamespace(text='', fraction=1.)
    games = WordGames(SimpleNamespace(ready=True), lambda wave: {'text': state.text}, lambda *_: {})
    games.voice_fraction = lambda _: state.fraction
    original_score = games.score_candidates
    cache = {}

    def score(text, candidates, fraction):
        key = (text, tuple(w for w, _ in candidates), fraction >= .04)
        if key not in cache:
            cache[key] = original_score(text, candidates, fraction)
        return cache[key]
    games.score_candidates = score
    wave = np.full(40000, .03, dtype=np.float32)
    fields = ['candidate', 'similarity', 'passed', 'ambiguous', 'matched_text',
              'comparison_text', 'normalization', 'blocked_by', 'decision_reason',
              'selection_reason', 'confirmation_waiting', 'streak', 'candidate_scores', 'tied_targets']
    cases, changed, counts = [], [], Counter()
    for index, trial in enumerate([*baseline['cases'], *controls()]):
        games.sessions.clear()
        params = {key: trial[key] for key in ['targets', 'threshold', 'consecutive'] if key in trial}
        session = games.start(StartWordGame(**params))['session_id']
        current = {key: value for key, value in trial.items() if key != 'checks'}
        current['checks'] = []
        result = None
        for number, frame in enumerate(trial['checks'], 1):
            state.text, state.fraction = frame['text'], frame['voice_fraction']
            result = games.frame(session, number, frame['seconds'], wave, b'replay', 16000)
            expected = {key: result[key] for key in fields}
            expected['confirmed'] = result['state'] == 'accepted'
            current['checks'].append({**{k: v for k, v in frame.items() if k != 'expected'}, 'expected': expected})
            counts['frames'] += 1
            if expected['confirmed']:
                counts['accepted'] += 1
                break
        cases.append(current)
        if index < len(baseline['cases']):
            old_accepted = any(c['expected']['confirmed'] for c in trial['checks'])
            new_accepted = result['state'] == 'accepted'
            counts[f'baseline_{old_accepted}_to_{new_accepted}'] += 1
            if old_accepted != new_accepted:
                changed.append({'id': trial['id'], 'targets': trial['targets'],
                                'before': old_accepted, 'after': new_accepted,
                                'text': result['prediction']['text']})
        if (index + 1) % 2000 == 0:
            print(f'Replayed {index + 1} cases', flush=True)
    web = Path(__file__).resolve().parent
    sources = [web / name for name in ['word_game.py', 'word_matching.py', 'answer_policy.py',
                                      'hangul_input.py', 'answer_families.json']]
    sources += [PROJECT / 'python/build/g2p/ko' / name for name in ['g2p.py', 'jamo_ipa.py', 'rules.py', 'rules_data.py']]
    g2p = KoreanG2P()
    partials = [chr(ch) for begin, end in [(0x1100, 0x1113), (0x1161, 0x1176), (0x11a8, 0x11c3)] for ch in range(begin, end)]
    partials += ['ᅵᆯ로폰채', 'ᄉᆯ로폰채', '책ᄉ', '사과ᄀ', '실로펀체', '실로폰채', 'ᅢᆨ']
    fixture = {'policy': POLICY, 'case_count': len(cases), **counts,
               'retained_baseline_cases': len(baseline['cases']),
               'baseline_fixture_sha256': digest(args.baseline),
               'source_hashes': {str(path.relative_to(PROJECT)): digest(path) for path in sources},
               'matrix_sha256': baseline['matrix_sha256'],
               'jamo_cases': [{'text': text, 'direct': hangul_to_ipa_phonemes(text), 'with_rules': g2p.to_ipa(text)} for text in partials],
               'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as out:
        out.write(json.dumps(fixture, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    report = {key: value for key, value in fixture.items() if key not in ['cases', 'jamo_cases']}
    report.update(jamo_cases=len(partials), output_sha256=digest(args.output), changed_acceptance=changed,
                  note='Saved ASR text replay and synthetic controls; not fresh audio accuracy. A replay ends on first acceptance. Original v5 cases may already end at v5 acceptance.')
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ['changed_acceptance', 'source_hashes']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
