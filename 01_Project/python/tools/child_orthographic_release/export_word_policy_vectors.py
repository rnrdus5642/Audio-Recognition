"""Replay saved inputs through the production web policy for C# parity tests.

The baseline contributes inputs only, never old expected decisions. Context
examples additionally exercise bounded words, affixes, ambiguity and negatives.
No acoustic inference, recordings, or user-history writes are performed.
"""
import argparse
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WEB = ROOT / '01_Project/python/tools/child_orthographic_web'
sys.path.insert(0, str(WEB))
from word_game import WordGames, StartWordGame, POLICY
from answer_policy import spelling_key


def build(baseline, output, summary):
    original = json.loads(gzip.decompress(baseline.read_bytes()))
    trials = original['cases']
    games = WordGames(SimpleNamespace(ready=True), lambda _: {'text': current['text']}, lambda *_: {})
    games.voice_fraction = lambda _: current['voice_fraction']
    games.g2p.to_ipa = lru_cache(maxsize=100000)(games.g2p.to_ipa)
    match = games.matcher.score_against
    cache = lru_cache(maxsize=500000)(lambda a, b: match(a, b))
    games.matcher.score_against = lambda a, b: cache(tuple(a), tuple(b))

    def add(key, targets, text):
        trials.append({'id': 'v5-context:' + key, 'targets': targets, 'checks': [
            {'text': text, 'seconds': t, 'voice_fraction': .3} for t in (.5, 1.)]})

    for trial in list(trials):
        if trial['id'].startswith('ideal:') and trial['id'].endswith(':clean'):
            word = trial['checks'][0]['text']
            add(trial['id'], trial['targets'], f'저는 {word} 라고 말했어요')
            if spelling_key(word) not in games.action_guard.families:
                add('copula:' + trial['id'], trial['targets'], f'이건 {word}인거 같아요')

    examples = [
        (['사과'], '이건 사과인거 같아요'), (['사과'], '이건 사과인 거 같아요'),
        (['사과'], '이건 사과인것같아요'), (['사과'], '동생도 사과를 따요'),
        (['사과'], '사과주스'), (['사과'], '사과 주스'),
        (['사과'], '사과인형 같아요'), (['사과'], '사과가 아니에요'),
        (['사과'], '사과 말고 배 주세요'), (['사과'], '여기 사과 있어요'),
        (['물'], '여기 물을 주세요'), (['물'], '여기 물고기 있어요'),
        (['흙'], '여기 흙이 있어요'), (['흙'], '흙탕물 같아요'),
        (['책'], '저건 책상 같아요'), (['알'], '저건 알파벳 같아요'),
        (['알'], '저기 알인거 같아요'), (['알'], '그건 알고 있어요'),
        (['씨앗'], '여기 씨앗도 있어요'), (['집'], '저기 집으로 가요'),
        (['집'], '이건 집게 같아요'), (['파'], '나는 파랑 좋아해요'),
        (['따'], '따뜻해 보여요'), (['따'], '간식도 다 먹어요'),
        (['들어'], '아주 힘들어 보여요'), (['들어'], '지금 흔들어 해 보자'),
        (['사과'], '이건 “사과?” 같아요'), (['사과'], 'ᄉ 사과 있어요'),
        (['사과'], '사과ᄌ 있어요'), (['낫', '낮'], '여기 낫 있어요'),
        (['먹이', '주자'], '먹이를 주자'), (['밥', '줘요'], '밥을 줘요'),
        (['사과', '배'], '사과 배'), (['배', '사과'], '사과 배'),
        (['사과', '사과야'], '사과야'), (['들거에요'], '나는 들거에요라고 했어요'),
        (['들거에요'], '나는 들거예요라고 했어요'), (['사과'], '사과는 아니에요'),
    ]
    for i, (targets, text) in enumerate(examples):
        add(str(i), targets, text)

    wave = np.zeros(40000, dtype=np.float32)
    current = {}
    total_frames = accepted = 0
    keys = ('candidate', 'similarity', 'passed', 'ambiguous', 'matched_text',
            'comparison_text', 'normalization', 'blocked_by', 'decision_reason',
            'selection_reason', 'confirmation_waiting', 'streak', 'candidate_scores', 'tied_targets')
    for n, trial in enumerate(trials, 1):
        games.sessions.clear()
        session = games.start(StartWordGame(targets=trial['targets']))['session_id']
        checks = []
        for index, check in enumerate(trial['checks'], 1):
            current = check
            result = games.frame(session, index, check['seconds'], wave, b'policy-parity', 16000)
            expected = {key: result[key] for key in keys}
            expected['confirmed'] = result['state'] == 'accepted'
            checks.append({key: check[key] for key in ('text', 'seconds', 'voice_fraction')} | {'expected': expected})
            total_frames += 1
            if expected['confirmed']:
                accepted += 1
                break
        trial['checks'] = checks
        if n % 2000 == 0:
            print(json.dumps({'cases': n, 'frames': total_frames}), flush=True)
    fixture = {'policy': POLICY, 'case_count': len(trials), 'frames': total_frames,
               'accepted': accepted, 'baseline_inputs_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
               'source_hashes': {name: hashlib.sha256((WEB / name).read_bytes()).hexdigest()
                                 for name in ('word_game.py', 'answer_policy.py', 'answer_families.json')},
               'matrix_sha256': hashlib.sha256((ROOT / '01_Project/shared/confusion_matrices/ko_child_v2.json').read_bytes()).hexdigest(),
               'cases': trials}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as target:
        target.write(json.dumps(fixture, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    receipt = {k: v for k, v in fixture.items() if k != 'cases'}
    receipt.update(output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    summary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    build(args.baseline.resolve(), args.output.resolve(), args.summary.resolve())
