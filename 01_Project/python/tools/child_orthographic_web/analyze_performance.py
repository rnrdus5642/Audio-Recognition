"""Explain the 2026-10-30 deployment evaluation without changing the running app."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np

from evaluate_performance import ROOT, RATE, encoded, safe, waveform, write
from word_game import StartWordGame, WordGames


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def replay_cached(games, row, targets):
    """Replay the same measured windows with an alternative candidate set."""
    session = games.start(StartWordGame(targets=targets))['session_id']
    wave = waveform(row)
    signal = np.zeros(RATE * 10, dtype=np.float32)
    signal[6400:6400 + len(wave)] = wave
    for index, frame in enumerate(row['frames'], 1):
        end = index * 8000
        chunk = signal[max(0, end - 40000):end]
        chunk = np.pad(chunk, (40000 - len(chunk), 0))
        prediction = frame['prediction']
        games.infer = lambda _wave, prediction=prediction: prediction
        answer = games.frame(session, index, frame['elapsed_seconds'], chunk, encoded(chunk), RATE)
        if answer['state'] != 'listening':
            break
    answer = games.finish(session, 'timeout')
    answer['trace'] = games.get(session)['trace']
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = safe(args.output)
    summary = load(output / 'summary.json')
    assert summary['status'] == 'complete'
    asr = load(output / 'asr-results.json')
    games_rows = [json.loads(line) for line in (output / 'game-results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(games_rows) == len({row['record_id'] for row in games_rows}) == 455
    asr_index = {(row['group'], row['record_id']): row for row in asr}
    with (ROOT / '01_Project/shared/words.csv').open(encoding='utf-8-sig', newline='') as stream:
        curriculum = [row['text'] for row in csv.DictReader(stream)]
    music_words = sorted({row['text'] for row in games_rows if row['group'] == 'music'})
    app_words = sorted(set(curriculum + music_words + ['잡아', '작아', '짹', '택', '물음표']))
    matcher = WordGames(SimpleNamespace(ready=True), None,
                        lambda state, frame: {'prediction': frame['prediction'], 'similarity': frame['similarity']})
    phonemes = {word: matcher.g2p.to_ipa(word) for word in app_words}
    pairs = []
    for spoken in app_words:
        for target in app_words:
            if spoken == target:
                continue
            similarity = matcher.matcher.score_against(phonemes[spoken], phonemes[target])[1]
            if similarity >= .85:
                pairs.append({'recognized_text': spoken, 'target': target, 'similarity': similarity,
                              'same_ipa': phonemes[spoken] == phonemes[target]})

    # The main negative test excludes pairs intentionally admitted by the matrix.
    # Examine those pairs separately; a score of 0.85 is NOT an error probability.
    extra = []
    for row in games_rows:
        if row['group'] == 'dev_word':
            continue
        for pair in pairs:
            if row['text'] != pair['recognized_text']:
                continue
            result = replay_cached(matcher, row, [pair['target']])
            extra.append({'record_id': row['record_id'], 'group': row['group'], 'reference': row['text'],
                          'target': pair['target'], 'ideal_score': pair['similarity'], 'result': result})
    write(output / 'similar-target-results.json', extra)

    group_names = {'dev_word': '개발 검증 단어', 'music': '악기놀이 녹음',
                   'local_child': '기존 아동 녹음', 'user_grab_control': '잡아·작아 직접 녹음'}
    rows = []
    for row in games_rows:
        original = asr_index[row['group'], row['record_id']]
        trial = row['trials']['single']
        record = trial['record']
        audio = safe(row['audio_path'])
        rows.append({'id': row['record_id'], 'word': row['text'], 'group': row['group'],
                     'group_label': group_names[row['group']], 'age': row['age'],
                     'speaker_key': row['speaker_key'], 'seconds': row['seconds'],
                     'full_text': original['prediction']['text'], 'full_exact': original['exact'],
                     'any_exact_window': row['any_exact_window'],
                     'accepted': trial['state'] == 'accepted',
                     'decision_text': record['prediction']['text'], 'decision_score': record['similarity'],
                     'one_pass_frame': any(frame['passed'] for frame in trial['trace']),
                     'audio': quote(os.path.relpath(audio, output).replace('\\', '/'), safe='/'),
                     'trace': [{key: frame[key] for key in ('elapsed_seconds', 'text', 'similarity', 'passed', 'streak')}
                               for frame in trial['trace']]})

    word_groups = []
    for group in group_names:
        for word in sorted({row['word'] for row in rows if row['group'] == group}):
            selected = [row for row in rows if row['group'] == group and row['word'] == word]
            word_groups.append({'word': word, 'group': group, 'clips': len(selected),
                                'speakers': len({row['speaker_key'] for row in selected}),
                                'full_exact': sum(row['full_exact'] for row in selected),
                                'any_exact_window': sum(row['any_exact_window'] for row in selected),
                                'single_accepted': sum(row['accepted'] for row in selected),
                                'full_predictions': dict(Counter(row['full_text'] for row in selected))})

    failures = {}
    for group in group_names:
        failed = [row for row in rows if row['group'] == group and not row['accepted']]
        failures[group] = {'failed_clips': len(failed),
                           'one_or_more_pass_frames_but_not_two_consecutive': sum(row['one_pass_frame'] for row in failed),
                           'no_pass_frames': sum(not row['one_pass_frame'] for row in failed)}
    extra_groups = []
    for source, target in sorted({(row['reference'], row['target']) for row in extra}):
        selected = [row for row in extra if row['reference'] == source and row['target'] == target]
        extra_groups.append({'spoken_label': source, 'target': target, 'clips': len(selected),
                             'accepted': sum(row['result']['state'] == 'accepted' for row in selected),
                             'ideal_score': selected[0]['ideal_score']})
    hashes = {}
    for path in [Path(__file__), Path(__file__).with_name('evaluate_performance.py'),
                 Path(__file__).with_name('word_game.py'), ROOT / '01_Project/shared/confusion_matrices/ko_child_v2.json']:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {'protocol': 'same measured ASR windows, same WordGames; extra negatives chosen because ideal score >=0.85',
             'unique_game_words': len({row['word'] for row in rows}), 'word_by_group': word_groups,
             'failure_summary': failures, 'ideal_high_similarity_pairs': pairs, 'similar_target_replays': extra_groups,
             'curriculum_words_without_game_audio': [word for word in curriculum if not any(row['word'] == word for row in rows)],
             'source_sha256': hashes}
    write(output / 'audit.json', audit)

    age_rows = ''.join(f'<tr><td>{age}세</td><td>{metrics["clips"]:,}</td><td>{metrics["speakers"]}</td>'
                       f'<td>{metrics["cer"]:.2%}</td><td>{summary["game_dev_by_age"][age]["single"]["correct"]}/80 '
                       f'({summary["game_dev_by_age"][age]["single"]["correct_rate"]:.1%})</td></tr>'
                       for age, metrics in summary['monitor_by_age'].items())
    local_rows = ''.join(f'<tr><td>{group_names[group]}</td><td>{metrics["single"]["correct"]}/{metrics["single"]["clips"]}</td>'
                         f'<td>{metrics["single"]["correct_rate"]:.1%}</td></tr>'
                         for group, metrics in summary['game_by_group'].items() if group != 'dev_word')
    pair_rows = ''.join(f'<tr><td>{html.escape(row["spoken_label"])}</td><td>{html.escape(row["target"])}</td>'
                       f'<td>{row["ideal_score"]:.1%}</td><td>{row["accepted"]}/{row["clips"]}</td></tr>' for row in extra_groups)
    document = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>음성인식·단어 판정 검사 · 2026-10-30</title><style>
:root{color-scheme:light;font-family:system-ui,"Malgun Gothic",sans-serif;color:#172132;background:#f4f6fa}body{max-width:1120px;margin:32px auto;padding:0 20px 60px;line-height:1.7}h1{font-size:28px;line-height:1.35}h2{font-size:21px;margin-top:36px}section,.card{background:white;border:1px solid #dce2eb;border-radius:12px;padding:20px;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e3e7ef;padding:10px;vertical-align:top}th{background:#f1f4f9}small,.muted{color:#596779}input,select{padding:9px;border:1px solid #adb8c9;border-radius:6px;font:inherit;max-width:100%}.filters{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:16px 0}.scroll{overflow:auto}.bad{color:#a62929;font-weight:700}.good{color:#176542;font-weight:700}summary{cursor:pointer;font-weight:650}audio{width:250px;max-width:100%}.detail{margin:12px 0;padding:12px;background:#f7f9fc;border-radius:8px}.metrics{display:flex;gap:20px;flex-wrap:wrap}.metrics strong{display:block;font-size:24px}code{word-break:break-all}a{color:#1556a5}.trace{font-size:13px;white-space:pre-wrap}button{cursor:pointer}
</style><body><h1>음성인식 모델과 정답 판정 검사</h1><p>2026-10-30 · 웹사이트에 적용된 <b>2.01에포크 모델</b> · 학습 재개 없이 검사</p>
<section><p><b>현재 토큰과 디코더는 현대 한글 11,172자를 모두 표현합니다.</b> 예전 모델에서 빠졌던 ‘놔’도 지원합니다. 다만 실제 음성을 정확히 인식하는 능력과 정답 판정은 별도이며, 단어·화자에 따른 실패가 확인됐습니다.</p>
<div class="metrics"><div><strong>__CER__</strong>연령별 동일 비중 평균 CER</div><div><strong>283 / 320</strong>개발 검증 단어 정답 통과</div><div><strong>268개</strong>게임 재생 검사 단어 종류</div></div>
<p>‘들어’는 악기놀이 녹음 3/6, ‘북채’는 1/2, ‘책’은 기존 아동 녹음 1/3이 통과했습니다. ‘잡아’는 악기놀이 6/6과 직접 녹음 5/5가 통과했습니다. 표본이 작고 일부 녹음 라벨은 청취 검수가 끝나지 않았으므로, 실패 단어를 곧바로 사용 불가능으로 단정하지 않습니다.</p></section>
<h2>텍스트 인식과 게임 판정</h2><p>CER는 원문에서 공백을 제외한 글자의 교체·삭제·삽입 수를 정답 글자 수로 나눈 값입니다. 낮을수록 좋으며 단어 정답률과 다릅니다. 게임 통과는 현재 85점 기준과 같은 후보 2회 연속 판정을 적용했습니다.</p>
<div class="scroll"><table><thead><tr><th>연령</th><th>CER 검사 음성</th><th>화자 ID 수</th><th>CER</th><th>별도 단어 표본 통과</th></tr></thead><tbody>__AGE_ROWS__</tbody></table></div>
<p class="muted">CER: 모니터 검증 1,797개, 약 2시간. 전체 합산 CER 9.32%, 연령 균등 평균 8.80%. 단어 판정: 개발 검증셋에서 연령별 80개씩, 화자 순환 방식으로 선정. 3세 단어 표본은 6명, 4세 13명, 5세 61명, 6세 80명입니다. 두 평가는 별도 표본이며 일부 음성은 중복됩니다.</p>
<table><thead><tr><th>기존 녹음 그룹</th><th>정답 통과 / 검사 수</th><th>통과율</th></tr></thead><tbody>__LOCAL_ROWS__</tbody></table>
<p>모든 그룹의 기본 판정은 정답 후보 하나를 넣은 재생 검사입니다. 정답 포함 무작위 후보 10개에서는 개발 검증 283/320이 정답 후보로 통과했습니다. 악기놀이 실제 후보 12개에서는 39/46이 라벨과 일치했고, 1개는 ‘놔’ 대신 ‘놓아’로 선택됐으며 6개는 재시도였습니다. 같은 동작을 뜻하는 두 표현을 별개 후보로 둔 조건입니다.</p>
<h2>틀린 단어와 비슷한 단어</h2><p>기준 점수가 85점 미만인 유사 오답을 정답으로 지정한 검사에서 개발 검증 2/320, 기존 아동 1/82가 잘못 통과했습니다. 이는 의도적으로 고른 어려운 오답 조건이며 일상 사용의 오탐률 추정치가 아닙니다. ‘작아’ 직접 녹음 2개를 ‘잡아’로 판정한 경우는 모두 거부됐습니다. 무음·합성 백색잡음 9회도 통과하지 않았습니다.</p>
<p><b>판정 기준 자체가 허용하는 비슷한 단어도 있습니다.</b> 아래는 다른 단어를 정답으로 지정한 추가 검사입니다. 텍스트가 완벽하게 인식돼도 해당 유사도 이상으로 처리될 수 있습니다. ‘실로폰/실로폰채’, ‘흔들어/흔들어요’처럼 표현 차이를 허용할지는 서비스 기준에 따라 결정해야 합니다. 점수는 확률이 아닙니다.</p>
<div class="scroll"><table><thead><tr><th>음성의 원래 라벨</th><th>대신 지정한 정답</th><th>라벨 간 유사도</th><th>실제 음성 통과</th></tr></thead><tbody>__PAIR_ROWS__</tbody></table></div>
<h2>단어별 결과와 원본 듣기</h2><p>‘전체 음성 일치’는 녹음 전체를 한 번 인식한 결과, ‘게임 통과’는 최근 2.5초를 0.5초마다 판정한 결과입니다. ‘실로폰채 → 실로폰 책’은 전체 인식에서 2/2가 틀렸지만 판정은 2/2 통과했습니다. ‘놔요 → 나요’도 통과 사례가 있습니다. 정확한 받아쓰기와 발음 유사도 허용을 구분해 확인할 수 있습니다.</p>
<div class="filters"><label>단어 검색 <input id="search" placeholder="놔요, 책, 실로폰…"></label><label>녹음 그룹 <select id="group"><option value="">전체</option><option value="music">악기놀이 녹음</option><option value="local_child">기존 아동 녹음</option><option value="dev_word">개발 검증 단어</option><option value="user_grab_control">잡아·작아 직접 녹음</option></select></label><label><input type="checkbox" id="failed"> 실패가 있는 단어만</label></div><p id="count"></p><div id="words"></div>
<h2>해석 범위와 다음 작업</h2><ul><li>모니터·개발 검증셋은 학습 중 모델 선택에 사용한 데이터입니다. 최종 미사용 테스트셋 성능으로 해석하지 않습니다.</li><li>게임 검사는 원본 앞에 0.4초 무음을 추가하고, 발화 후 무음을 채운 10초 타임라인을 재생했습니다. 브라우저 마이크·네트워크 지연·실제 방 소음은 포함하지 않았습니다.</li><li>기존 아동·악기놀이 녹음은 연령 미확인, 일부 라벨과 분할의 청취 검수 미완료입니다. 라벨이 틀렸거나 발음이 목표와 달랐을 가능성도 남아 있습니다.</li><li>통과가 0회였던 23개 단어 대부분은 표본 1개입니다. ‘북채’, ‘택배’처럼 한 번 맞게 인식해도 두 번 연속 조건을 만족하지 못한 사례도 있습니다. 원본과 판정 시점의 텍스트를 함께 검토해야 합니다.</li><li>웹 입력은 후보 1~20개, 후보당 한글 1~20글자입니다. 내부 공백·숫자·영문은 거부하며 끝의 물음표는 비교에서 제외합니다. 이는 모델의 글자 누락과 다른 입력 제한입니다.</li><li>우선 실패 녹음의 라벨을 청취 확인하고 실제 사용할 단어·허용 표현 목록을 확정한 뒤, 다른 단어를 잘못 통과시키는 비율까지 보며 판정 기준을 조정하는 것이 좋습니다.</li></ul>
<p>모델 식별자 <code>__MODEL_ID__</code></p><p><a href="summary.json">전체 집계 JSON</a> · <a href="word-results.json">단어별 집계 JSON</a> · <a href="audit.json">추가 분석 JSON</a> · <a href="plan.json">검사 조건</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
function render(){const search=document.getElementById('search').value.trim(),group=document.getElementById('group').value,failed=document.getElementById('failed').checked,root=document.getElementById('words');root.replaceChildren();const selected=data.filter(r=>(!search||r.word.includes(search))&&(!group||r.group===group));const grouped=new Map();selected.forEach(r=>{if(!grouped.has(r.word))grouped.set(r.word,[]);grouped.get(r.word).push(r)});let shown=0,clips=0;
for(const [word,rows] of [...grouped.entries()].sort((a,b)=>a[0].localeCompare(b[0],'ko'))){const count=rows.filter(r=>r.accepted).length;if(failed&&count===rows.length)continue;shown++;clips+=rows.length;const d=node('details',undefined,'card');d.append(node('summary',`${word} · 게임 ${count}/${rows.length} · 전체 음성 일치 ${rows.filter(r=>r.full_exact).length}/${rows.length}`,count<rows.length?'bad':'good'));d.addEventListener('toggle',()=>{if(!d.open||d.dataset.built)return;d.dataset.built='1';for(const r of rows){const card=node('div',undefined,'detail');card.append(node('p',`${r.group_label} · ${r.age==='unknown'?'연령 미확인':r.age+'세'} · ${r.seconds.toFixed(2)}초 · ${r.accepted?'통과':'재시도'}`,r.accepted?'good':'bad'));card.append(node('p',`전체 음성: ${r.full_text||'(빈 문자열)'} / ${r.accepted?'확정한':'최고 점수'} 구간: ${r.decision_text||'(빈 문자열)'} (${(100*r.decision_score).toFixed(1)}점)`));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=r.audio;card.append(audio);const tr=node('details');tr.append(node('summary','시간별 인식 결과'));tr.append(node('div',r.trace.map(f=>`${f.elapsed_seconds.toFixed(1)}초  ${f.text||'(빈 문자열)'}  ${(100*f.similarity).toFixed(1)}점  연속 ${f.streak}회`).join('\n'),'trace'));card.append(tr);card.append(node('small',r.id));d.append(card)}});root.append(d)}document.getElementById('count').textContent=`${shown}개 단어 · ${clips}개 녹음 (그룹마다 표본과 라벨 신뢰도가 다릅니다.)`;}
['search','group','failed'].forEach(id=>document.getElementById(id).addEventListener('input',render));render();
</script></body></html>'''
    replacements = {'__CER__': f'{summary["monitor_macro_cer"]:.2%}', '__AGE_ROWS__': age_rows,
                    '__LOCAL_ROWS__': local_rows, '__PAIR_ROWS__': pair_rows,
                    '__MODEL_ID__': html.escape(summary['plan']['model']['model_id']),
                    '__DATA__': json.dumps(rows, ensure_ascii=False).replace('<', '\\u003c')}
    for token, value in replacements.items():
        document = document.replace(token, value)
    (output / 'report.html').write_text(document, encoding='utf-8')
    print(json.dumps({'report': str(output / 'report.html'), 'unique_words': audit['unique_game_words'],
                      'failure_summary': failures, 'extra_pairs': extra_groups,
                      'curriculum_without_audio': audit['curriculum_words_without_game_audio']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
