"""Write the local, self-contained curriculum audit report."""
import argparse
from collections import Counter
import gzip
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from evaluate_performance import safe, ROOT, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    p=safe(parser.parse_args().source)
    def load(name):
        return json.loads((p/name).read_text(encoding='utf-8'))
    summary=load('summary.json')
    catalog=load('catalog.json')
    coverage=load('word-coverage.json')
    audio=load('audio-evaluation/word-results.json')
    stages=load('audio-evaluation/stage-results.json')
    risk=load('stage-phonetic-risks.json')
    with gzip.open(p/'audio-evaluation/stage-trials.json.gz','rt',encoding='utf-8') as stream:
        trials=json.load(stream)
    isolated=load('audio-evaluation/isolated-asr.json')
    context=load('audio-evaluation/context-asr.json')
    names={'music':'악기 연주하기','dinosaur':'공룡알 키우기','plant':'식물 키우기'}
    groups={'music':'기존 악기놀이 녹음','local_child':'기존 아동 녹음','dev_curriculum':'개발 검증 단어','dev_context':'개발 검증 문장'}
    audio_by_word={r['word']:r for r in audio}
    missing=[]
    word_rows=[]
    for row in coverage:
        stat=row['original_form']['train']
        alt=row['alternative_counts']['train'] if row['alternative_counts'] else None
        absent=stat['substring_utterances']==0 and not (alt and alt['substring_utterances'])
        if absent:missing.append(row['word'])
        examples=[]
        for clip in isolated+context:
            is_context='curriculum_words' in clip
            if row['word'] not in (clip['curriculum_words'] if is_context else [clip['curriculum_word']]):
                continue
            examples.append({'kind':'문장 속 검사' if is_context else '단독 발화 검사','group':groups[clip['group']],
                             'reference':clip['text'],'prediction':clip['prediction']['text'],'age':clip['age'],
                             'audio':quote(os.path.relpath(safe(clip['audio_path']),p).replace('\\','/'),safe='/')})
        word_rows.append({**row,'audio_result':audio_by_word[row['word']],'training_string_absent':absent,'examples':examples})
    wrong=[{'reference':r['reference'],'stage':r['stage'],'chosen':r['current']['matched_target'],
            'recognized':r['current']['prediction']['text'],'score':r['current']['similarity']}
           for r in trials if not r['is_positive'] and r['current']['state']=='accepted']
    counts={'supplied_entries':199,'unique_expressions':146,'stages':27,'missing_training_string':len(missing),
            'isolated_audio_words':16,'isolated_audio_clips':58,'context_words':47,'context_clips':140,
            'no_audio_checked':len(summary['no_isolated_or_context_audio'])}
    write(p/'report-data.json',{'counts':counts,'word_results':word_rows,'stage_results':stages,'actual_out_of_stage_passes':wrong})
    activity_table=''.join(f'<tr><td>{names[a]}</td><td>9</td><td>{x["entries"]}</td><td>{x["unique_words"]}</td></tr>' for a,x in summary['activities'].items())
    stage_table=''.join(f'<tr><td>{names[s["activity"]]} {s["number"]}</td><td>{html.escape(s["procedure"])}</td>'
                        f'<td>{len(s["targets"])}</td><td>{html.escape(", ".join(s["tested_positive_words"])) or "—"}</td>'
                        f'<td>{str(s["positive_current_accepted"])+"/"+str(s["positive_clips"]) if s["positive_clips"] else "단독 음성 미확인"}</td></tr>' for s in stages)
    wrong_table=''.join(f'<tr><td>{html.escape(r["reference"])}</td><td>{html.escape(r["recognized"])}</td><td>{html.escape(r["chosen"])}</td><td>{r["score"]*100:.1f}</td></tr>' for r in wrong)
    doc=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>27단계 정답 표현 분석</title>
<style>:root{font-family:system-ui,"Malgun Gothic",sans-serif;color:#18253a;background:#f4f6fa}body{max-width:1180px;margin:28px auto;padding:0 20px 60px;line-height:1.7}h1{font-size:28px;line-height:1.35}h2{font-size:21px;margin-top:30px}h3{font-size:17px}.box{background:#fff;border:1px solid #dce3ee;border-radius:12px;padding:20px;margin:18px 0}.metrics{display:flex;flex-wrap:wrap;gap:25px}.metrics strong{font-size:27px;display:block}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dfe5ef;padding:10px;text-align:left;vertical-align:top}th{background:#eaf0f8}.scroll{overflow:auto}.muted,small{color:#58677b}.bad{color:#a32a2a;font-weight:650}.good{color:#126445;font-weight:650}input,select{font:inherit;padding:8px;border:1px solid #b3bfd0;border-radius:6px;max-width:100%}.filters{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin:15px 0}summary{cursor:pointer;font-weight:650}audio{width:260px;max-width:100%}.sample{padding:10px;margin:10px 0;background:#f3f6fa;border-radius:6px}.tag{display:inline-block;background:#e7eef9;padding:2px 8px;border-radius:5px;margin:2px}a{color:#1554aa}code{word-break:break-all}</style><body>
<h1>3개 활동 · 27단계의 정답 표현 분석</h1><p>2026-10-30 · 사용자 제공 원문 199개 항목 · 현재 웹의 약 2.01에포크 모델</p>
<div class="box"><div class="metrics"><div><strong>199개</strong>단계별 등록 항목</div><div><strong>146개</strong>중복 제외 표현</div><div><strong>146 / 146</strong>토큰·디코더 표현 가능</div><div><strong>72개</strong>학습 라벨에 해당 문자열 없음</div></div>
<p><b>핵심은 현재 단계에서 허용한 표현인지 판단하는 것입니다.</b> ‘들어 → 들어요’처럼 같은 단계 안의 표현 변경은 정답으로 볼 수 있습니다. ‘쳐요 → 줘요’처럼 다른 동작으로 통과하는 문제는 별도로 막아야 합니다.</p><p>72개가 학습 라벨에 없다는 사실은 출력 불가능을 뜻하지 않습니다. 실제로 ‘실로폰채’는 학습 라벨에서 0회지만 기존 녹음 2개 모두 정답 판정에 통과했습니다. 다만 전체 음성의 글자 출력은 모두 ‘실로폰 책’이었습니다.</p></div>
<h2>목록과 검증 범위</h2><table><thead><tr><th>활동</th><th>단계</th><th>항목 수</th><th>활동 내 중복 제외</th></tr></thead><tbody>__ACTIVITIES__</tbody></table>
<p>활동 사이에도 겹치는 표현이 있어 전체 고유 표현은 146개입니다. 탬버린 놓기의 ‘놓아야죠’ 누락을 포함하여 제공한 목록을 그대로 보존했습니다. 끝의 물음표는 판정에서 제외됩니다. 단계당 1~11개 후보이므로 현재 웹의 최대 20개 제한 안에 들어갑니다.</p>
<p><b>실제 단독 음성 검사는 16개 표현·58개 녹음에 한정됩니다.</b> 추가 47개 표현은 개발 검증 문장 140개에서 해당 표현이 보존되는지 확인했습니다. 나머지 83개는 이번에 실제 음성으로 확인하지 못했습니다. 문장 속 인식 성공은 그 단어만 말했을 때의 성공을 보장하지 않습니다.</p>
<h2>서로 다른 동작이 높은 점수를 받는 문제</h2><p>아래 점수는 모델이 왼쪽 표현을 정확히 출력했을 때 현재 발음 유사도 계산이 주는 점수입니다. 두 번 연속 같은 결과가 나오면 85점 기준을 통과할 수 있습니다. 확률이나 실제 오탐률은 아닙니다.</p>
<table><thead><tr><th>인식된 표현</th><th>다른 단계의 정답</th><th>현재 점수</th></tr></thead><tbody><tr><td>넣을거에요</td><td>놓을거에요</td><td>95.6</td></tr><tr><td>넣을거야</td><td>놓을거야</td><td>95.0</td></tr><tr><td>닦을거에요</td><td>담을거에요</td><td>92.0</td></tr><tr><td>쳐요</td><td>줘요</td><td>88.0</td></tr></tbody></table>
<p>‘거에요·거야·까요’처럼 공통으로 붙은 부분이 긴 표현에서는 동작을 구별하는 소리의 차이가 전체 점수에서 작게 반영됩니다. 단계별 후보를 제한해도 오답 음성이 현재 단계의 후보와 85점 이상이면 통과할 수 있으므로, 단계 제한만으로 해결되지는 않습니다.</p>
<details class="box"><summary>실제 음성에서 다른 단계로 통과한 7건</summary><p>의도적으로 다른 단계의 후보를 지정한 검사입니다. 일반 사용의 오탐률로 계산하지 않습니다. 음성 라벨 자체의 청취 검수는 미완료입니다.</p><table><thead><tr><th>음성 라벨</th><th>인식 텍스트</th><th>통과한 후보</th><th>점수</th></tr></thead><tbody>__WRONG__</tbody></table></details>
<h2>‘놔’와 ‘놔요’는 별도로 확인해야 합니다</h2><p>제공된 놓기 목록에는 ‘놓아·놓아요’가 있고 ‘놔·놔요’는 없습니다. 현재 계산에서 <b>놔 → 놓아는 73.3점, 놔요 → 놓아요는 84점</b>이므로 정확히 ‘놔’ 또는 ‘놔요’로 인식되면 85점 기준을 통과하지 못합니다. 이 축약형을 허용할지는 정답 목록의 정책으로 정할 필요가 있습니다. 이번 분석에서는 임의로 추가하지 않았습니다.</p>
<h2>현재 단계 후보를 실제로 넣은 결과</h2><p>정답 후보 하나가 아닌, 제공한 각 단계의 전체 후보를 사용했습니다. 같은 단계의 다른 허용 표현으로 확정되어도 정답 통과로 셌습니다. 동일 녹음이 여러 단계에서 반복 사용됐으므로 행별 결과를 합산해 전체 인식률로 해석하지 않습니다. ‘미확인’은 실패가 아닙니다.</p>
<div class="scroll"><table><thead><tr><th>활동·단계</th><th>절차</th><th>허용 표현</th><th>단독 음성이 있는 표현</th><th>정답 통과/녹음</th></tr></thead><tbody>__STAGES__</tbody></table></div>
<p class="muted">실로폰채 2/2, 북채 1/2, 들어 4/9, 들어요 1/2, 쳐 6/8, 쳐요 2/4, 흔들어 4/4, 흔들어요 1/2, 두드려 2/3, 두드려요 2/2, 놓아 3/4, 집 12/12, 까 1/1, 들까 0/1, 닦아요 0/1, 닦아야지 0/1. 표본과 화자가 적으므로 단어별 일반 성능으로 단정하지 않습니다.</p>
<p>현재는 같은 후보가 2회 연속이어야 확정됩니다. 같은 단계의 허용 후보끼리 교대해도 2회로 인정하는 방식도 따로 계산했지만, 이번 표본에서는 추가 통과가 없었습니다. 연속 조건만 바꿔도 문제가 해결된다는 근거는 나오지 않았습니다.</p>
<h2>146개 표현 상세</h2><p>학습 목록 640,268개와 개발 검증 목록 120,653개를 확인했습니다. ‘완성형’은 문장 속에서 앞뒤가 어절 경계인 표현이며, 한 단어만 녹음했다는 뜻이 아닙니다. ‘조사형’은 명사에 이·가·을·를·에 등 정해진 조사가 붙은 경우입니다. 두 수치는 동일 문장에서 겹칠 수 있어 합산하지 않습니다. ‘문자열 포함’에는 ‘물고기’ 속 ‘물’처럼 다른 낱말의 일부도 포함됩니다.</p>
<div class="filters"><label>표현 검색 <input id="search" placeholder="들어, 화분, 넣을…"></label><label>단계 <select id="stage"><option value="">전체 27단계</option>__OPTIONS__</select></label><label><input id="missing" type="checkbox"> 학습 문자열 없는 표현만</label><label><input id="untested" type="checkbox"> 단독 음성 미확인만</label></div><p id="count"></p><div id="words"></div>
<details class="box"><summary>학습 라벨에서 문자열 자체가 확인되지 않은 72개 표현</summary><p>띄어쓰기 및 ‘거예요’ 대응 표기까지 확인했습니다. 조각을 조합해 일반화할 수 있으므로 미지원 목록이 아닙니다.</p><p>__MISSING__</p></details>
<h2>적용 방향</h2><ol><li>현재 활동·단계를 입력으로 정하고 그 단계의 허용 표현만 비교합니다. ‘들어/들어요’ 같은 표현은 같은 행동의 정답으로 묶되, 화면에는 실제 인식 텍스트를 유지합니다.</li><li>넣기/놓기, 닦기/담기, 치기/주기처럼 다른 행동을 구별하는 소리의 차이를 더 엄격하게 평가하는 방법을 검토합니다. 임계값 일괄 하향은 피하고 정답 통과와 다른 행동의 오답 통과를 함께 검증해야 합니다.</li><li>‘실로폰채/북채’, 긴 활용형, 아직 단독 음성이 없는 명사·한 음절 표현을 실제 사용 방식으로 녹음해 검증합니다. 특히 식물 활동은 이번에 단독 정답 판정 음성이 확보되지 않았습니다.</li><li>‘놓아/놔’, ‘거에요/거예요’ 등 허용할 표기와 축약형을 명시합니다. 원문 목록의 변형을 자동으로 추가하지 않았습니다.</li></ol>
<p class="muted">한글 토큰 지원과 정확한 받아쓰기는 별개입니다. 이번 검사는 개발 데이터와 기존 녹음을 사용했으며 최종 미사용 테스트셋 성능이 아닙니다. 기존 녹음의 나이 및 일부 라벨·구간은 청취 확인이 필요합니다. 웹의 모델·판정 설정·학습 상태는 변경하지 않았습니다.</p>
<p><a href="report-data.json">표현별 분석 JSON</a> · <a href="catalog.json">원문 단계 목록 JSON</a> · <a href="stage-phonetic-risks.json">단계 간 유사도 위험 JSON</a> · <a href="audio-evaluation/summary.json">음성 검사 조건</a></p>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
function render(){const q=document.getElementById('search').value.trim(),st=document.getElementById('stage').value,missing=document.getElementById('missing').checked,untested=document.getElementById('untested').checked;const root=document.getElementById('words');root.replaceChildren();let count=0;for(const r of data){if(q&&!r.word.includes(q)||st&&!r.stages.includes(st)||missing&&!r.training_string_absent||untested&&r.audio_result.isolated_clips)continue;count++;const t=r.original_form.train,a=r.audio_result,d=node('details',undefined,'box');d.append(node('summary',`${r.word} · 학습 완성형 ${t.boundary_utterances}건 · 단독 음성 ${a.isolated_clips}개 · 문장 속 ${a.context_clips}개`,r.training_string_absent?'bad':''));d.addEventListener('toggle',()=>{if(!d.open||d.dataset.built)return;d.dataset.built='1';d.append(node('p',`출력 글자 지원: ${r.token_roundtrip_ok?'가능':'실패'} / 학습 문자열 포함 ${t.substring_utterances}건 / 완성형 ${t.boundary_utterances}건 / 명사+조사 ${t.noun_particle_utterances}건 / 단독 발화 라벨 ${t.standalone_utterances}건`));if(r.alternative_form){const v=r.alternative_counts.train;d.append(node('p',`다른 표기 ${r.alternative_form}: 완성형 ${v.boundary_utterances}건, 문자열 포함 ${v.substring_utterances}건 (원문 표기는 그대로 유지)`))}d.append(node('p',`완성형을 포함한 학습 발화: 3세 ${t.boundary_by_age['3']}, 4세 ${t.boundary_by_age['4']}, 5세 ${t.boundary_by_age['5']}, 6세 ${t.boundary_by_age['6']} / 화자 ID ${t.boundary_speakers}개`));d.append(node('p',a.isolated_clips?`전체 음성 받아쓰기 일치 ${a.isolated_exact_text}/${a.isolated_clips}. 단계별 정답 통과: `+Object.entries(a.by_stage).map(([id,v])=>`${id} ${v.current_accepted}/${v.clips}`).join(', '):'단어만 말한 음성의 정답 판정: 미확인'));d.append(node('p',a.context_clips?`문장 속 해당 표현 유지: ${a.context_preserved}/${a.context_clips} (단독 발화 통과율이 아닙니다.)`:'문장 속 음성 검사: 미확인'));for(const sample of r.examples){const box=node('div',undefined,'sample');box.append(node('p',`${sample.kind} · ${sample.group} · ${sample.age==='unknown'?'연령 미확인':sample.age+'세'}`));box.append(node('p',`라벨: ${sample.reference}`));box.append(node('p',`인식: ${sample.prediction||'(빈 문자열)'}`));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=sample.audio;box.append(audio);d.append(box)}});root.append(d)}document.getElementById('count').textContent=`${count}개 표현. 펼치면 원본 음성과 인식 결과를 볼 수 있습니다.`}['search','stage','missing','untested'].forEach(id=>document.getElementById(id).addEventListener('input',render));render();
</script></body></html>'''
    options=''.join(f'<option value="{s["id"]}">{names[s["activity"]]} {s["number"]} · {html.escape(s["procedure"])}</option>' for s in catalog)
    replacements={'__ACTIVITIES__':activity_table,'__WRONG__':wrong_table,'__STAGES__':stage_table,'__OPTIONS__':options,
                  '__MISSING__':html.escape(', '.join(missing)), '__DATA__':json.dumps(word_rows,ensure_ascii=False).replace('<','\\u003c')}
    for key,value in replacements.items():doc=doc.replace(key,value)
    (p/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps({'report':str(p/'report.html'),'counts':counts,'wrong_audio_trials':len(wrong)},ensure_ascii=False))


if __name__=='__main__':main()
