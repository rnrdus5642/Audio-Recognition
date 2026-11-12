'use strict';
const $ = (id) => document.getElementById(id);
let ready = false, historyData = null;

function message(text = '', error = false) {
  $('message').textContent = text;
  $('message').classList.toggle('error', error);
}

function syncControls() {
  const active = !!window.WordGame?.active;
  $('game-targets').disabled = active;
  $('age').disabled = active;
  window.WordGame?.syncControls();
}

async function pollStatus(){try{const r=await fetch('/api/status');if(!r.ok)throw Error('서버에 연결할 수 없습니다.');const s=await r.json();ready=s.ready;$('epoch-label').textContent=`${s.epoch_equivalents.toFixed(2)}에포크 ${s.provisional === false ? '최종 선택 모델' : '미리보기'}`;$('device-label').textContent=s.device==='cpu'?'CPU 추론':'GPU 추론';$('latency-note').textContent=s.device==='cpu'?'CPU 추론 기준':'GPU 추론 기준';$('model-fingerprint').textContent=`MODEL ${s.model_id.slice(0,12)}`;$('model-status').className=`status ${s.error?'failed':ready?'ready':''}`;$('model-status').lastElementChild.textContent=s.error?'모델 로드 실패':ready?'사용 준비 완료':'모델 준비 중';if(s.error)message(s.error,true);syncControls();if(!ready&&!s.error)setTimeout(pollStatus,2000);}catch(e){message('서버 연결을 확인하고 있습니다. 잠시 기다려 주세요.',true);setTimeout(pollStatus,4000);}}

function wavBlob(buffers,sampleRate){let size=buffers.reduce((n,b)=>n+b.length,0);const result=new ArrayBuffer(44+size*2),view=new DataView(result);function str(offset,s){for(let i=0;i<s.length;i++)view.setUint8(offset+i,s.charCodeAt(i));}str(0,'RIFF');view.setUint32(4,36+size*2,true);str(8,'WAVE');str(12,'fmt ');view.setUint32(16,16,true);view.setUint16(20,1,true);view.setUint16(22,1,true);view.setUint32(24,sampleRate,true);view.setUint32(28,sampleRate*2,true);view.setUint16(32,2,true);view.setUint16(34,16,true);str(36,'data');view.setUint32(40,size*2,true);let offset=44;for(const buffer of buffers){for(const x of buffer){const v=Math.max(-1,Math.min(1,x));view.setInt16(offset,v<0?v*32768:v*32767,true);offset+=2;}}return new Blob([result],{type:'audio/wav'});}

function showResult(item){
  $('empty-result').hidden=true; $('result').hidden=false;
  const p=item.prediction,s=item.score;
  $('prediction').textContent=p.text||'(인식된 문장 없음)';
  $('reference-result').hidden=!item.reference; $('reference-text').textContent=item.reference;
  $('cer').textContent=s?`${(s.cer*100).toFixed(1)}%`:'—';
  $('cer-note').textContent=s?`정답 ${s.reference_characters}글자 · 오류 ${s.errors}글자`:'정답을 입력하면 계산합니다.';
  $('latency').textContent=`${p.inference_seconds.toFixed(2)}초`;
  $('latency-note').textContent=`${item.device.toUpperCase()} 추론 기준`;
  $('duration').textContent=`${p.audio_seconds.toFixed(2)}초`;
  $('playback').src=item.audio_url; $('jamo').textContent=p.jamo||'(자모 출력 없음)';
  $('result-badge').textContent=s?(s.exact?'정답과 일치':'차이 있음'):'인식 완료';
  $('result-badge').className=`result-badge ${s?(s.exact?'good':'different'):''}`;
  $('game-result').hidden=!item.game;
  if(item.game){
    const game=item.game;
    $('result-badge').textContent={accepted:'정답 통과',retry:'재시도',stopped:'중단'}[game.outcome];
    $('result-badge').className=`result-badge ${game.outcome==='accepted'?'good':'different'}`;
    const word=game.matched_target||(game.outcome==='accepted'?item.reference:'');
    $('game-result-word').textContent=word?`선택된 단어: ${word}`:'확정된 단어 없음';
    $('game-result-candidates').textContent=game.candidate_scores?.map(candidate=>`${candidate.target} ${(candidate.similarity*100).toFixed(1)}점${candidate.blocked_by ? ` (${candidate.blocked_by}: 다른 동작)` : ''}`).join(' · ')||'';
    if(!s) $('cer-note').textContent='여러 후보 판정 · 정답 전사 미지정';
  }
}
async function loadHistory(){try{const r=await fetch('/api/history');if(!r.ok)return;historyData=await r.json();$('history-count').textContent=historyData.total;$('exact-count').textContent=historyData.scored_count?`${historyData.exact_count} / ${historyData.scored_count}`:'—';$('history-cer').textContent=historyData.corpus_cer===null?'—':`${(historyData.corpus_cer*100).toFixed(1)}%`;$('export').disabled=!historyData.total;const parent=$('history');parent.replaceChildren();if(!historyData.items.length){const empty=document.createElement('div');empty.className='history-empty';empty.textContent='첫 번째 음성을 테스트해 보세요. 결과와 녹음이 여기에 쌓입니다.';parent.append(empty);return;}for(const item of historyData.items){const button=document.createElement('button');button.className='history-item';const line=document.createElement('div');line.className='history-line';const title=document.createElement('strong');title.textContent=item.game?(item.game.matched_target||item.reference||'여러 후보 · 미확정'):(item.prediction.text||'(인식된 문장 없음)');const badge=document.createElement('span');badge.className='mini-badge';badge.textContent=item.game?({accepted:'정답 통과',retry:'재시도',stopped:'중단'}[item.game.outcome]):item.score?`${(item.score.cer*100).toFixed(1)}%`:'정답 없음';line.append(title,badge);const detail=document.createElement('p');const age=item.age==='adult'?'성인':item.age==='unknown'?'연령 미지정':`${item.age}세`;detail.textContent=`${new Date(item.created_at).toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'})} · ${age} · ${item.game?.settings?.targets?.join(', ')||item.reference||'정답 미입력'}`;button.append(line,detail);button.onclick=()=>{showResult(item);$('result-heading').scrollIntoView({behavior:'smooth',block:'center'});};parent.append(button);}}catch(e){console.warn('History unavailable',e);}}
$('export').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(historyData,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`speech-test-${new Date().toISOString().slice(0,10)}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};

pollStatus();
loadHistory();
