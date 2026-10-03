'use strict';
const $ = id => document.getElementById(id);
const state = { phase:'idle', ready:false, maxSeconds:15, stream:null, context:null, source:null, node:null,
  chunks:[], samples:0, rate:0, timer:null, rows:[], selected:null, stopAck:null };

async function api(url, options={}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '서버 연결을 확인해 주세요.');
  return body;
}
function error(message='') { $('error').textContent=message; $('error').hidden=!message; }
function controls() {
  const recording=state.phase==='recording', waiting=['requesting','stopping','recognizing'].includes(state.phase);
  $('record').disabled=waiting || (!recording && !state.ready);
  $('record').classList.toggle('recording',recording);
  $('record-label').textContent=recording?'녹음 끝내기':state.phase==='recognizing'?'인식 중…':state.phase==='requesting'?'마이크 연결 중…':'녹음 시작';
  $('record-state').textContent=recording?'말씀해 주세요':state.phase==='recognizing'?'녹음한 음성을 인식하고 있습니다':state.phase==='requesting'?'마이크 사용을 허용해 주세요':'녹음 준비';
}
function show(row) {
  state.selected=row.id;
  $('result-text').textContent=row.prediction.text || '인식된 텍스트가 없습니다';
  $('result-text').classList.toggle('empty',!row.prediction.text);
  $('result-meta').textContent=`녹음 ${row.seconds.toFixed(1)}초`;
  $('result-tools').hidden=false;
  $('playback').src=row.audio_url;
  $('download-audio').href=row.audio_url;
  $('download-audio').download=`음성-${row.id.slice(0,8)}.wav`;
  $('jamo').textContent=row.prediction.jamo || '출력된 자모가 없습니다';
  document.querySelectorAll('.history-item').forEach(button => button.setAttribute('aria-current',String(button.dataset.id===row.id)));
}
function history() {
  $('history').replaceChildren();
  $('history-count').textContent=state.rows.length?String(state.rows.length):'';
  $('history-empty').hidden=!!state.rows.length;
  $('export').disabled=!state.rows.length;
  for (const row of state.rows) {
    const li=document.createElement('li'), button=document.createElement('button');
    button.type='button'; button.className='history-item'; button.dataset.id=row.id;
    button.setAttribute('aria-current',String(row.id===state.selected));
    const text=document.createElement('strong'), time=document.createElement('small');
    text.textContent=row.prediction.text || '(인식된 텍스트 없음)';
    time.textContent=new Date(row.created_at).toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'})+` · ${row.seconds.toFixed(1)}초`;
    button.append(text,time); button.addEventListener('click',()=>show(row)); li.append(button); $('history').append(li);
  }
}
async function refreshStatus() {
  try {
    const result=await api('/api/status');
    if (result.instance.app!=='child-jamo-microphone') throw new Error('마이크 테스트 서버의 주소를 확인해 주세요.');
    state.ready=result.model.ready && !result.model.error;
    state.maxSeconds=result.max_seconds;
    $('model-status').textContent=state.ready?'현재 모델 · 준비 완료':result.model.error?'모델 연결 오류':'모델 준비 중';
    $('model-status').className='badge'+(state.ready?' ready':result.model.error?' failed':'');
    if (result.model.error) error(result.model.error);
  } catch (exc) { state.ready=false; $('model-status').textContent='서버 연결 확인 중'; $('model-status').className='badge failed'; }
  controls();
}
function meter(rms) {
  const value=Math.min(100,Math.max(0,((20*Math.log10(Math.max(rms,1e-5))+60)/60)*100));
  $('meter-fill').style.width=`${value}%`;
  document.querySelector('.meter').setAttribute('aria-valuenow',String(Math.round(value)));
}
function updateTimer() { $('timer').innerHTML=`${(state.samples/Math.max(1,state.rate)).toFixed(1)} <small>/ ${state.maxSeconds}초</small>`; }
async function release() {
  clearTimeout(state.timer); state.timer=null;
  state.stream?.getTracks().forEach(track=>track.stop());
  state.node?.disconnect(); state.source?.disconnect();
  if (state.context && state.context.state!=='closed') await state.context.close().catch(()=>{});
  state.stream=state.node=state.source=state.context=null;
  meter(0);
}
async function start() {
  if (state.phase!=='idle' || !state.ready) return;
  error(); state.phase='requesting'; controls();
  $('playback').pause();
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('이 PC의 Chrome 또는 Edge에서 http://127.0.0.1:8776 주소로 열어 주세요.');
    state.stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false},video:false});
    state.context=new AudioContext();
    await state.context.resume();
    await state.context.audioWorklet.addModule('/assets/capture.js');
    state.rate=state.context.sampleRate; state.chunks=[]; state.samples=0;
    state.node=new AudioWorkletNode(state.context,'microphone-capture');
    state.node.port.onmessage=({data})=>{
      if (data.stopped) { state.stopAck?.(); return; }
      if (!data.samples || !['recording','stopping'].includes(state.phase)) return;
      const remaining=Math.max(0,Math.floor(state.rate*state.maxSeconds)-state.samples);
      const part=data.samples.subarray(0,remaining);
      if (part.length) { state.chunks.push(part); state.samples+=part.length; }
      const rms=Math.sqrt(data.samples.reduce((sum,x)=>sum+x*x,0)/data.samples.length);
      meter(rms); updateTimer();
      if (state.samples>=state.rate*state.maxSeconds && state.phase==='recording') stop();
    };
    state.source=state.context.createMediaStreamSource(state.stream);
    state.phase='recording'; controls(); updateTimer();
    state.source.connect(state.node); state.node.connect(state.context.destination);
    // The worklet emits only zeros to the destination: no microphone feedback.
    state.stream.getAudioTracks()[0].addEventListener('ended',()=>{
      if (state.phase==='recording') { error('마이크 연결이 종료되어 녹음을 마쳤습니다.'); stop(); }
    });
    state.timer=setTimeout(()=>{ if (state.phase==='recording') stop(); },state.maxSeconds*1000+200);
  } catch (exc) {
    await release(); state.phase='idle'; controls();
    const messages={NotAllowedError:'마이크 권한이 꺼져 있습니다. 주소창 옆 사이트 설정에서 마이크를 허용한 뒤 다시 시작해 주세요.',NotFoundError:'연결된 마이크를 찾지 못했습니다.',NotReadableError:'마이크를 사용할 수 없습니다. 연결 상태와 다른 앱의 마이크 사용을 확인해 주세요.'};
    error(messages[exc.name] || exc.message);
  }
}
function wav() {
  const buffer=new ArrayBuffer(44+state.samples*2), view=new DataView(buffer);
  const word=(offset,value)=>{ for(let i=0;i<value.length;i++) view.setUint8(offset+i,value.charCodeAt(i)); };
  word(0,'RIFF'); view.setUint32(4,36+state.samples*2,true); word(8,'WAVE'); word(12,'fmt ');
  view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true);
  view.setUint32(24,state.rate,true); view.setUint32(28,state.rate*2,true); view.setUint16(32,2,true); view.setUint16(34,16,true);
  word(36,'data'); view.setUint32(40,state.samples*2,true);
  let offset=44;
  for (const part of state.chunks) for (const value of part) { const x=Math.max(-1,Math.min(1,value)); view.setInt16(offset,Math.round(x<0?x*32768:x*32767),true); offset+=2; }
  return new Blob([buffer],{type:'audio/wav'});
}
async function stop() {
  if (state.phase!=='recording') return;
  state.phase='stopping'; clearTimeout(state.timer); controls();
  try {
    await new Promise(resolve=>{
      const fallback=setTimeout(resolve,1000);
      state.stopAck=()=>{ clearTimeout(fallback); resolve(); };
      state.node.port.postMessage('stop');
    });
    state.stopAck=null;
    await release();
    if (state.samples/state.rate<.1) throw new Error('녹음이 너무 짧습니다. 단어를 말한 뒤 녹음을 끝내 주세요.');
    const form=new FormData(); form.append('file',wav(),'microphone.wav');
    state.phase='recognizing'; controls();
    $('result-text').textContent='인식하고 있습니다…'; $('result-text').classList.add('empty');
    $('result-tools').hidden=true; $('result-meta').textContent='';
    const row=await api('/api/transcribe',{method:'POST',body:form,signal:AbortSignal.timeout(120000)});
    state.rows=[row,...state.rows].slice(0,50); state.selected=row.id; history(); show(row);
    $('hint').textContent='다른 단어도 녹음해 보세요. 아래에서 이전 결과를 다시 볼 수 있습니다.';
  } catch (exc) {
    error(exc.name==='TimeoutError'?'인식 응답이 늦어지고 있습니다. 잠시 후 다시 시도해 주세요.':exc.message);
    if (state.phase==='recognizing') { $('result-text').textContent='인식하지 못했습니다. 다시 시도해 주세요.'; $('result-text').classList.add('empty'); }
  } finally { state.chunks=[]; state.phase='idle'; controls(); }
}
$('record').addEventListener('click',()=>state.phase==='recording'?stop():start());
$('export').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),items:state.rows},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob), link=document.createElement('a'); link.href=url; link.download='음성인식-테스트결과.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
});
window.addEventListener('pagehide',()=>{ state.stream?.getTracks().forEach(track=>track.stop()); state.context?.close().catch(()=>{}); });
refreshStatus(); setInterval(refreshStatus,3000);
api('/api/history').then(data=>{ state.rows=data.items; history(); if(state.rows.length) show(state.rows[0]); }).catch(()=>error('이전 기록을 불러오지 못했습니다. 서버 연결을 확인해 주세요.'));
