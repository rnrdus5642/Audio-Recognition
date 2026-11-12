'use strict';
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString('ko-KR');
const partitionNames = {train:'학습 후보', dev:'검증용', test:'시험용', review:'기존 보류 자료'};
const clarityNames = {clear:'또렷하게 들림', partial:'일부만 들림', unclear:'알아듣기 어려움', hold:'판단 보류'};
const state = {items:[], total:0, offset:0, limit:30, current:null, seed:31, listVersion:0, clipVersion:0, revealed:false, labelSeen:false, saving:false};
const filterIds = ['search','speaker','partition','status','clarity-filter','max-seconds','sort'];
const filterKeys = ['q','speaker','partition','status','clarity','max_seconds','sort'];
const store = {
  get(key, fallback=null) {try {return JSON.parse(localStorage.getItem('age3.'+key)) ?? fallback;} catch {return fallback;}},
  set(key,value) {try {localStorage.setItem('age3.'+key,JSON.stringify(value));} catch {toast('브라우저 임시 메모를 저장하지 못했습니다. 저장 버튼으로 기록을 남겨주세요.',true);}},
  remove(key) {try {localStorage.removeItem('age3.'+key);} catch {}}
};
let toastTimer;
function toast(message,error=false) {clearTimeout(toastTimer); $('toast').textContent=message; $('toast').classList.toggle('error',error); $('toast').hidden=false; toastTimer=setTimeout(()=>{$('toast').hidden=true;},error?6500:3000);}
async function api(path,body) {
  const response = await fetch(path,body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || '요청을 처리하지 못했습니다.');
  return data;
}
function filters() {return Object.fromEntries(filterIds.map((id,i)=>[filterKeys[i],$(id).value]));}
function rememberView() {store.set('view',{...filters(),offset:state.offset,seed:state.seed,lastId:state.current?.id});}
function body() {
  const c=state.current;
  return {id:c.id,audio_sha256:c.audio_sha256,revision:c.revision,
    clarity:document.querySelector('input[name=clarity]:checked')?.value || '',
    relation:$('relation').value,heard:$('heard').value,note:$('note').value,label_seen:state.labelSeen};
}
function draft() {
  if (!state.current || state.saving) return;
  store.set('draft.'+state.current.id,body());
  $('saved-state').textContent='임시 메모 · 아직 저장하지 않음';
}
function renderLabel() {
  if (!state.current) return;
  const visible=!$('blind').checked || state.revealed;
  if (visible) state.labelSeen=true;
  $('label-hidden').hidden=visible;
  $('label-text').hidden=!visible;
  $('label-text').textContent=visible?state.current.label_original:'';
  $('relation').disabled=!visible;
}
function renderList() {
  const fragment=document.createDocumentFragment();
  for (const [index,item] of state.items.entries()) {
    const row=document.createElement('button'); row.type='button'; row.className='clip-row'; row.dataset.id=item.id;
    const active=item.id===state.current?.id; row.classList.toggle('active',active); row.setAttribute('aria-current',active?'true':'false');
    const icon=document.createElement('span'); icon.className='row-icon'; icon.textContent=active?'▶':'▷'; icon.setAttribute('aria-hidden','true');
    const copy=document.createElement('span'); copy.className='row-copy';
    const title=document.createElement('strong'); title.textContent=$('blind').checked?`녹음 ${String(item.number).padStart(5,'0')}`:item.label_original;
    const meta=document.createElement('span'); meta.textContent=`화자 ${String(item.speaker_no).padStart(2,'0')} · ${item.seconds.toFixed(1)}초 · ${partitionNames[item.partition]}`;
    copy.append(title,meta); row.append(icon,copy);
    if (item.revision) {const done=document.createElement('span'); done.className='row-done'; done.textContent='✓'; done.title=clarityNames[item.clarity] || '검토 완료'; row.append(done);}
    row.addEventListener('click',()=>choose(index,$('autoplay').checked)); fragment.append(row);
  }
  if (!state.items.length) {const p=document.createElement('p');p.className='loading';p.textContent='조건에 맞는 녹음이 없습니다.';fragment.append(p);}
  $('clip-list').replaceChildren(fragment);
  $('match-count').textContent=fmt(state.total)+'개';
  $('page-number').textContent=state.total?`${Math.floor(state.offset/state.limit)+1} / ${Math.ceil(state.total/state.limit)}`:'0 / 0';
  $('previous-page').disabled=state.offset===0;
  $('next-page').disabled=state.offset+state.limit>=state.total;
  $('order-hint').textContent=$('sort').value==='diverse'?'여러 화자의 녹음을 번갈아 들어보세요.':'선택한 순서로 녹음을 보여줍니다.';
}
async function loadList({preferId=null,last=false,auto=false,keepCurrent=false}={}) {
  const version=++state.listVersion;
  const params=new URLSearchParams({...filters(),offset:state.offset,limit:state.limit,seed:state.seed});
  try {
    const result=await api('/api/clips?'+params);
    if (version!==state.listVersion) return;
    if (!result.items.length && result.total && state.offset) {
      state.offset=Math.floor((result.total-1)/state.limit)*state.limit;
      return loadList({preferId,last,auto,keepCurrent});
    }
    state.items=result.items;state.total=result.total;
    $('initial-loading').hidden=true;
    renderList(); rememberView();
    if (keepCurrent && state.current) return;
    if (!result.items.length) {
      ++state.clipVersion; $('player').pause();state.current=null;
      $('review-workspace').hidden=true;$('empty-state').hidden=false;return;
    }
    const found=state.items.findIndex(item=>item.id===preferId);
    await choose(found>=0?found:last?state.items.length-1:0,auto);
  } catch(error) {toast(error.message,true);$('initial-loading').textContent='자료를 불러오지 못했습니다. 새로고침해서 다시 시도해 주세요.';}
}
async function choose(index,auto=false) {
  if (state.saving || !state.items[index]) return;
  const version=++state.clipVersion;
  $('player').pause();
  try {
    const c=await api('/api/clip/'+state.items[index].id);
    if (version!==state.clipVersion) return;
    state.current=c;state.revealed=false;state.labelSeen=Boolean(c.label_seen);
    $('empty-state').hidden=true;$('review-workspace').hidden=false;
    $('clip-title').textContent=`녹음 ${String(c.number).padStart(5,'0')}`;
    $('clip-speaker').textContent=`화자 ${String(c.speaker_no).padStart(2,'0')} · 3세`;
    $('clip-duration').textContent=c.seconds.toFixed(2)+'초';
    $('partition-badge').textContent=partitionNames[c.partition];
    $('partition-badge').classList.toggle('review',c.partition==='review');
    $('original-file').textContent=c.record_id.replace(/^AIHUB_108:/,'');
    $('source-category').textContent='AIHub 자유대화 음성 · '+c.category;
    $('source-issues').textContent=c.issues.length?'기존 검토 대상: '+c.issues.map(issue=>({unresolved_non_hangul_notation:'라벨에 추가 표기 포함',audio_duration_mismatch:'라벨과 음성 길이 차이'}[issue] || issue)).join(', '):'자동 점검에서 보류 사유 없음 · 청취 품질 미확인';
    $('clip-details').open=false;
    const pending=store.get('draft.'+c.id);
    const values=pending && pending.revision===c.revision && pending.audio_sha256===c.audio_sha256?pending:c;
    if (values.label_seen) state.labelSeen=true;
    for(const input of document.querySelectorAll('input[name=clarity]')) input.checked=input.value===values.clarity;
    $('heard').value=values.heard || '';$('note').value=values.note || '';$('relation').value=values.relation || '';
    $('saved-state').textContent=values===pending?'임시 메모 불러옴 · 저장 필요':c.revision?'저장된 검토 기록':'아직 검토하지 않음';
    renderLabel();renderList();rememberView();
    $('player').src=c.audio_url;$('player').load();$('player').playbackRate=Number($('speed').value);
    $('wave-svg').replaceChildren();$('wave-loading').hidden=false;
    api('/api/waveform/'+c.id).then(data=>{
      if(version!==state.clipVersion)return;
      const max=Math.max(.01,...data.peaks);
      const fragment=document.createDocumentFragment();
      for(const [i,peak] of data.peaks.entries()) {
        const bar=document.createElementNS('http://www.w3.org/2000/svg','rect');
        const height=Math.max(2,peak/max*84);
        for(const [key,value] of Object.entries({x:i*6+1,y:(100-height)/2,width:3.5,height,rx:1.7}))bar.setAttribute(key,value);
        fragment.append(bar);
      }
      $('wave-svg').replaceChildren(fragment);$('wave-loading').hidden=true;
    }).catch(()=>{if(version===state.clipVersion)$('wave-loading').textContent='파형을 불러오지 못했습니다. 아래 재생 버튼을 이용하세요.';});
    if(auto) $('player').play().catch(()=>toast('재생 버튼을 눌러 소리를 들어주세요.'));
  } catch(error) {toast(error.message,true);}
}
async function refreshStats() {
  const stats=await api('/api/stats');
  $('total-clips').textContent=fmt(stats.total);$('total-speakers').textContent=fmt(stats.speakers);
  $('total-hours').textContent=(stats.seconds/3600).toFixed(1)+'시간';
  $('reviewed-count').textContent=fmt(stats.reviewed);$('review-progress').max=stats.total;$('review-progress').value=stats.reviewed;
  return stats;
}
async function step(direction) {
  if(state.saving)return;
  const index=state.items.findIndex(item=>item.id===state.current?.id);
  if(index<0)return choose(direction>0?0:state.items.length-1,$('autoplay').checked);
  const next=index+direction;
  if(next>=0 && next<state.items.length)return choose(next,$('autoplay').checked);
  if(direction>0 && state.offset+state.limit<state.total) {state.offset+=state.limit;await loadList({auto:$('autoplay').checked});}
  else if(direction<0 && state.offset>0) {state.offset-=state.limit;await loadList({last:true,auto:$('autoplay').checked});}
  else toast(direction>0?'현재 목록의 마지막 녹음입니다.':'현재 목록의 첫 녹음입니다.');
}
async function save(next=false) {
  if(!state.current || state.saving)return;
  const payload=body();
  if(!payload.clarity) {toast('들리는 정도를 하나 선택해 주세요.',true);return;}
  const nextItem=state.items[state.items.findIndex(c=>c.id===payload.id)+1];
  state.saving=true;$('save').disabled=true;$('save-next').disabled=true;
  try {
    const result=await api('/api/review',payload);
    store.remove('draft.'+payload.id);
    Object.assign(state.current,payload,{revision:result.revision});
    const listed=state.items.find(c=>c.id===payload.id);if(listed)Object.assign(listed,state.current);
    $('saved-state').textContent='저장 완료';renderList();toast('검토 기록을 저장했습니다.');
    await refreshStats();
    state.saving=false;
    if(next) {
      if($('status').value==='unreviewed' || $('clarity-filter').value) await loadList({preferId:nextItem?.id,auto:$('autoplay').checked});
      else await step(1);
    } else if($('status').value==='unreviewed' || $('clarity-filter').value) await loadList({keepCurrent:true});
  } catch(error) {toast(error.message,true);}
  finally {state.saving=false;$('save').disabled=false;$('save-next').disabled=false;}
}
let searchTimer;
for(const id of filterIds) {
  $(id).addEventListener(id==='search'?'input':'change',()=>{
    clearTimeout(searchTimer);searchTimer=setTimeout(()=>{state.offset=0;loadList();},id==='search'?300:0);
  });
}
$('reset-filters').addEventListener('click',()=>{for(const id of filterIds)$(id).value=id==='sort'?'diverse':'';state.offset=0;loadList();});
$('shuffle').addEventListener('click',()=>{state.seed=Math.floor(Math.random()*2147483647);state.offset=0;loadList();});
$('previous-page').addEventListener('click',()=>{state.offset=Math.max(0,state.offset-state.limit);loadList();});
$('next-page').addEventListener('click',()=>{state.offset+=state.limit;loadList();});
$('previous-clip').addEventListener('click',()=>step(-1));$('next-clip').addEventListener('click',()=>step(1));
$('save').addEventListener('click',()=>save());$('save-next').addEventListener('click',()=>save(true));
$('reveal-label').addEventListener('click',()=>{state.revealed=true;renderLabel();});
$('blind').checked=store.get('blind',true);
$('blind').addEventListener('change',()=>{store.set('blind',$('blind').checked);state.revealed=false;renderLabel();renderList();});
$('autoplay').checked=store.get('autoplay',false);
$('autoplay').addEventListener('change',()=>store.set('autoplay',$('autoplay').checked));
$('speed').addEventListener('change',()=>{$('player').playbackRate=Number($('speed').value);});
$('player').addEventListener('loadedmetadata',()=>{$('player').playbackRate=Number($('speed').value);});
$('player').addEventListener('error',()=>{if(state.current)toast('오디오를 재생하지 못했습니다. 다음 녹음을 선택하거나 페이지를 새로고침해 주세요.',true);});
$('player').addEventListener('timeupdate',()=>{
  const fraction=$('player').currentTime/($('player').duration || 1);
  Array.from($('wave-svg').children).forEach((bar,i)=>bar.classList.toggle('played',i/120<fraction));
});
$('waveform').addEventListener('click',event=>{if(Number.isFinite($('player').duration)){const rect=$('waveform').getBoundingClientRect();$('player').currentTime=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*$('player').duration;}});
for(const element of document.querySelectorAll('input[name=clarity],#heard,#note,#relation'))element.addEventListener('input',draft);
document.addEventListener('keydown',event=>{
  if(event.ctrlKey && event.key==='Enter'){event.preventDefault();save(true);return;}
  if(/INPUT|TEXTAREA|SELECT|BUTTON|AUDIO/.test(document.activeElement.tagName))return;
  if(event.code==='Space' && state.current){event.preventDefault();if($('player').paused)$('player').play().catch(()=>toast('재생 버튼을 눌러주세요.'));else $('player').pause();}
});
async function init() {
  try {
    const stats=await refreshStats();
    for(const speaker of stats.speaker_options){const option=document.createElement('option');option.value=speaker.id;option.textContent=`화자 ${String(speaker.id).padStart(2,'0')} · ${fmt(speaker.count)}개`;$('speaker').append(option);}
    const view=store.get('view',{});
    filterIds.forEach((id,i)=>{if(view[filterKeys[i]]!==undefined)$(id).value=view[filterKeys[i]];});
    state.seed=Number.isInteger(view.seed)?view.seed:31;state.offset=Number.isInteger(view.offset)?Math.max(0,view.offset):0;
    await loadList({preferId:view.lastId});
  } catch(error) {toast(error.message,true);$('initial-loading').textContent='청취 서버에 연결하지 못했습니다. 실행 바로가기로 서버를 다시 열어주세요.';}
}
init();
