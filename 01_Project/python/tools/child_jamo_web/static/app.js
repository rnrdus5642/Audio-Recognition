'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  split: 'test', page: 1, pages: 1, selected: null,
  // One counter for every asynchronous answer that can land on screen.
  // Each operation captures it at the start and drops its own reply if
  // the counter has moved on - otherwise a slow upload or batch can
  // repaint the panel under a sample the user has already left.
  generation: 0,
  listGeneration: 0,     // separate: filter replies can race each other
  queued: null,          // the newest selection still awaiting inference
  attempt: 0,            // counts real inference requests, for the UI
  listItems: [], mode: null,
  splits: {},            // what /api/status said about every dataset
  sources: {},           // the two places recordings come from
  source: 'aihub',       // which one this page is showing
  sample: null,          // the selected row, for the transcribe dispatch
};

function nextGeneration() {
  state.generation += 1;
  return state.generation;
}

/* Everything that renders a result clears whatever mode was showing. */
function clearResults() {
  $('result').hidden = true;
  // Identity of whatever is painted, so the rendered panel can be checked
  // against the current selection instead of being read off its text.
  $('result').dataset.sampleId = '';
  $('result').dataset.generation = '';
  $('result').dataset.attempt = '';
  $('batch-result').hidden = true;
  $('batch-result').dataset.generation = '';
  $('batch-rows').replaceChildren();
  $('batch-aggregate').replaceChildren();
  $('original-result').hidden = true;
  $('original-result').dataset.sampleId = '';
  $('original-segments').replaceChildren();
  $('original-coverage').replaceChildren();
  $('diff').replaceChildren();
  setText($('reference-note'), '');
  note($('upload-note'), '');
}

/* ---------- helpers ---------- */

function setText(node, value) { node.textContent = value == null ? '' : String(value); }

function note(node, message, isError) {
  setText(node, message || '');
  node.classList.toggle('error', Boolean(isError));
}

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `요청이 실패했습니다 (${response.status})`;
    try { const body = await response.json(); if (body.detail) detail = body.detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

/* Nobody wrote down how old the children in the personal recordings are,
 * so the page says so rather than showing a number it does not have. */
function ageLabel(row) { return row.age === 'unknown' ? '나이 미상' : `${row.age}세`; }

const STYLE_NAMES = { free: '자유발화', word: '단어 목록', formatted: '낭독' };
function styleLabel(row) { return STYLE_NAMES[row.style] || row.style; }

const CHILD_NAMES = { child1: '아동 1', child2: '아동 2' };
function childLabel(row) { return CHILD_NAMES[row.child] || row.speaker; }

function splitLabel(split) {
  const meta = state.splits[split];
  return meta ? meta.label : split;
}

/* ---------- which source this page is ---------- */

/* The path decides, because the path is what the user sees, links and
 * refreshes. `/?split=personal` predates the split into two pages and is
 * still a link someone may have kept, so it is honoured by resolving to
 * the source that split belongs to - and then canonicalised, so the
 * address bar afterwards says where they actually are. */
function resolveSource(sources, splits) {
  for (const [name, meta] of Object.entries(sources)) {
    if (meta.path === location.pathname) return name;
  }
  const wanted = new URLSearchParams(location.search).get('split');
  if (wanted && splits[wanted]) return splits[wanted].source;
  return 'aihub';
}

/* The dataset to open on. A split belonging to the other source is not an
 * error to show the other source's data for - `/aihub?split=personal`
 * opens this source's default instead, never the personal recordings. */
function resolveSplit(source, splits, fallback) {
  const wanted = new URLSearchParams(location.search).get('split');
  if (wanted && splits[wanted] && splits[wanted].source === source) return wanted;
  return fallback;
}

/* Keep the address bar saying what is on screen, without pushing a new
 * entry per dropdown change - the back button belongs to the two pages. */
function syncUrl() {
  const query = new URLSearchParams({ split: state.split });
  if (!$('child-filter').hidden && $('child').value) {
    query.set('child', $('child').value);
  }
  const url = `${state.sources[state.source].path}?${query}`;
  if (url !== location.pathname + location.search) {
    history.replaceState(null, '', url);
  }
}

function seconds(value) {
  const whole = Math.floor(value);
  return `${String(Math.floor(whole / 60)).padStart(2, '0')}:` +
         `${String(whole % 60).padStart(2, '0')}`;
}

/* Predicted text, with uncomposable jamo visibly marked rather than hidden. */
function renderPrediction(node, prediction) {
  node.replaceChildren();
  if (!prediction.segments.length) { setText(node, '(예측 없음)'); return; }
  for (const segment of prediction.segments) {
    const span = document.createElement('span');
    span.textContent = segment.text;
    if (segment.kind === 'orphan') {
      span.className = 'orphan';
      span.title = `음절로 합쳐지지 않은 자모: ${segment.jamo.join('')}`;
    }
    node.appendChild(span);
  }
}

/* ---------- model status ---------- */

async function refreshStatus() {
  const box = $('model-status');
  try {
    const status = await api('/api/status');
    const model = status.model;
    if (model.ready) {
      box.className = 'status status-ready';
      setText($('model-status-text'), '모델 준비 완료');
      state.model = model;
      renderModelFacts();
    } else if (model.error) {
      box.className = 'status status-error';
      setText($('model-status-text'), '모델 오류');
    } else {
      box.className = 'status status-loading';
      setText($('model-status-text'), '모델 불러오는 중…');
    }
    return status;
  } catch (err) {
    box.className = 'status status-error';
    setText($('model-status-text'), '서버에 연결할 수 없습니다');
    throw err;
  }
}

/* The recorded corpus score belongs to the AIHub test split, so it is
 * printed on the AIHub page and nowhere else. The personal recordings page
 * says where that number lives instead of repeating it: a percentage shown
 * beside these recordings would be read as theirs however it is worded,
 * and there is no aggregate for them to put in its place. */
function renderModelFacts() {
  const model = state.model;
  // Which source this page shows decides what may be printed here, so
  // nothing is printed until that is settled - otherwise the corpus score
  // flashes up on the personal recordings page before it is replaced.
  if (!model || !Object.keys(state.sources).length) return;
  const test = model.historical_test;
  const personal = state.source === 'recordings';
  const facts = document.createElement('div');
  const line1 = document.createElement('p');
  line1.style.margin = '0 0 4px';
  line1.append(document.createTextNode('사용 모델: '));
  const b = document.createElement('b');
  b.textContent = 'HuBERT XLarge 한국어 + 자모 CTC (직접 미세조정)';
  line1.append(b, document.createTextNode(
    ` · ${(model.parameters / 1e6).toFixed(0)}M 파라미터 · ${model.device} · ${model.precision}`));
  facts.appendChild(line1);
  const line2 = document.createElement('p');
  line2.style.margin = '0';
  if (personal) {
    line2.textContent =
      '이 모델의 기록된 전체 성적은 말뭉치(AIHub) 테스트 세트에 대한 것이라 ' +
      'AIHub 데이터 페이지에만 적어 둡니다. 직접 녹음한 이 음성들의 정확도가 ' +
      '아니고, 이 음성들의 결과도 그 점수에 들어가 있지 않습니다. 여기서는 ' +
      '녹음 하나하나의 결과만 보여 드립니다.';
    facts.appendChild(line2);
  } else if (test) {
    line2.textContent =
      `참고: 이 모델의 전체 테스트 성적은 자모 오류율 ${(test.jer * 100).toFixed(2)}%, ` +
      `${test.recordings.toLocaleString()}개 중 ${test.exact.toLocaleString()}개 완전 일치입니다. ` +
      `화면에 보이는 개별 결과와는 별개의 과거 집계입니다.`;
    facts.appendChild(line2);
  }
  $('model-facts').replaceChildren(facts);
}

/* ---------- browsing ---------- */

/* Only the filters the chosen dataset actually offers.
 *
 * Age belongs to the corpus rows and the child to the personal ones, so
 * each is hidden for the other dataset. A hidden one must also stop being
 * applied: leaving an invisible `age=5` in place made the personal clips
 * - which have no age - come up as an empty list with nothing on screen
 * to explain why. The select keeps its value, so going back to the corpus
 * restores a filter the user can see and undo. */
function filters() {
  const out = {
    split: state.split, q: $('query').value.trim(),
    bucket: $('bucket').value, style: $('style').value,
  };
  if (!$('age-filter').hidden) out.age = $('age').value;
  if (!$('child-filter').hidden) out.child = $('child').value;
  return out;
}

/* The corpus rows carry an age; the personally recorded ones carry a child.
 * Only the filter that means something for the chosen dataset is offered. */
function applySplitChrome() {
  const meta = state.splits[state.split] || {};
  const personal = state.source === 'recordings';
  // Hiding is what takes a filter out of `filters()`, so these two lines
  // are also what stops the other dataset's filter from being applied.
  $('child-filter').hidden = !personal;
  $('age-filter').hidden = Boolean(personal);
  $('batch').hidden = meta.batchable === false;
  const note = $('dataset-note');
  setText(note, personal ? (state.split === 'original' ? ORIGINAL_NOTE : PERSONAL_NOTE) : '');
  note.hidden = !personal;
  renderModelFacts();
  syncUrl();
}

const PERSONAL_NOTE =
  '직접 녹음해 오신 아동 음성입니다. 원문은 함께 주신 한글 전사에서 그대로 ' +
  '옮긴 것이고, 구간 경계는 전사 순서와 음량으로 맞춘 것이라 아직 전부에 대해 ' +
  '사람이 들어 보는 청취 검수를 마치지 않았습니다. 발음 진단용 전문가 표기가 ' +
  '아닙니다.';
const ORIGINAL_NOTE =
  '받은 녹음 파일 그대로입니다. 전체에 맞춰진 전사가 없어 정답과 비교하지 ' +
  '않고, 모델이 들은 그대로를 구간별로만 보여 드립니다.';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* Resolves true only once the list holds this split's own rows.
 *
 * The train split is parsed in the background and answers 202 until it is
 * ready. Returning early on that left the previous split's rows in
 * `listItems`, and the caller then auto-selected a recording from the
 * split the user had just left. So the old rows are cleared immediately
 * and the caller waits here instead. */
async function loadList(page) {
  const token = ++state.listGeneration;
  state.page = page || 1;
  setText($('result-count'), '불러오는 중…');
  let found;
  for (;;) {
    const params = new URLSearchParams({ ...filters(), page: state.page, size: 24 });
    found = await api(`/api/samples?${params}`);
    // A slower reply for an older filter must not replace a newer one.
    if (token !== state.listGeneration) return false;
    if (!found.loading) break;
    state.listItems = [];
    setText($('result-count'), '데이터셋을 준비하는 중입니다…');
    showListMessage('목록을 준비하는 중입니다…');
    await sleep(1000);
    if (token !== state.listGeneration) return false;
  }
  state.pages = found.pages;
  state.listItems = found.items;
  setText($('result-count'), `${found.total.toLocaleString()}개`);
  setText($('page-label'), `${found.page} / ${found.pages}`);
  $('prev-page').disabled = found.page <= 1;
  $('next-page').disabled = found.page >= found.pages;

  const list = $('sample-list');
  list.replaceChildren();
  for (const item of found.items) {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    // Keyed by id: several recordings share a transcript, and matching on
    // the text highlighted all of them at once.
    button.dataset.id = item.id;
    button.setAttribute('aria-current', String(state.selected === item.id));
    const line1 = document.createElement('span');
    line1.className = 'line1';
    line1.textContent = item.text || (item.kind === 'personal_original'
      ? item.wav : '(전사 없음)');
    const line2 = document.createElement('span');
    line2.className = 'line2';
    line2.textContent = (item.child ? `${childLabel(item)} · ` : `${ageLabel(item)} · `)
      + `${styleLabel(item)} · ${item.seconds.toFixed(2)}초`;
    button.append(line1, line2);
    // Selecting transcribes straight away: the point of the page is to
    // hear a clip and see what the model made of it, not to press twice.
    button.addEventListener('click', () => selectSample(item.id, true));
    li.appendChild(button);
    list.appendChild(li);
  }
  if (!found.items.length) showListMessage('조건에 맞는 음성이 없습니다.');
  return true;
}

function showListMessage(text) {
  const list = $('sample-list');
  list.replaceChildren();
  const li = document.createElement('li');
  li.style.padding = '14px 8px';
  li.style.color = '#666d78';
  li.textContent = text;
  list.appendChild(li);
}

/* Clears the playing clip and everything the page was saying about it.
 *
 * Every line here belongs to one catalogued recording: its title, its
 * source MP3 and review flags, the neighbours to step to, and the button
 * label that says whether it is recognised whole or in chunks. They go
 * together, because leaving any one of them behind attaches one
 * recording's provenance to a different recording's audio. */
function clearSelection() {
  state.selected = null;
  state.sample = null;
  clearResults();
  setText($('sample-source'), '');
  $('sample-source').hidden = true;
  const player = $('player');
  player.pause();
  player.removeAttribute('src');
  player.load();
  setText($('sample-title'), '음성을 고르는 중…');
  setText($('sample-meta'), '');
  setText($('audio-hint'), '');
  setText($('orphan-note'), '');
  $('transcribe').textContent = '이 음성 인식하기';
  $('prev-sample').disabled = true;
  $('next-sample').disabled = true;
  $('prev-sample').dataset.id = '';
  $('next-sample').dataset.id = '';
}

/* ---------- selection ---------- */

async function selectSample(sampleId, autoTranscribe) {
  const token = nextGeneration();         // any in-flight result is now stale
  state.selected = sampleId;
  state.mode = 'sample';
  clearResults();
  note($('action-note'), '');
  for (const button of $('sample-list').querySelectorAll('button')) {
    button.setAttribute('aria-current', 'false');
  }

  const sample = await api(`/api/sample/${encodeURIComponent(sampleId)}`);
  if (token !== state.generation) return;
  state.sample = sample;

  const original = sample.kind === 'personal_original';
  setText($('sample-title'), sample.text || (original ? sample.wav : '(전사 없음)'));
  setText($('sample-meta'),
    `${splitLabel(sample.split)} · ` +
    (sample.child ? `${childLabel(sample)} · ${ageLabel(sample)} · ` : `${ageLabel(sample)} · `) +
    `${styleLabel(sample)} · ${sample.seconds.toFixed(2)}초 · 화자 ${sample.speaker} · ${sample.wav}`);
  renderSource(sample);
  $('transcribe').textContent = original
    ? '원본 전체를 구간별로 인식하기' : '이 음성 인식하기';
  const player = $('player');
  player.src = sample.audio_url;
  player.load();
  setText($('audio-hint'), original
    ? `녹음 전체가 그대로 재생됩니다 (16kHz 모노, ${seconds(sample.seconds)}). ` +
      `아래 구간 시각은 이 재생본의 시각과 같습니다.`
    : `재생되는 소리는 모델에 들어가는 신호와 같습니다 (16kHz 모노, ${sample.seconds.toFixed(2)}초).`);
  $('prev-sample').disabled = !sample.previous;
  $('next-sample').disabled = !sample.next;
  $('prev-sample').dataset.id = sample.previous ? sample.previous.id : '';
  $('next-sample').dataset.id = sample.next ? sample.next.id : '';

  for (const button of $('sample-list').querySelectorAll('button')) {
    button.setAttribute('aria-current', String(button.dataset.id === sampleId));
  }
  if (autoTranscribe) await requestTranscription();
}

/* Where a personally recorded clip came from, and how sure its label is. */
function renderSource(sample) {
  const box = $('sample-source');
  const source = sample.source;
  if (!source) { box.hidden = true; setText(box, ''); return; }
  const parts = [`출처: ${source.dataset}`, `원본 파일 ${source.file}`];
  if (source.start_s != null) {
    parts.push(`원본 ${source.start_s.toFixed(2)}~${source.end_s.toFixed(2)}초 구간`);
  }
  if (source.kind) parts.push(source.kind);
  parts.push(source.human_reviewed ? '청취 검수 완료' : '청취 검수 전');
  if (source.priority_review) parts.push('우선 검수 대상');
  if (source.expected_answer) {
    // The elicitation's target answer. Recorded as an annotation because it
    // is not what the child was transcribed as saying.
    parts.push(`질문의 기대 정답은 "${source.expected_answer}" (전사 라벨과 별개)`);
  }
  if (source.note) parts.push(source.note);
  setText(box, parts.join(' · '));
  box.hidden = false;
}

/* ---------- inference ---------- */

/* Latest-only scheduling.
 *
 * Only one inference runs at a time, but a selection made while one is in
 * flight must still end up transcribed. A plain "busy, give up" guard
 * dropped it: the first reply was discarded as stale and the second was
 * never requested, so choosing B during A's inference left B with no
 * result at all - contradicting the documented select-to-transcribe UX.
 *
 * So a request is queued rather than refused, and when the running one
 * settles the loop picks up whatever the newest selection is by then.
 * Intermediate selections are skipped on purpose; the user only ever
 * wants the one they are looking at.
 */
let running = null;

function requestTranscription() {
  state.queued = state.generation;
  if (running) return running;
  running = (async () => {
    try {
      while (state.queued !== null) {
        const token = state.queued;
        state.queued = null;
        if (token !== state.generation || !state.selected) continue;
        await runTranscription(token, state.selected);
      }
    } finally {
      running = null;
      $('transcribe').disabled = false;
    }
  })();
  return running;
}

/* The click handler: a real click must not pass its MouseEvent along as a
 * generation, which is what silently threw away every manual result. */
function transcribe() {
  if (!state.selected) return;
  return requestTranscription();
}

async function runTranscription(token, sampleId) {
  if (state.sample && state.sample.id === sampleId
      && state.sample.kind === 'personal_original') {
    return runOriginal(token, sampleId);
  }
  const attempt = ++state.attempt;
  $('transcribe').disabled = true;
  note($('action-note'), '인식하는 중입니다…');
  try {
    const result = await api('/api/transcribe', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId,
                             request_id: `${token}:${attempt}` }),
    });
    // A reply for a sample the user has already moved away from is dropped.
    if (token !== state.generation || result.sample_id !== state.selected) return;
    showResult(result);
    $('result').dataset.sampleId = result.sample_id;
    $('result').dataset.generation = String(token);
    // Bumped on every real request, so a repeat of the same sample is
    // visibly a new answer even when the text comes out identical.
    $('result').dataset.attempt = String(attempt);
    note($('action-note'),
      `${result.prediction.inference_seconds.toFixed(2)}초 만에 인식했습니다` +
      (result.prediction.cached ? ' (저장된 결과 재사용)' : '') + '.');
  } catch (err) {
    if (token === state.generation) note($('action-note'), err.message, true);
  }
}

function showResult(result) {
  const reference = result.reference;
  const metrics = result.metrics;
  setText($('reference-note'),
    result.reference_note ? result.reference_note.text : '');
  setText($('ref-transcript'), reference ? reference.transcript : '(전사 없음)');
  setText($('ref-surface'), reference ? reference.surface : '(비교할 목표가 없습니다)');
  renderPrediction($('pred-text'), result.prediction);

  const box = $('metrics');
  box.replaceChildren();
  // With no reference there is no rate. It says so; it does not say 0%.
  const cards = metrics ? [
    [`${(metrics.jer * 100).toFixed(1)}%`, '자모 오류율 (띄어쓰기 제외)'],
    [`${metrics.errors} / ${metrics.reference_jamo}`,
     '편집 횟수 / 예상 발음 자모 수'],
    [metrics.exact ? '완전 일치' : '차이 있음', '예상 발음 자모와 비교'],
    [`${result.prediction.seconds.toFixed(2)}초`, '인식한 음성 길이'],
  ] : [
    ['비교 불가', '비교할 전사가 없습니다'],
    [`${result.prediction.syllables}자`, '합쳐진 음절 수'],
    [`${result.prediction.seconds.toFixed(2)}초`, '인식한 음성 길이'],
  ];
  for (const [value, label] of cards) {
    const card = document.createElement('div');
    card.className = 'metric';
    const b = document.createElement('b');
    b.textContent = value;
    if (label === '예상 발음과 비교') b.className = metrics.exact ? 'ok' : 'no';
    const span = document.createElement('span');
    span.textContent = label;
    card.append(b, span);
    box.appendChild(card);
  }

  setText($('ref-jamo'), reference ? reference.jamo : '-');
  setText($('pred-jamo'), result.prediction.jamo.replace(/ /g, ' '));

  const diff = $('diff');
  diff.replaceChildren();
  for (const op of (result.alignment || [])) {
    const span = document.createElement('span');
    span.className = `op-${op.op}`;
    if (op.op === 'equal') span.textContent = op.reference.join('');
    else if (op.op === 'replace') span.textContent = `${op.reference.join('')}→${op.hypothesis.join('')}`;
    else if (op.op === 'delete') span.textContent = `${op.reference.join('')}→(없음)`;
    else span.textContent = `(추가)→${op.hypothesis.join('')}`;
    span.title = { equal: '같음', replace: '다르게 인식', delete: '빠뜨림', insert: '덧붙임' }[op.op];
    diff.appendChild(span);
  }
  setText($('orphan-note'), result.prediction.orphan_jamo
    ? `음절로 합쳐지지 않은 자모 ${result.prediction.orphan_jamo}개는 빨간색으로 그대로 표시했습니다.`
    : '');
  $('result').hidden = false;
  loadHistory();
}

/* ---------- whole recordings ---------- */

/* A long recording is recognised in bounded chunks and reported chunk by
 * chunk. The audio element keeps playing the whole file; a segment button
 * only seeks it, so what is heard is always the original, not a cut. */
async function runOriginal(token, sampleId) {
  const attempt = ++state.attempt;
  $('transcribe').disabled = true;
  const planned = state.sample && state.sample.chunks
    ? state.sample.chunks.count : '';
  note($('action-note'),
    `녹음 전체를 ${planned}개 구간으로 나눠 인식하는 중입니다. 조금 걸립니다…`);
  try {
    const result = await api('/api/transcribe-original', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id: sampleId,
                             request_id: `${token}:${attempt}` }),
    });
    if (token !== state.generation || result.sample_id !== state.selected) return;
    showOriginal(result);
    $('original-result').dataset.sampleId = result.sample_id;
    $('original-result').dataset.generation = String(token);
    $('original-result').dataset.attempt = String(attempt);
    note($('action-note'),
      `${result.coverage.segments}개 구간, ${result.coverage.inference_seconds.toFixed(1)}초 만에 ` +
      `인식했습니다 (전체 ${seconds(result.coverage.total_samples / 16000)}).`);
  } catch (err) {
    if (token === state.generation) note($('action-note'), err.message, true);
  }
}

function showOriginal(result) {
  setText($('original-note'),
    `${result.reference_note.text} 구간은 12초 안팎으로 나누되 경계는 가장 ` +
    `조용한 지점으로 옮겼습니다. 모든 샘플이 빠짐없이 한 번씩 들어갑니다.`);

  const coverage = result.coverage;
  const box = $('original-coverage');
  box.replaceChildren();
  // Durations, not sample counters: this is the number the listener can
  // check against the player. The exact sample counts stay in the API
  // response and the validation record, where they are the proof.
  const whole = coverage.total_samples / 16000;
  const done = coverage.covered_samples / 16000;
  const cards = [
    ['비교 불가', '비교할 전사가 없습니다'],
    [coverage.complete ? '전체 포함' : '누락 있음',
     coverage.complete ? `녹음 전체 ${seconds(whole)} 를 빠짐없이 인식`
                       : `인식 ${seconds(done)} / 전체 ${seconds(whole)}`],
    [`${coverage.segments}개`, `구간 (최대 ${coverage.max_seconds.toFixed(1)}초)`],
  ];
  for (const [value, label] of cards) {
    const card = document.createElement('div');
    card.className = 'metric';
    const b = document.createElement('b'); b.textContent = value;
    const s = document.createElement('span'); s.textContent = label;
    card.append(b, s); box.appendChild(card);
  }

  const list = $('original-segments');
  list.replaceChildren();
  for (const segment of result.segments) {
    const row = document.createElement('div');
    row.className = 'segment' + (segment.boundary_cuts_speech ? ' cut' : '');
    const at = document.createElement('div');
    at.className = 'at';
    at.textContent = `${seconds(segment.start_s)} – ${seconds(segment.end_s)}`;
    const said = document.createElement('div');
    said.className = 'said';
    renderPrediction(said, segment.prediction);
    if (segment.boundary_cuts_speech) {
      const warn = document.createElement('span');
      warn.className = 'warn';
      warn.textContent = '이 구간 끝은 조용한 곳이 아니라 말 중간에서 ' +
        '잘렸을 수 있습니다.';
      said.appendChild(warn);
    }
    const play = document.createElement('button');
    play.type = 'button';
    play.className = 'ghost';
    play.style.cssText = 'padding:2px 10px;font-size:12px';
    play.textContent = '여기서 듣기';
    play.dataset.at = String(segment.start_s);
    play.addEventListener('click', () => {
      const player = $('player');
      player.currentTime = segment.start_s;
      player.play().catch(() => {});
    });
    row.append(at, said, play);
    list.appendChild(row);
  }
  $('original-result').hidden = false;
  loadHistory();
}

/* ---------- batch ---------- */

async function runBatch() {
  if (!state.selected || $('batch').disabled) return;
  const token = nextGeneration();
  state.mode = 'batch';
  clearResults();
  $('batch').disabled = true;
  try {
    // The server resolves "the next ten from here" inside the current
    // filter, so the scope does not depend on which page is displayed.
    const params = new URLSearchParams({ ...filters(), sample_id: state.selected,
                                         count: 10 });
    const scope = await api(`/api/batch-scope?${params}`);
    if (token !== state.generation) return;
    if (!scope.count) { note($('action-note'), '인식할 음성이 없습니다.', true); return; }
    note($('action-note'), `선택한 음성부터 ${scope.count}개를 인식하는 중입니다…`);
    const result = await api('/api/batch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_ids: scope.items.map((item) => item.id),
                             request_id: String(token) }),
    });
    if (token !== state.generation) return;   // a newer action already won
    if (result.all_failed) {
      note($('action-note'),
        `${result.requested}개 모두 인식하지 못했습니다: ` +
        result.failures.map((f) => f.reason).slice(0, 2).join(' / '), true);
      return;
    }
    const aggregate = $('batch-aggregate');
    aggregate.replaceChildren();
    const cards = [
      // Null stays null: no recording, no rate.
      [result.aggregate.jer === null ? '-'
        : `${(result.aggregate.jer * 100).toFixed(1)}%`,
       '이번 묶음의 자모 오류율 (띄어쓰기 제외)'],
      [`${result.aggregate.exact} / ${result.aggregate.scored}`,
       '예상 발음과 완전히 같은 개수'],
      [`${result.aggregate.recordings} / ${result.requested}`, '인식 성공 / 요청'],
    ];
    // Recognised but with nothing to compare against: named, not averaged in.
    if (result.aggregate.unscored) {
      cards.push([`${result.aggregate.unscored}개`, '비교할 전사가 없어 제외']);
    }
    for (const [value, label] of cards) {
      const card = document.createElement('div');
      card.className = 'metric';
      const b = document.createElement('b'); b.textContent = value;
      const s = document.createElement('span'); s.textContent = label;
      card.append(b, s); aggregate.appendChild(card);
    }
    const rows = $('batch-rows');
    rows.replaceChildren();
    for (const item of result.results) {
      const row = document.createElement('div');
      row.className = 'batch-row';
      const head = document.createElement('div');
      const strong = document.createElement('strong');
      strong.textContent = item.sample.text || '(전사 없음)';
      const badge = document.createElement('span');
      badge.className = item.metrics ? (item.metrics.exact ? 'ok' : 'no') : '';
      badge.textContent = !item.metrics ? '  비교 불가'
        : item.metrics.exact ? '  완전 일치'
        : `  오류율 ${(item.metrics.jer * 100).toFixed(1)}%`;
      const listen = document.createElement('button');
      listen.type = 'button';
      listen.className = 'ghost';
      listen.style.cssText = 'margin-left:8px;padding:2px 10px;font-size:12px';
      listen.textContent = '이 음성 열기';
      listen.addEventListener('click', () => selectSample(item.sample_id, true));
      head.append(strong, badge, listen);
      const pair = document.createElement('div');
      pair.className = 'pair';
      for (const [key, value] of [['예상 발음 표기',
                                   item.reference ? item.reference.surface
                                                  : '(전사 없음)'],
                                  ['모델 인식', item.prediction.text]]) {
        const cell = document.createElement('div');
        const k = document.createElement('div'); k.className = 'k'; k.textContent = key;
        const v = document.createElement('div'); v.textContent = value;
        cell.append(k, v); pair.appendChild(cell);
      }
      row.append(head, pair);
      rows.appendChild(row);
    }
    $('batch-result').hidden = false;
    $('batch-result').dataset.generation = String(token);
    note($('action-note'),
      `${result.results.length}개 인식을 마쳤습니다.` +
      (result.failures.length ? ` ${result.failures.length}개 실패.` : ''),
      Boolean(result.failures.length));
  } catch (err) {
    if (token === state.generation) note($('action-note'), err.message, true);
  } finally {
    $('batch').disabled = false;
  }
}

/* ---------- upload ---------- */

async function runUpload() {
  const input = $('file');
  if (!input.files || !input.files.length) {
    note($('upload-note'), '파일을 먼저 고르세요.', true);
    return;
  }
  const form = new FormData();
  form.append('file', input.files[0]);
  form.append('reference', $('file-reference').value.trim());
  // Claim the screen now, so a sample selected while this is in flight
  // wins and this reply is discarded instead of overwriting it.
  const token = nextGeneration();
  state.mode = 'upload';
  // The uploaded file is not a catalogued recording, so the catalogued
  // one's provenance goes before this one's name is on screen. Otherwise
  // an upload made after a personal clip inherited that child's source
  // MP3, review flags and expected-answer note.
  clearSelection();
  setText($('sample-title'), input.files[0].name);
  $('upload').disabled = true;
  note($('upload-note'), '업로드하고 인식하는 중입니다…');
  try {
    const result = await api('/api/upload', { method: 'POST', body: form });
    if (token !== state.generation) return;
    setText($('sample-title'), result.filename);
    setText($('sample-meta'),
      `내가 올린 파일 · ${result.seconds.toFixed(2)}초 · ` +
      `원본 ${result.original_sample_rate}Hz → 16kHz 모노`);
    $('player').src = result.audio_url;
    $('player').load();
    setText($('audio-hint'), `재생되는 소리는 모델에 들어가는 신호와 같습니다 (16kHz 모노).`);
    setText($('ref-transcript'), result.reference ? result.reference.transcript : '(원문 없음)');
    setText($('ref-surface'), result.reference ? result.reference.surface : '(원문을 입력하면 비교할 수 있습니다)');
    renderPrediction($('pred-text'), result.prediction);
    const box = $('metrics');
    box.replaceChildren();
    if (result.metrics) {
      showResult({ reference: result.reference, prediction: result.prediction,
                   metrics: result.metrics, alignment: result.alignment });
    } else {
      $('result').hidden = false;
      setText($('ref-jamo'), '-');
      setText($('pred-jamo'), result.prediction.jamo);
      $('diff').replaceChildren();
    }
    note($('upload-note'), result.reference_error ||
      `${result.prediction.inference_seconds.toFixed(2)}초 만에 인식했습니다.`,
      Boolean(result.reference_error));
  } catch (err) {
    if (token === state.generation) note($('upload-note'), err.message, true);
  } finally {
    $('upload').disabled = false;
  }
}

/* ---------- history ---------- */

async function loadHistory() {
  try {
    // Asked for by source, so the server filters before it takes the last
    // fifty. Trimming here instead would let the other page's runs crowd
    // this page's out of a list they were never going to be shown in.
    const found = await api(`/api/history?source=${state.source}`);
    const box = $('history');
    box.replaceChildren();
    if (!found.items.length) { setText(box, '아직 없습니다.'); return; }
    const table = document.createElement('table');
    const header = document.createElement('tr');
    for (const label of ['시각', '데이터셋', '원문', '모델 인식', '오류율']) {
      const th = document.createElement('th'); th.textContent = label; header.appendChild(th);
    }
    table.appendChild(header);
    for (const item of found.items) {
      const tr = document.createElement('tr');
      for (const value of [item.at.slice(11), splitLabel(item.split),
                           item.transcript || '(전사 없음)', item.text,
                           // No reference, no rate - and never a 0% stand-in.
                           item.metrics ? `${(item.metrics.jer * 100).toFixed(1)}%`
                                        : '비교 불가']) {
        const td = document.createElement('td'); td.textContent = value; tr.appendChild(td);
      }
      table.appendChild(tr);
    }
    box.appendChild(table);
  } catch (_) { /* history is a convenience, not worth an error banner */ }
}

/* ---------- boot ---------- */

async function boot() {
  const status = await refreshStatus();
  state.splits = status.splits;
  state.sources = status.sources;
  state.source = resolveSource(status.sources, status.splits);
  const source = status.sources[state.source];

  // The two sources are two pages, so the picker on each one offers only
  // its own datasets. Nobody has to scroll past the other source's
  // recordings to reach theirs, and nothing here can select them.
  setText($('page-title'), source.title);
  setText($('page-blurb'), source.blurb);
  document.title = source.title;
  for (const [name, meta] of Object.entries(status.sources)) {
    const tab = $(`tab-${name}`);
    if (!tab) continue;
    tab.textContent = meta.label;
    tab.href = meta.path;
    if (name === state.source) tab.setAttribute('aria-current', 'page');
    else tab.removeAttribute('aria-current');
  }

  const select = $('split');
  select.replaceChildren();
  const hints = {};
  const mine = Object.entries(status.splits)
    .filter(([, meta]) => meta.source === state.source);
  for (const [key, meta] of mine) {
    const option = document.createElement('option');
    option.value = key;
    option.textContent = `${meta.label}${meta.recordings ? ` · ${meta.recordings.toLocaleString()}개` : ''}`;
    hints[key] = meta.hint;
    select.appendChild(option);
  }

  // A dataset can be opened straight from the address bar, but only one
  // this page is allowed to show: a split from the other source falls back
  // to this source's default rather than displaying the other's data.
  state.split = resolveSplit(state.source, status.splits, source.default);
  const wanted = new URLSearchParams(location.search);
  if (['child1', 'child2'].includes(wanted.get('child'))) {
    $('child').value = wanted.get('child');
  }
  select.value = state.split;
  setText($('split-hint'), hints[state.split] || '');
  applySplitChrome();

  select.addEventListener('change', async () => {
    state.split = select.value;
    setText($('split-hint'), hints[state.split] || '');
    applySplitChrome();
    // The previously selected sample belongs to the old split, so it goes
    // before the new list is even requested.
    const token = nextGeneration();
    clearSelection();
    note($('action-note'), '');
    const loaded = await loadList(1);
    // Both guards matter: a newer split change supersedes the list, and a
    // click in the meantime supersedes the automatic pick.
    if (!loaded || token !== state.generation) return;
    if (state.listItems.length) await selectSample(state.listItems[0].id, true);
    else setText($('sample-title'), '조건에 맞는 음성이 없습니다');
  });
  for (const id of ['age', 'bucket', 'style', 'child']) {
    $(id).addEventListener('change', () => { syncUrl(); loadList(1); });
  }
  let timer = null;
  $('query').addEventListener('input', () => {
    clearTimeout(timer); timer = setTimeout(() => loadList(1), 250);
  });
  $('prev-page').addEventListener('click', () => loadList(state.page - 1));
  $('next-page').addEventListener('click', () => loadList(state.page + 1));
  $('transcribe').addEventListener('click', transcribe);
  $('batch').addEventListener('click', runBatch);
  $('upload').addEventListener('click', runUpload);
  $('random').addEventListener('click', async () => {
    // Claim the screen first: a pick that arrives after the user has
    // chosen something else - or switched split - must not replace it.
    const token = nextGeneration();
    const params = new URLSearchParams(filters());
    try {
      const picked = await api(`/api/random?${params}`);
      if (token !== state.generation) return;
      await selectSample(picked.id, true);
    } catch (err) {
      if (token === state.generation) note($('action-note'), err.message, true);
    }
  });
  for (const id of ['prev-sample', 'next-sample']) {
    $(id).addEventListener('click', (event) => {
      const next = event.currentTarget.dataset.id;
      if (next) selectSample(next, true);
    });
  }

  await loadList(1);
  // Open on something real and already transcribed, so the page is useful
  // the moment it loads rather than after three clicks.
  if (state.listItems.length) {
    await selectSample(state.listItems[0].id, true);
  }
  if (!status.model.ready) {
    fetch('/api/warm', { method: 'POST' }).then(refreshStatus).catch(() => {});
  }
  loadHistory();
}

boot().catch((err) => {
  note($('action-note'), err.message, true);
});
