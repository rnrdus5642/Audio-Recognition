'use strict';

const $ = (id) => document.getElementById(id);
const state = {
  scope: 'priority', source: '', page: 1, pages: 1,
  queue: [],            // ids in the order shown
  current: null,        // the record on screen
  generation: 0,        // newest request wins; older replies are dropped
  dirty: false,         // an answer typed but not saved
  busy: false,          // a save is in flight; nothing moves
  loading: false,       // the next recording is on its way; no answer target
  // The four arrows have a disabled state of their own - the ends of the
  // list and the ends of the pager - which the guards restore on the way
  // out instead of enabling everything.
  nav: {'prev-item': true, 'next-item': true,
        'prev-page': true, 'next-page': true},
  choices: {}, flags: {},
};

function nextGeneration() { return ++state.generation; }

/* Two different locks, because they protect against different things.
 *
 * While a save is in flight nothing at all moves: disabling only the save
 * button left save-stay, "unusable", the arrows, the pager, the list and
 * the selects live, and a click on any of them could land an answer on one
 * recording and then navigate to another before the reply came back.
 *
 * While the *next* recording is being fetched, the answer form is inert but
 * navigation still works - navigating is how you leave. Leaving the old
 * form editable during that fetch was the answer-loss race: an answer typed
 * into it was silently thrown away by the rebuild when the reply landed. */
function applyLock() {
  const busy = state.busy;
  const inert = busy || state.loading;
  for (const id of ['save', 'save-stay', 'unusable', 'heard-text', 'note']) {
    $(id).disabled = inert;
  }
  for (const input of document.querySelectorAll(
      '#choice-list input, #flag-list input')) {
    input.disabled = inert;
  }
  for (const id of ['scope', 'source']) $(id).disabled = busy;
  for (const id of ['prev-item', 'next-item', 'prev-page', 'next-page']) {
    $(id).disabled = busy || state.nav[id];
  }
  for (const button of $('queue').querySelectorAll('button')) {
    button.disabled = busy;
  }
  document.body.classList.toggle('busy', inert);
}
function setBusy(busy) { state.busy = busy; applyLock(); }
function setLoading(loading) { state.loading = loading; applyLock(); }
function setNav(id, disabled) { state.nav[id] = disabled; applyLock(); }
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

const STATUS_LABEL = { answered: '답함', unclear: '모르겠음', unusable: '쓸 수 없음' };

/* ---------- answer form ---------- */

/* The form is rebuilt per recording rather than reused, so an answer can
 * never be carried over onto the next id by a stale input still holding
 * the previous value. */
function buildForm(item) {
  const list = $('choice-list');
  list.replaceChildren();
  const options = [
    ['expected', `전사대로 들림 — "${item.expected_surface}"`],
    ...(item.prediction_differs
      ? [['prediction', `모델이 쓴 대로 들림 — "${item.predicted_text}"`]] : []),
    ['other', '다르게 들림 (직접 적기)'],
    ['unclear', '판단하기 어려움'],
  ];
  for (const [value, label] of options) {
    const row = document.createElement('label');
    row.className = 'choice';
    const input = document.createElement('input');
    input.type = 'radio';
    input.name = 'choice';
    input.value = value;
    // Nothing is preselected: a default would be an answer nobody gave.
    input.checked = false;
    input.addEventListener('change', () => {
      state.dirty = true;
      $('heard-row').hidden = value !== 'other';
      if (value === 'other') $('heard-text').focus();
    });
    const span = document.createElement('span');
    span.textContent = label;
    row.append(input, span);
    list.appendChild(row);
  }

  const flags = $('flag-list');
  flags.replaceChildren();
  for (const [value, label] of Object.entries(state.flags)) {
    const row = document.createElement('label');
    row.className = 'choice';
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.name = 'flag';
    input.value = value;
    input.addEventListener('change', () => { state.dirty = true; });
    const span = document.createElement('span');
    span.textContent = label;
    row.append(input, span);
    flags.appendChild(row);
  }

  $('heard-row').hidden = true;
  $('heard-text').value = '';
  $('note').value = '';
  state.dirty = false;

  // Restore what was answered before, so revisiting shows the saved answer.
  const saved = item.annotation;
  if (saved) {
    if (saved.status === 'unclear') {
      const radio = list.querySelector('input[value="unclear"]');
      if (radio) radio.checked = true;
    } else if (saved.choice) {
      const radio = list.querySelector(`input[value="${saved.choice}"]`);
      if (radio) radio.checked = true;
      $('heard-row').hidden = saved.choice !== 'other';
    }
    if (saved.heard_text) $('heard-text').value = saved.heard_text;
    if (saved.note) $('note').value = saved.note;
    for (const flag of saved.flags || []) {
      const box = flags.querySelector(`input[value="${flag}"]`);
      if (box) box.checked = true;
    }
    state.dirty = false;
  }
}

function readForm() {
  const choice = document.querySelector('#choice-list input:checked');
  const flags = [...document.querySelectorAll('#flag-list input:checked')]
    .map((box) => box.value);
  return {
    choice: choice ? choice.value : null,
    // Sent exactly as typed. `trim()` here would quietly change an answer
    // the page promises to keep verbatim; emptiness is checked separately.
    heard_text: $('heard-text').value,
    note: $('note').value,
    flags,
  };
}

/* ---------- item ---------- */

/* The answer target itself: the controls an answer can be typed into.
 *
 * Emptied the moment a recording stops being the one on screen, so there is
 * never a live form belonging to a recording the page has moved off. */
function clearAnswerForm() {
  state.current = null;
  state.dirty = false;
  $('choice-list').replaceChildren();
  $('flag-list').replaceChildren();
  $('heard-row').hidden = true;
  $('heard-text').value = '';
  $('note').value = '';
}

/* Nothing about the previous recording survives a list change. */
function clearDetail() {
  nextGeneration();
  clearAnswerForm();
  setNav('prev-item', true);
  setNav('next-item', true);
  for (const id of ['player', 'context-player']) {
    const audio = $(id);
    audio.pause();
    audio.removeAttribute('src');
    audio.load();
  }
  $('context-block').hidden = true;
  setText($('item-title'), '음성을 고르세요');
  setText($('item-meta'), '');
  $('history-block').hidden = true;
  $('reveal').open = false;
  for (const id of ['reveal-transcript', 'reveal-expected', 'reveal-prediction']) {
    setText($(id), '');
  }
  note($('answer-note'), '');
}

async function openItem(recordId, { force } = {}) {
  if (state.busy) return false;
  // Reopening the *same* recording rebuilds the form, so an unsaved answer
  // would be discarded by a stray click on the row already selected. The
  // guard covers that case as well as moving to a different one.
  if (!force && state.dirty) {
    if (!window.confirm('저장하지 않은 답이 있습니다. 저장하지 않고 넘어갈까요?')) {
      return false;
    }
  }
  const token = nextGeneration();
  // The old recording stops being answerable here, before the await, not
  // after the reply comes back. Otherwise an answer typed while the next
  // item was loading belonged to a recording the page had already left,
  // and the rebuild below discarded it without a word.
  clearAnswerForm();
  setLoading(true);
  let item;
  try {
    item = await api(`/api/review/item/${encodeURIComponent(recordId)}`);
  } catch (err) {
    if (token === state.generation) {
      setLoading(false);
      note($('answer-note'), err.message, true);
    }
    return false;
  }
  // A newer click won: it owns the lock now and will release it itself.
  if (token !== state.generation) return false;
  setLoading(false);
  state.current = item;
  state.dirty = false;

  const position = state.queue.indexOf(recordId);
  setText($('item-title'),
    `${position >= 0 ? `${position + 1}. ` : ''}${item.transcript || '(전사 없음)'}`);
  // Friendly labels only: no record ids, no speaker hashes on screen.
  const bits = [item.source_label];
  if (item.child_label) bits.push(item.child_label);
  if (item.age && item.age !== 'unknown') bits.push(`${item.age}세`);
  bits.push(`${item.seconds.toFixed(2)}초`);
  setText($('item-meta'), bits.join(' · '));

  $('player').src = item.audio_url;
  $('player').load();
  const context = $('context-block');
  if (item.context_url) {
    $('context-player').src = item.context_url;
    $('context-player').load();
    context.hidden = false;
  } else {
    $('context-player').removeAttribute('src');
    context.hidden = true;
  }

  setText($('reveal-transcript'), item.transcript);
  setText($('reveal-expected'), item.expected_surface);
  setText($('reveal-prediction'), item.predicted_text);
  $('reveal').open = false;

  buildForm(item);
  note($('answer-note'), item.annotation
    ? `${STATUS_LABEL[item.annotation.status] || ''} · ${item.annotation.at} 에 저장됨` +
      (item.annotation.revisions > 1 ? ` (${item.annotation.revisions}번째 답)` : '')
    : '');
  if (item.annotation_stale) {
    note($('answer-note'),
      '이전 답은 지금의 음성·목표와 맞지 않아 아직 답하지 않은 것으로 ' +
      '둡니다. 다시 들어 보고 답해 주세요.', true);
  }
  $('history-block').hidden = !item.annotation;
  if (item.annotation) loadHistory(recordId, token);

  for (const button of $('queue').querySelectorAll('button')) {
    button.setAttribute('aria-current', String(button.dataset.id === recordId));
  }
  setNav('prev-item', position <= 0);
  setNav('next-item', position < 0 || position >= state.queue.length - 1);
  return true;
}

async function loadHistory(recordId, token) {
  try {
    const found = await api(`/api/review/history/${encodeURIComponent(recordId)}`);
    // A slow history reply must not fill in under a recording the user has
    // already moved to.
    if ((token !== undefined && token !== state.generation)
        || !state.current || state.current.id !== recordId) return;
    const box = $('history');
    box.replaceChildren();
    const table = document.createElement('table');
    for (const row of found.items) {
      const tr = document.createElement('tr');
      for (const value of [row.at, STATUS_LABEL[row.status] || row.status,
                           row.heard_text || row.choice || '',
                           (row.flags || []).join(', ')]) {
        const td = document.createElement('td');
        td.textContent = value;
        tr.appendChild(td);
      }
      table.appendChild(tr);
    }
    box.appendChild(table);
  } catch (_) { /* history is a convenience */ }
}

/* ---------- saving ---------- */

async function save({ advance, unusable }) {
  // `loading` matters as well as `busy`: the form is empty and belongs to
  // no recording while the next one is on its way, and the submit event can
  // still arrive from the keyboard.
  if (!state.current || state.busy || state.loading) return;
  // Everything the answer is about is captured now, before any await: the
  // recording, the identity of what was played, and the revision that was
  // on screen. The reply is only applied if the screen is still showing it.
  const item = state.current;
  const recordId = item.id;
  const binding = item.binding;
  const expectedRevision = item.revision;
  const token = state.generation;
  const form = readForm();
  let status = 'answered';
  let choice = form.choice;
  if (unusable) {
    status = 'unusable';
    choice = null;
    if (!form.flags.length) {
      note($('answer-note'), '쓸 수 없는 이유를 하나 이상 골라 주세요.', true);
      return;
    }
  } else if (!choice) {
    note($('answer-note'), '어떻게 들리는지 하나 골라 주세요.', true);
    return;
  } else if (choice === 'unclear') {
    status = 'unclear';
    choice = null;
  } else if (choice === 'other' && !form.heard_text.trim()) {
    note($('answer-note'), '들린 대로 적어 주세요.', true);
    return;
  }

  // Inert only while the request is actually open. Released before any
  // navigation, which needs the controls back.
  setBusy(true);
  note($('answer-note'), '저장하는 중…');
  let result;
  try {
    result = await api('/api/review/annotation', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // The id, the identity of the audio and the revision all travel with
      // the answer, so a save can never land on whatever recording happens
      // to be on screen by then, nor overwrite a newer answer.
      body: JSON.stringify({ record_id: recordId, binding,
                             expected_revision: expectedRevision,
                             status, choice,
                             heard_text: form.heard_text, note: form.note,
                             flags: form.flags }),
    });
  } catch (err) {
    setBusy(false);
    // A failed save says so. It never reports success it did not get.
    if (token === state.generation) {
      note($('answer-note'), `저장하지 못했습니다: ${err.message}`, true);
    }
    return;
  }
  setBusy(false);

  // Only the screen that asked gets to act on the answer. If the user
  // moved on while this was in flight, their newer work stands.
  if (token !== state.generation || !state.current
      || state.current.id !== recordId) {
    note($('answer-note'), '앞선 음성의 답을 저장했습니다.');
    await refreshState();
    return;
  }
  state.dirty = false;
  state.current.revision = result.annotation.revision;
  state.current.annotation = result.annotation;
  note($('answer-note'), '저장했습니다.');
  await refreshState();
  if (advance) {
    const next = state.queue[state.queue.indexOf(recordId) + 1];
    if (next) await openItem(next, { force: true });
    else note($('answer-note'), '저장했습니다. 목록의 마지막입니다.');
  } else {
    $('history-block').hidden = false;
    loadHistory(recordId, state.generation);
  }
}

/* ---------- lists ---------- */

function renderQueue(items, { showStatus }) {
  const list = $('queue');
  list.replaceChildren();
  state.queue = items.map((item) => item.id);
  for (const item of items) {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.id = item.id;
    const line1 = document.createElement('span');
    line1.className = 'line1';
    line1.textContent = item.transcript || '(전사 없음)';
    const line2 = document.createElement('span');
    line2.className = 'line2';
    const bits = [item.label || (item.pool === 'personal' ? '직접 녹음' : '말뭉치')];
    if (item.reason) bits.push(item.reason);
    if (item.status) bits.push(STATUS_LABEL[item.status]);
    line2.textContent = bits.join(' · ');
    button.append(line1, line2);
    if (item.status) button.classList.add('done');
    button.addEventListener('click', () => openItem(item.id));
    li.appendChild(button);
    list.appendChild(li);
  }
  if (!items.length) {
    const li = document.createElement('li');
    li.style.cssText = 'padding:14px 8px;color:#666d78';
    li.textContent = '해당하는 음성이 없습니다.';
    list.appendChild(li);
  }
  // Freshly built buttons start enabled; a lock that is still held has to
  // reach them too.
  applyLock();
}

/* The wider list is one source at a time - the server has no combined
 * paging over both - so "전체" is removed there instead of silently
 * meaning "personal". */
function applyScopeChrome() {
  const everything = $('source').querySelector('option[value=""]');
  const flagged = state.scope === 'flagged';
  everything.disabled = flagged;
  everything.hidden = flagged;
  if (flagged && !state.source) {
    state.source = 'personal';
    $('source').value = 'personal';
  }
}

async function loadQueue(openFirst) {
  const token = nextGeneration();
  applyScopeChrome();
  setText($('queue-count'), '불러오는 중…');
  if (state.scope === 'priority') {
    $('flagged-pager').hidden = true;
    setNav('prev-page', true);
    setNav('next-page', true);
    const found = await api(`/api/review/state${state.source ? `?source=${state.source}` : ''}`);
    if (token !== state.generation) return;
    state.choices = found.choices;
    state.flags = found.flags;
    renderQueue(found.priority, { showStatus: true });
    setText($('queue-count'), `${found.priority.length}개`);
    setText($('scope-hint'), found.note);
  } else {
    $('flagged-pager').hidden = false;
    const params = new URLSearchParams({
      source: state.source, page: state.page, size: 30 });
    const found = await api(`/api/review/flagged?${params}`);
    if (token !== state.generation) return;
    state.pages = found.pages;
    renderQueue(found.items, { showStatus: true });
    setText($('queue-count'), `${found.total.toLocaleString()}개`);
    setText($('page-label'), `${found.page} / ${found.pages}`);
    setNav('prev-page', found.page <= 1);
    setNav('next-page', found.page >= found.pages);
    setText($('scope-hint'), '참고용 목록입니다. 전부 확인하실 필요는 없습니다.');
  }
  if (openFirst && state.queue.length) await openItem(state.queue[0]);
}

async function refreshState() {
  const found = await api('/api/review/state');
  state.choices = found.choices;
  state.flags = found.flags;
  const p = found.progress.priority;
  setText($('progress-text'),
    `우선 확인 ${p.answered + p.unclear + p.unusable} / ${p.total} 완료`);
  const summary = $('summary');
  summary.replaceChildren();
  const rows = [
    ['우선 확인', p],
    ['직접 녹음', found.progress.by_source.personal],
    ['말뭉치 (AIHub)', found.progress.by_source.valid],
  ];
  for (const [label, tally] of rows) {
    const row = document.createElement('div');
    row.className = 'summary-row';
    const name = document.createElement('b');
    name.textContent = label;
    const value = document.createElement('span');
    value.textContent =
      `${tally.answered}개 답함 · ${tally.unclear}개 모르겠음 · ` +
      `${tally.unusable}개 쓸 수 없음 · ${tally.remaining}개 남음`;
    row.append(name, value);
    summary.appendChild(row);
  }
  setText($('review-foot'),
    `차이가 난 녹음은 직접 녹음 ${found.flagged_counts.personal}개, ` +
    `말뭉치 ${found.flagged_counts.valid}개입니다. ` +
    `우선 확인 목록만 답해 주셔도 다음 단계로 넘어갈 수 있습니다.`);
  // Update the queue's status badges without reloading the list.
  const status = new Map(found.priority.map((e) => [e.id, e.status]));
  for (const button of $('queue').querySelectorAll('button')) {
    if (status.has(button.dataset.id) && status.get(button.dataset.id)) {
      button.classList.add('done');
    }
  }
  return found;
}

/* ---------- boot ---------- */

async function boot() {
  await refreshState();
  await loadQueue(true);

  /* Changing the list changes what is on the right too. Leaving the old
   * recording's player and answer sitting there invites answering one
   * source's clip while another source's list is shown. */
  async function switchList(apply, previous) {
    if (state.dirty
        && !window.confirm('저장하지 않은 답이 있습니다. 목록을 바꿀까요?')) {
      previous();                       // put the select back
      return;
    }
    apply();
    state.page = 1;
    clearDetail();
    await loadQueue(true);
  }

  $('scope').addEventListener('change', () => switchList(
    () => { state.scope = $('scope').value; },
    () => { $('scope').value = state.scope; }));
  $('source').addEventListener('change', () => switchList(
    () => { state.source = $('source').value; },
    () => { $('source').value = state.source; }));
  // Paging changes the generation, so it is held back while a save is in
  // flight for the same reason the arrows are: the reply has to come back
  // to the screen that asked for it.
  for (const [id, step] of [['prev-page', -1], ['next-page', 1]]) {
    $(id).addEventListener('click', async () => {
      if (state.busy) return;
      state.page = Math.min(state.pages, Math.max(1, state.page + step));
      await loadQueue(false);
    });
  }
  $('answer').addEventListener('submit', (event) => {
    event.preventDefault();
    save({ advance: true });
  });
  $('save-stay').addEventListener('click', () => save({ advance: false }));
  $('unusable').addEventListener('click', () => save({ advance: false, unusable: true }));
  for (const id of ['heard-text', 'note']) {
    $(id).addEventListener('input', () => { state.dirty = true; });
  }
  for (const [id, step] of [['prev-item', -1], ['next-item', 1]]) {
    $(id).addEventListener('click', () => {
      if (!state.current || state.busy) return;
      const position = state.queue.indexOf(state.current.id) + step;
      if (position >= 0 && position < state.queue.length) {
        openItem(state.queue[position]);
      }
    });
  }
  // Playing audio must never count as leaving the answer behind.
  window.addEventListener('beforeunload', (event) => {
    if (state.dirty) { event.preventDefault(); event.returnValue = ''; }
  });
}

boot().catch((err) => {
  note($('answer-note'), err.message, true);
  setText($('progress-text'), '준비되지 않았습니다');
});
