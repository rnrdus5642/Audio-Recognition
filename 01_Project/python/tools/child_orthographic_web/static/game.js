'use strict';

window.WordGame = (() => {
  const fields = ['game-threshold', 'game-consecutive'];
  const waveform = $('game-waveform'), drawing = waveform.getContext('2d');
  let state = null;

  function targets() {
    return [...new Set($('game-targets').value.normalize('NFC').split(/[,\n\r]+/)
      .map(word => word.trim()).filter(Boolean).map(word => word.replace(/[?？]+$/, '').trim()))];
  }

  function displayTargets(words, accepted = '') {
    $('game-prompt').textContent = accepted ? '확인한 정답' : words.length > 1 ? '후보 중 한 단어를 말해 보세요' : '이 단어를 말해 보세요';
    $('game-target').textContent = accepted || (words.length > 1 ? '어떤 단어가 들릴까요?' : words[0] || '정답 후보를 입력하세요');
    const list = $('game-candidates'); list.replaceChildren(); list.hidden = words.length < 2;
    words.forEach(word => {
      const chip = document.createElement('span'); chip.textContent = word;
      if (word === accepted) chip.className = 'chosen';
      list.append(chip);
    });
  }

  function displayScores(scores) {
    const rows = $('game-candidate-scores'); rows.replaceChildren();
    $('game-scores-table').hidden = scores.length < 2;
    scores.forEach(item => {
      const row = document.createElement('tr'), word = document.createElement('th'), score = document.createElement('td');
      word.scope = 'row'; word.textContent = item.target;
      score.textContent = `${(item.similarity * 100).toFixed(1)}점${item.blocked_by ? ' · 다른 동작' : ''}`;
      row.append(word, score); rows.append(row);
    });
  }

  function controls() {
    const active = !!state?.active;
    $('game-record').disabled = active ? !state.capturing : !ready;
    $('game-record').setAttribute('aria-pressed', String(!!state?.capturing));
    $('game-label').textContent = active ? (state.capturing ? '듣기 중지' : state.stopping ? '결과 정리 중…' : '마이크 연결 중…') : state?.completed ? '다시 듣기' : '듣기 시작';
    fields.forEach(id => $(id).disabled = active);
    document.body.classList.toggle('game-recording', !!state?.capturing);
  }

  function counter(streak, required) {
    $('game-streak').textContent = `${streak} / ${required}`;
    $('game-dots').replaceChildren();
    for (let i = 0; i < required; i++) {
      const dot = document.createElement('i'); dot.className = i < streak ? 'filled' : '';
      $('game-dots').append(dot);
    }
  }

  function setup() {
    if (state?.active) return;
    const words = targets();
    $('game-target-count').textContent = String(words.length);
    displayTargets(words);
    $('game-threshold-value').textContent = `${$('game-threshold').value}점`;
    $('game-time').textContent = '0.0초 · 시간 제한 없음';
    $('game-state').textContent = '준비'; $('game-card').className = 'game-card';
    $('game-instruction').textContent = words.length > 1 ? '목록에서 원하는 단어 하나를 말하면 확인합니다.' : '단어 하나를 말하면 정답인지 확인합니다.';
    $('game-heard').textContent = '듣기를 시작하면 여기에 표시됩니다.';
    $('game-heard').classList.add('placeholder'); $('game-text-state').textContent = '대기';
    $('game-score').textContent = '—';
    $('game-leading').textContent = '—'; displayScores([]);
    $('game-matched-text').textContent = '—';
    $('game-decision-reason').textContent = '—';
    counter(0, Number($('game-consecutive').value));
  }

  function draw(s) {
    if (!s.capturing) return;
    const data = new Uint8Array(s.analyser.fftSize); s.analyser.getByteTimeDomainData(data);
    drawing.clearRect(0, 0, waveform.width, waveform.height);
    drawing.strokeStyle = '#15918d'; drawing.lineWidth = 1.6; drawing.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = i * waveform.width / (data.length - 1), y = 26 + (data[i] - 128) / 128 * 25;
      i ? drawing.lineTo(x, y) : drawing.moveTo(x, y);
    }
    drawing.stroke(); s.animation = requestAnimationFrame(() => draw(s));
  }

  function stopDevice(s) {
    s.capturing = false;
    clearInterval(s.timer); cancelAnimationFrame(s.animation);
    s.stream?.getTracks().forEach(track => track.stop());
    if (s.worklet) { s.worklet.port.onmessage = null; s.worklet.disconnect(); }
    if (s.context && s.context.state !== 'closed') s.context.close().catch(() => {});
    drawing.clearRect(0, 0, waveform.width, waveform.height);
    controls(); syncControls();
  }

  function snapshot(s) {
    const samples = new Float32Array(s.ring.length);
    samples.set(s.ring.subarray(s.pointer));
    samples.set(s.ring.subarray(0, s.pointer), s.ring.length - s.pointer);
    return wavBlob([samples], s.rate);
  }

  async function request(s, path, body, json = false) {
    for (let attempt = 0; attempt < 4; attempt++) {
      const controller = new AbortController(); s.controller = controller;
      const timeout = setTimeout(() => controller.abort(), 30000);
      try {
        const response = await fetch(path, { method: 'POST', body: json ? JSON.stringify(body) : body,
          headers: json ? { 'Content-Type': 'application/json' } : {}, signal: controller.signal });
        const answer = await response.json();
        if (response.ok) return answer;
        const error = new Error(typeof answer.detail === 'string' ? answer.detail : '입력값이나 서버 상태를 확인해 주세요.');
        error.retryable = [409, 503].includes(response.status) || response.status >= 500;
        throw error;
      } catch (error) {
        if (attempt === 3 || !(error.retryable || error instanceof TypeError || error.name === 'AbortError')) throw error;
        await new Promise(resolve => setTimeout(resolve, 300 * (attempt + 1)));
      } finally { clearTimeout(timeout); s.controller = null; }
    }
  }

  function progress(s, answer) {
    if (s !== state) return;
    counter(answer.streak, s.settings.consecutive);
    const heard = answer.prediction?.text || (answer.prediction ? '(인식된 텍스트 없음)' : '분석된 음성이 없습니다.');
    if ($('game-heard').textContent !== heard) $('game-heard').textContent = heard;
    $('game-heard').classList.toggle('placeholder', !answer.prediction?.text);
    $('game-text-state').textContent = answer.state === 'listening' ? '인식 중' : '마지막 인식';
    $('game-score').textContent = `${(answer.similarity * 100).toFixed(1)}점 / 기준 ${Math.round(s.settings.threshold * 100)}점`;
    $('game-leading').textContent = answer.ambiguous ? `${answer.tied_targets.join(' · ')} (동점)` : answer.candidate || '—';
      const normalizationLabels = { repeated_word: '반복', listed_word_affix: '조사·어미 분리' };
      $('game-matched-text').textContent = normalizationLabels[answer.normalization]
        ? `${answer.matched_text} → ${answer.comparison_text} (${normalizationLabels[answer.normalization]})` : answer.matched_text || '—';
    const reasons = { different_action: `${answer.blocked_by}로 인식되어 보류`, ambiguous: '후보 점수가 같아 보류',
      uncertain_check: '잠깐 흔들린 인식을 기다리는 중', confirming: '같은 후보를 확인하는 중',
      no_voice: '말소리 대기', below_threshold: '유사도가 기준보다 낮음', waiting: '말소리 대기' };
      $('game-decision-reason').textContent = answer.selection_reason === 'latest_distinct_exact_word'
        ? '문장 속 여러 정답 중 뒤에 나온 단어 확인'
        : answer.state === 'accepted' ? '확인 횟수 충족' : reasons[answer.decision_reason] || '—';
    displayScores(answer.candidate_scores || []);
    if (!s.stopping) {
      $('game-state').textContent = answer.blocked_by ? '다른 동작으로 인식' : answer.ambiguous ? '후보 구분 중' : answer.streak ? '확인 중' : '듣는 중';
      $('game-instruction').textContent = answer.blocked_by ? `${answer.blocked_by}로 들렸어요. 정답 후보를 다시 말해 주세요.`
          : answer.ambiguous ? '여러 후보의 점수가 같아 아직 어느 단어인지 확정하지 않았어요.'
        : answer.confirmation_waiting ? '인식이 잠깐 흔들렸어요. 계속 듣고 있어요.'
          : answer.streak ? '비슷한 발음이 들렸어요. 한 번 더 확인하고 있어요.' : '정답 단어가 들어간 말을 편하게 해 주세요.';
    }
  }

  async function complete(s, answer) {
    if (s !== state || s.completed) return;
    stopDevice(s); s.active = false; s.completed = true;
    progress(s, answer);
    const labels = { accepted: '정답 확인', retry: '다시 말해 주세요', stopped: '듣기 중지' };
    $('game-state').textContent = labels[answer.state];
    $('game-card').className = `game-card ${answer.state}`;
    displayTargets(s.settings.targets, answer.matched_target || '');
    $('game-instruction').textContent = answer.state === 'accepted' ? '정답으로 확인했어요. 다음 단어를 테스트해 보세요.' : answer.state === 'retry' ? '아직 정답을 확인하지 못했어요. 다시 말해 주세요.' : '다시 듣기를 누르면 새로 시작합니다.';
    if (answer.record) {
      showResult(answer.record);
      message('판정 결과와 비교에 사용한 음성을 저장했습니다. 아래에서 다시 들을 수 있습니다.');
      await loadHistory();
    } else message('저장할 판정 구간이 없습니다. 다시 듣기를 눌러 주세요.');
    controls(); syncControls();
    window.dispatchEvent(new CustomEvent('wordgame:finished', { detail: answer }));
  }

  function fail(s, error) {
    if (s !== state) return;
    stopDevice(s); s.active = false; s.failed = true;
    $('game-text-state').textContent = '중지됨';
    $('game-state').textContent = '테스트 중지'; $('game-card').className = 'game-card failed';
    $('game-instruction').textContent = '연결이나 마이크 설정을 확인하고 다시 시작해 주세요.';
    message(error.name === 'NotAllowedError' ? '마이크 권한이 필요합니다. 주소창의 마이크 설정을 확인해 주세요.' : error.message, true);
    if (s.total > 0) {
      const link = document.createElement('a'); s.recoveryUrl = URL.createObjectURL(snapshot(s));
      link.href = s.recoveryUrl; link.download = 'word-test-unsaved.wav';
      link.textContent = '최근 녹음 구간 내려받기';
      $('game-recovery').replaceChildren(link); $('game-recovery').hidden = false;
    }
    controls(); syncControls();
  }

  async function pump(s) {
    if (s.processing || !s.active || s.failed) return;
    s.processing = true;
    try {
      while (s.active) {
        if (s.stopping !== 'stopped' && s.total - s.sent >= s.rate * s.settings.hop_seconds) {
          s.sent = s.total; s.index++;
          const form = new FormData();
          form.append('file', snapshot(s), 'window.wav');
          form.append('frame_index', String(s.index)); form.append('elapsed_seconds', String(s.sent / s.rate));
          const answer = await request(s, `/api/game/${s.id}/frame`, form);
          if (s !== state) return;
          progress(s, answer);
          if (answer.state !== 'listening') { await complete(s, answer); return; }
        } else break;
      }
      if (s.stopping && s.active) {
        const answer = await request(s, `/api/game/${s.id}/finish`, { reason: s.stopping }, true);
        await complete(s, answer);
      }
    } catch (error) { fail(s, error); }
    finally { s.processing = false; }
  }

  function stop(reason = 'stopped') {
    const s = state;
    if (!s?.active || s.stopping) return;
    s.stopping = reason; stopDevice(s); $('game-state').textContent = '마이크 중지 · 결과 정리 중';
    $('game-text-state').textContent = '결과 정리 중';
    void pump(s);
  }

  async function start() {
    if (state?.active || !ready) return;
    const words = targets();
    if (!words.length || words.length > 20 || words.some(word => !/^[가-힣]{1,20}$/.test(word))) {
      message('정답 후보는 1~20개, 각각 한글 1~20글자로 입력해 주세요. 쉼표나 줄바꿈으로 구분하고, 끝의 물음표는 생략해 비교합니다.', true); $('game-targets').focus(); return;
    }
    if (!navigator.mediaDevices?.getUserMedia) { message('Edge 또는 Chrome에서 마이크를 사용해 주세요.', true); return; }
    if (state?.recoveryUrl) URL.revokeObjectURL(state.recoveryUrl);
    $('game-recovery').replaceChildren(); $('game-recovery').hidden = true;
    setup(); message();
    const s = state = { active: true, capturing: false, stopping: '', processing: false,
      completed: false, failed: false, total: 0, sent: 0, pointer: 0, index: 0 };
    controls(); syncControls(); $('game-state').textContent = '준비 중'; $('game-text-state').textContent = '연결 중';
    try {
      const started = await request(s, '/api/game/start', { targets: words, age: $('age').value,
        threshold: Number($('game-threshold').value) / 100,
        consecutive: Number($('game-consecutive').value), timeout_seconds: null }, true);
      s.id = started.session_id; s.settings = started.settings; displayTargets(s.settings.targets);
      s.stream = await navigator.mediaDevices.getUserMedia({ audio: {
        channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      s.context = new AudioContext(); await s.context.resume(); s.rate = s.context.sampleRate;
      s.ring = new Float32Array(Math.round(s.rate * s.settings.window_seconds));
      await s.context.audioWorklet.addModule('/assets/pcm-recorder.js');
      const input = s.context.createMediaStreamSource(s.stream);
      s.analyser = s.context.createAnalyser(); s.analyser.fftSize = 512;
      s.worklet = new AudioWorkletNode(s.context, 'pcm-recorder');
      const mute = s.context.createGain(); mute.gain.value = 0;
      input.connect(s.analyser); input.connect(s.worklet); s.worklet.connect(mute); mute.connect(s.context.destination);
      s.worklet.port.onmessage = event => {
        if (!s.capturing) return;
        const samples = new Float32Array(event.data);
        for (const sample of samples) { s.ring[s.pointer] = sample; s.pointer = (s.pointer + 1) % s.ring.length; }
        s.total += samples.length;
        if (!s.processing && s.total - s.sent >= s.rate * s.settings.hop_seconds) void pump(s);
      };
      s.stream.getAudioTracks().forEach(track => track.addEventListener('ended', () => { if (s.capturing) stop(); }));
      s.capturing = true; s.started = performance.now();
      $('game-text-state').textContent = '음성 대기';
      $('game-state').textContent = '듣는 중'; $('game-instruction').textContent = '편하게 단어 하나를 말해 주세요.';
      $('game-card').className = 'game-card listening'; controls(); syncControls(); draw(s);
      s.timer = setInterval(() => {
        const seconds = (performance.now() - s.started) / 1000;
        $('game-time').textContent = `${seconds.toFixed(1)}초 · 시간 제한 없음`;
        void pump(s);
      }, 100);
    } catch (error) { fail(s, error); }
  }

  $('game-record').onclick = () => state?.capturing ? stop() : start();
  $('game-targets').addEventListener('input', setup);
  fields.forEach(id => $(id).addEventListener('input', setup));
  window.addEventListener('beforeunload', () => { if (state) { stopDevice(state); state.controller?.abort(); } });
  setup();
  return { get active() { return !!state?.active; }, syncControls: controls };
})();
syncControls();
