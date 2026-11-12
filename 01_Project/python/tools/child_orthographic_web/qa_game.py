"""Check automatic word decisions with real child audio, the CUDA model and Edge."""
import asyncio
from array import array
from datetime import datetime
import gzip
import json
from pathlib import Path
import random
import wave

from playwright.async_api import async_playwright
from qa_live import ROOT, QA_ROOT, BASE, TRACKS

QA = QA_ROOT / 'game-qa' / datetime.now().strftime('run-%Y%m%d-%H%M%S')


async def main():
    QA.mkdir(parents=True)
    with gzip.open(ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세/manifests-20261027/monitor.jsonl.gz', 'rt', encoding='utf-8') as stream:
        row = next(json.loads(line) for line in stream if json.loads(line)['text'].strip() == '출발')
    source = (ROOT / row['audio_path']).resolve()
    assert source.drive.upper() == 'C:' and source.is_relative_to(ROOT)
    with wave.open(str(source), 'rb') as reader:
        assert reader.getsampwidth() == 2 and reader.getnchannels() == 1
        rate, audio = reader.getframerate(), reader.readframes(reader.getnframes())
    fixtures = {}
    rng = random.Random(42)
    for name, data in {
        'speech': bytes(int(rate * 0.4) * 2) + audio + bytes(rate * 12 * 2),
        'silence': bytes(rate * 8 * 2),
        'noise': array('h', (max(-32768, min(32767, round(rng.gauss(0, 330)))) for _ in range(rate * 8))).tobytes(),
    }.items():
        path = QA / f'{name}.wav'
        with wave.open(str(path), 'wb') as writer:
            writer.setparams((1, 2, rate, 0, 'NONE', 'not compressed')); writer.writeframes(data)
        fixtures[name] = path
    curriculum = json.loads((QA_ROOT / 'curriculum-20261030-033454/selected-audio.json').read_text(encoding='utf-8'))
    wrong_action = next(item for item in curriculum if item['record_id'] == 'child:child1_31')
    wrong_path = (ROOT / wrong_action['audio_path']).resolve()
    assert wrong_path.drive.upper() == 'C:' and wrong_path.is_relative_to(ROOT)
    with wave.open(str(wrong_path), 'rb') as reader:
        assert reader.getsampwidth() == 2 and reader.getnchannels() == 1 and reader.getframerate() == rate
        wrong_audio = reader.readframes(reader.getnframes())
    fixtures['wrong_action'] = QA / 'wrong-action.wav'
    with wave.open(str(fixtures['wrong_action']), 'wb') as writer:
        writer.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        writer.writeframes(bytes(int(rate * .4) * 2) + wrong_audio + bytes(rate * 12 * 2))
    report = {'model_mocked': False, 'microphone': 'Edge file-backed test microphone; physical microphone not tested',
              'source': row['audio_path'], 'age': row['age'], 'label': '출발',
              'wrong_action_source': wrong_action['audio_path'], 'checks': {}, 'records': [], 'page_errors': []}
    owned, pending, peak = set(), set(), 0

    async def response_seen(response):
        if response.request.method == 'POST' and '/api/game/' in response.url and response.status == 200:
            item = (await response.json()).get('record')
            if item and item['id'] not in owned:
                owned.add(item['id']); report['records'].append(item)

    def request_seen(request):
        nonlocal peak
        if request.url.endswith('/frame') and request.method == 'POST':
            pending.add(request); peak = max(peak, len(pending))

    async def open_page(playwright, fixture='speech', denied=False):
        browser = await playwright.chromium.launch(channel='msedge', headless=True, args=[
            '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
            f'--use-file-for-fake-audio-capture={fixtures[fixture]}', '--autoplay-policy=no-user-gesture-required'])
        context = await browser.new_context(viewport={'width': 1440, 'height': 1100}, permissions=['microphone'])
        await context.add_init_script(TRACKS)
        await context.add_init_script("""window.qaHeard = []; window.qaReasons = [];
          document.addEventListener('DOMContentLoaded', () => {
            const heard = document.querySelector('#game-heard');
            new MutationObserver(() => qaHeard.push({
              text: heard.textContent,
              capturing: document.querySelector('#game-record').getAttribute('aria-pressed') === 'true'
            })).observe(heard, {childList: true, subtree: true, characterData: true});
            const reason = document.querySelector('#game-decision-reason');
            new MutationObserver(() => qaReasons.push(reason.textContent))
              .observe(reason, {childList: true, subtree: true, characterData: true});
          });""")
        if denied:
            await context.add_init_script("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('QA denied','NotAllowedError'); };")
        page = await context.new_page()
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        page.on('response', response_seen)
        page.on('request', request_seen)
        page.on('requestfinished', lambda request: pending.discard(request))
        page.on('requestfailed', lambda request: pending.discard(request))
        await page.goto(BASE)
        await page.wait_for_function("!document.querySelector('#game-record').disabled", timeout=60000)
        assert await page.locator('#game-panel').is_visible()
        assert await page.locator('#game-heard').is_visible()
        assert await page.locator('#game-heard').evaluate('(element)=>!element.closest("details")')
        assert await page.locator('#device-label').inner_text() == 'GPU 추론'
        return browser, page

    async def run_trial(page, target, expected, listen_seconds=5, selected=None):
        await page.locator('#game-targets').fill(target)
        await page.locator('#age').select_option(str(row['age']))
        assert await page.locator('#game-timeout').count() == 0
        assert '듣기를 시작하면' in await page.locator('#game-heard').inner_text()
        await page.evaluate('qaHeard.length = 0')
        await page.locator('#game-record').click()
        await page.wait_for_function("document.querySelector('#game-record').getAttribute('aria-pressed')==='true'", timeout=10000)
        assert await page.locator('#age').is_disabled()
        assert await page.locator('#game-targets').is_disabled()
        if expected == 'stopped':
            await page.wait_for_function('(seconds)=>parseFloat(document.querySelector("#game-time").textContent)>=seconds', arg=listen_seconds, timeout=25000)
            assert await page.evaluate('WordGame.active')
            assert await page.locator('#game-record').get_attribute('aria-pressed') == 'true'
            assert '시간 제한 없음' in await page.locator('#game-time').inner_text()
            await page.locator('#game-record').click()
        await page.wait_for_function("!WordGame.active && !document.querySelector('#game-record').disabled", timeout=20000)
        assert await page.evaluate("qaTracks.length > 0 && qaTracks.every(t=>t.readyState==='ended')")
        assert expected in (await page.locator('#game-card').get_attribute('class')), await page.locator('#game-state').inner_text()
        words = list(dict.fromkeys(word.strip() for word in target.replace('\n', ',').split(',') if word.strip()))
        display = selected or (words[0] if len(words) == 1 else '어떤 단어가 들릴까요?')
        assert await page.locator('#game-target').inner_text() == display
        assert await page.locator('#game-record').is_enabled()
        assert await page.locator('#game-text-state').inner_text() == '마지막 인식'
        if expected == 'accepted':
            observations = await page.evaluate('qaHeard')
            assert any(item['capturing'] and item['text'] == '출발' for item in observations), observations
            assert await page.locator('#game-heard').inner_text() == '출발'
            report['checks']['current_asr_text_visible_while_listening_and_retained_on_completion'] = True
        report['checks']['new_trial_clears_previous_current_text'] = True

    try:
        async with async_playwright() as playwright:
            browser, page = await open_page(playwright)
            assert await page.locator('#live-tab,#mic-tab,#file-tab,#live-panel,#mic-panel,#file-panel,#audio-file,#reference').count() == 0
            assert not await page.evaluate("Array.from(document.scripts).some(script=>script.src.endsWith('/live.js'))")
            report['checks']['only_answer_mode_exposed'] = True
            await page.locator('#game-record').click()
            assert '정답 후보는' in await page.locator('#message').inner_text()
            assert await page.evaluate('qaTracks.length') == 0
            report['checks']['missing_target_does_not_open_microphone'] = True
            await page.locator('#game-targets').fill(','.join(chr(0xac00 + index) for index in range(21)))
            await page.locator('#game-record').click()
            assert '1~20개' in await page.locator('#message').inner_text()
            assert await page.evaluate('qaTracks.length') == 0
            report['checks']['too_many_candidates_do_not_open_microphone'] = True
            await run_trial(page, '출발', 'accepted')
            accepted = report['records'][-1]
            assert accepted['source'] == 'game' and accepted['device'] == 'cuda'
            assert accepted['game']['streak'] == 2 and not accepted['game']['settings']['threshold_calibrated']
            assert accepted['prediction']['text'].replace(' ', '') == '출발'
            response = await page.request.get(BASE + accepted['audio_url'])
            assert (await response.body()).startswith(b'RIFF')
            report['checks']['real_child_word_auto_confirm_and_mic_stop'] = True
            report['checks']['confirming_audio_saved_with_decision'] = True
            candidates = '자동차,\n출발, 책, 출발'
            await run_trial(page, candidates, 'accepted', selected='출발')
            accepted = report['records'][-1]
            assert accepted['game']['matched_target'] == '출발'
            assert accepted['game']['settings']['targets'] == ['자동차', '출발', '책']
            assert accepted['score'] is None and not accepted['reference']
            assert await page.locator('#game-candidates span').count() == 3
            assert await page.locator('#game-candidates .chosen').inner_text() == '출발'
            assert await page.locator('#game-target-count').inner_text() == '3'
            assert await page.locator('#game-targets').input_value() == candidates
            assert await page.locator('#game-result-word').inner_text() == '선택된 단어: 출발'
            assert await page.locator('#cer').inner_text() == '—'
            assert await page.locator('#game-candidate-scores tr').count() == 3
            report['checks']['real_multiword_selects_nonfirst_candidate_and_deduplicates'] = True
            await page.locator('#history .history-item').first.click()
            assert await page.locator('#game-result-word').inner_text() == '선택된 단어: 출발'
            assert '자동차' in await page.locator('#game-result-candidates').inner_text()
            report['checks']['history_preserves_selected_word_and_candidate_scores'] = True
            await page.screenshot(path=str(QA / 'desktop-accepted.png'), full_page=True)
            await page.set_viewport_size({'width': 390, 'height': 844})
            await page.screenshot(path=str(QA / 'mobile-accepted.png'), full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth<=innerWidth')
            report['checks']['mobile_no_overflow'] = True
            await page.set_viewport_size({'width': 1440, 'height': 1100})
            await run_trial(page, '자동차', 'stopped', listen_seconds=16)
            assert report['records'][-1]['game']['outcome'] == 'stopped'
            assert report['records'][-1]['game']['settings']['timeout_seconds'] is None
            assert report['records'][-1]['game']['last_checked_seconds'] > 15
            report['checks']['continues_past_15_seconds_and_manual_stop_releases_microphone'] = True
            await page.screenshot(path=str(QA / 'desktop-stopped.png'), full_page=True)
            await browser.close()

            browser, page = await open_page(playwright)
            entered, release = asyncio.Event(), asyncio.Event()

            async def hold_frame(route):
                response = await route.fetch(); entered.set(); await release.wait()
                await route.fulfill(response=response)

            await page.route('**/api/game/*/frame', hold_frame)
            await page.locator('#game-targets').fill('자동차')
            await page.locator('#game-record').click()
            await asyncio.wait_for(entered.wait(), timeout=15)
            await page.locator('#game-record').click()
            assert await page.evaluate("qaTracks.every(t=>t.readyState==='ended')")
            release.set()
            await page.wait_for_function('!WordGame.active')
            assert 'stopped' in await page.locator('#game-card').get_attribute('class')
            report['checks']['manual_stop_during_pending_frame'] = True
            await browser.close()

            browser, page = await open_page(playwright)
            aborted = False

            async def lose_confirmed_response(route):
                nonlocal aborted
                response = await route.fetch(); answer = await response.json()
                if answer['state'] == 'accepted' and not aborted:
                    aborted = True
                    item = answer['record']
                    if item['id'] not in owned:
                        owned.add(item['id']); report['records'].append(item)
                    await route.abort('failed')
                else:
                    await route.fulfill(response=response)

            await page.route('**/api/game/*/frame', lose_confirmed_response)
            before = len(owned)
            await run_trial(page, '자동차, 출발, 책', 'accepted', selected='출발')
            assert aborted and len(owned) == before + 1
            report['checks']['lost_confirmation_response_retried_without_duplicate_record'] = True
            await browser.close()

            for kind in ('silence', 'noise'):
                browser, page = await open_page(playwright, kind)
                await run_trial(page, '출발, 책, 자동차', 'stopped')
                report['checks'][f'{kind}_not_accepted'] = True
                await browser.close()

            browser, page = await open_page(playwright, 'wrong_action')
            await run_trial(page, '먹이, 밥, 줘, 줘요, 줘야지, 줄거야, 주자, 줄거에요, 줘야죠, 줄까, 줄까요', 'stopped')
            assert await page.evaluate("qaReasons.some(text=>text.includes('쳐요로 인식되어 보류'))")
            assert report['records'][-1]['game']['outcome'] == 'stopped'
            assert report['records'][-1]['game']['blocked_by'] == '쳐요'
            assert '쳐요: 다른 동작' in await page.locator('#game-result-candidates').inner_text()
            await page.locator('.game-debug').evaluate('(element)=>element.open=true')
            await page.screenshot(path=str(QA / 'different-action.png'), full_page=True)
            report['checks']['actual_chyeoyo_audio_not_accepted_as_jwoyo_and_reason_shown'] = True
            await browser.close()

            browser, page = await open_page(playwright, denied=True)
            await page.locator('#game-targets').fill('출발')
            await page.locator('#game-record').click()
            await page.wait_for_function("document.querySelector('#message').textContent.includes('마이크 권한이 필요')")
            assert not await page.evaluate('WordGame.active')
            assert await page.locator('#game-record').is_enabled()
            report['checks']['permission_error_recoverable'] = True
            await browser.close()

            browser, page = await open_page(playwright)
            await page.route('**/api/game/*/frame', lambda route: route.fulfill(status=503, content_type='application/json', body='{"detail":"QA connection failure"}'))
            await page.locator('#game-targets').fill('출발')
            await page.locator('#game-record').click()
            await page.wait_for_function("!WordGame.active && document.querySelector('#message').classList.contains('error')", timeout=15000)
            assert await page.evaluate("qaTracks.every(t=>t.readyState==='ended')")
            assert await page.locator('#game-recovery a').count() == 1
            async with page.expect_download() as download:
                await page.locator('#game-recovery a').click()
            await (await download.value).save_as(str(QA / 'recovered.wav'))
            assert (QA / 'recovered.wav').read_bytes().startswith(b'RIFF')
            report['checks']['connection_error_releases_microphone_and_preserves_audio'] = True
            await browser.close()
            assert peak == 1, peak
            assert not report['page_errors'], report['page_errors']
            report['peak_parallel_inference_requests'] = peak
            report['passed'] = True
    finally:
        source_dir = (ROOT / '03_Recordings/원문자모_성능테스트').resolve()
        destination = (QA / 'records').resolve(); destination.mkdir(exist_ok=True)
        assert source_dir.is_relative_to(ROOT) and destination.is_relative_to(ROOT)
        assert source_dir.drive.upper() == destination.drive.upper() == 'C:'
        for record_id in owned:
            assert len(record_id) == 32 and all(c in '0123456789abcdef' for c in record_id)
            for suffix in ('.json', '.wav'):
                source, target = source_dir / (record_id + suffix), destination / (record_id + suffix)
                assert source.parent == source_dir and target.parent == destination
                if source.exists(): source.rename(target)
        (QA / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'passed': report.get('passed', False), 'checks': report['checks'], 'records': len(owned),
                          'artifacts': str(QA)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
