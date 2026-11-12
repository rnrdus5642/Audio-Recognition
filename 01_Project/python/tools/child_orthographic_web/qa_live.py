"""Real CUDA model + Edge audio-worklet checks; uses file-backed test microphones."""
import asyncio
from datetime import datetime
import json
from pathlib import Path
import wave

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[4]
BASE = 'http://127.0.0.1:8782'
QA_ROOT = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test'
QA = QA_ROOT / 'live-qa' / datetime.now().strftime('run-%Y%m%d-%H%M%S')
TRACKS = """(() => {
  window.qaTracks = [];
  const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (...args) => {
    const stream = await original(...args);
    window.qaTracks.push(...stream.getTracks());
    return stream;
  };
})()"""


async def main():
    QA.mkdir(parents=True)
    with wave.open(str(QA_ROOT / 'qa/age-3.wav'), 'rb') as source:
        assert source.getnchannels() == 1 and source.getsampwidth() == 2
        rate, speech = source.getframerate(), source.readframes(source.getnframes())
    fixture = QA / 'microphone.wav'
    with wave.open(str(fixture), 'wb') as output:
        output.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        output.writeframes((b'\0' * int(rate * 0.3) * 2 + speech + b'\0' * int(rate * 1.2) * 2) * 12)
    report = {'model_mocked': False, 'device': 'cuda',
              'microphone': 'Edge file-backed microphone; physical device not tested',
              'checks': {}, 'previews': [], 'results': [], 'page_errors': []}
    owned, requests, peak_requests = set(), set(), 0
    browser = None

    async def response_seen(response):
        if response.request.method != 'POST' or response.status != 200:
            return
        if response.url.endswith('/api/preview'):
            report['previews'].append((await response.json())['prediction'])
        elif response.url.endswith('/api/transcribe'):
            item = await response.json()
            owned.add(item['id'])
            report['results'].append(item)

    def request_seen(request):
        nonlocal peak_requests
        if request.method == 'POST' and request.url.rsplit('/', 1)[-1] in ('preview', 'transcribe'):
            requests.add(request)
            peak_requests = max(peak_requests, len(requests))

    async def make_page(browser):
        context = await browser.new_context(viewport={'width': 1440, 'height': 1100}, permissions=['microphone'])
        await context.add_init_script(TRACKS)
        page = await context.new_page()
        page.on('pageerror', lambda error: report['page_errors'].append(str(error)))
        page.on('response', response_seen)
        page.on('request', request_seen)
        page.on('requestfinished', lambda request: requests.discard(request))
        page.on('requestfailed', lambda request: requests.discard(request))
        await page.goto(BASE)
        await page.wait_for_function("document.querySelector('#model-status').textContent.includes('사용 준비 완료')", timeout=60000)
        assert await page.locator('#device-label').inner_text() == 'GPU 추론'
        await page.locator('#live-tab').click()
        return context, page

    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(channel='msedge', headless=True, args=[
                '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
                f'--use-file-for-fake-audio-capture={fixture}', '--autoplay-policy=no-user-gesture-required'])
            context, page = await make_page(browser)
            await page.locator('#age').select_option('3')
            await page.screenshot(path=str(QA / 'desktop-ready.png'), full_page=True)
            await page.locator('#live-record').click()
            await page.wait_for_function("[...document.querySelectorAll('.live-provisional')].some(p=>p.textContent && p.textContent!=='듣고 있습니다…')", timeout=20000)
            assert await page.locator('#live-record').is_enabled()
            assert await page.locator('#live-record').get_attribute('aria-pressed') == 'true'
            report['checks']['partial_text_while_capturing'] = True
            await page.wait_for_function("document.querySelectorAll('.live-final').length >= 3 && Number(document.querySelector('#live-time').textContent.slice(3,5)) >= 16", timeout=30000)
            report['checks']['continuous_capture_over_15_seconds'] = True
            await page.screenshot(path=str(QA / 'desktop-live.png'), full_page=True)
            await page.locator('#live-record').click()
            assert await page.evaluate("qaTracks.length > 0 && qaTracks.every(t=>t.readyState==='ended')")
            await page.wait_for_function('!LiveRecognition.active', timeout=60000)
            assert await page.locator('#live-state').inner_text() == '실시간 인식 완료'
            finals = await page.locator('.live-final').count()
            live_results = [item for item in report['results'] if item['source'] == 'live']
            assert finals == len(live_results) and finals >= 3
            assert all(item['device'] == 'cuda' and 0.1 <= item['prediction']['audio_seconds'] <= 12.1 for item in live_results)
            report['checks']['final_segments_saved_once'] = True
            history = (await (await page.request.get(BASE + '/api/history')).json())['items']
            assert owned <= {item['id'] for item in history}
            for item in live_results:
                audio = await page.request.get(BASE + item['audio_url'])
                assert audio.status == 200 and (await audio.body()).startswith(b'RIFF')
            report['checks']['final_audio_playback'] = True
            await page.set_viewport_size({'width': 390, 'height': 844})
            await page.screenshot(path=str(QA / 'mobile-live.png'), full_page=True)
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            report['checks']['mobile_no_horizontal_overflow'] = True
            await page.set_viewport_size({'width': 1440, 'height': 1100})

            # Regression: the upload and single-recording paths share the new backend.
            await page.locator('#file-tab').click()
            await page.locator('#audio-file').set_input_files(str(QA_ROOT / 'qa/age-4.wav'))
            async with page.expect_response(lambda r: r.url.endswith('/api/transcribe') and r.request.method == 'POST') as response:
                await page.locator('#upload').click()
            assert (await response.value).status == 200
            await page.wait_for_function("!document.querySelector('#upload').disabled")
            await page.locator('#mic-tab').click()
            await page.locator('#record').click()
            await page.wait_for_function("document.querySelector('#record-time').textContent.startsWith('00:03')", timeout=10000)
            async with page.expect_response(lambda r: r.url.endswith('/api/transcribe') and r.request.method == 'POST') as response:
                await page.locator('#record').click()
            assert (await response.value).status == 200
            await page.wait_for_function("!document.querySelector('#record').disabled")
            report['checks']['legacy_upload_and_recording'] = True
            await context.close()

            # Hold one real preview response to verify that stop releases the mic
            # immediately and the late preview cannot replace the final result.
            context, page = await make_page(browser)
            entered, release = asyncio.Event(), asyncio.Event()

            async def hold_preview(route):
                response = await route.fetch()
                entered.set()
                await release.wait()
                await route.fulfill(response=response)

            await page.route('**/api/preview', hold_preview)
            before = len(report['results'])
            await page.locator('#live-record').click()
            await asyncio.wait_for(entered.wait(), timeout=20)
            await page.locator('#live-record').click()
            assert await page.evaluate("qaTracks.every(t=>t.readyState==='ended')")
            report['checks']['stop_releases_mic_during_pending_response'] = True
            release.set()
            await page.wait_for_function('!LiveRecognition.active', timeout=20000)
            assert len(report['results']) == before + 1
            assert await page.locator('.live-provisional').count() == 0
            expected = report['results'][-1]['prediction']['text'] or '(인식된 문장 없음)'
            assert await page.locator('.live-final').inner_text() == expected
            assert report['results'][-1]['session_id'] != live_results[0]['session_id']
            report['checks']['late_preview_cannot_overwrite_final_and_new_session_is_distinct'] = True
            await context.close()

            # A failed local connection preserves captured PCM as downloadable WAV.
            context, page = await make_page(browser)

            async def unavailable(route):
                await route.fulfill(status=503, content_type='application/json', body='{"detail":"QA connection failure"}')

            await page.route('**/api/preview', unavailable)
            await page.route('**/api/transcribe', unavailable)
            await page.locator('#live-record').click()
            await page.wait_for_function("!LiveRecognition.active && document.querySelector('#message').classList.contains('error')", timeout=20000)
            assert await page.evaluate("qaTracks.every(t=>t.readyState==='ended')")
            assert await page.locator('#live-recovery a').count() >= 1
            async with page.expect_download() as download:
                await page.locator('#live-recovery a').first.click()
            await (await download.value).save_as(str(QA / 'recovered.wav'))
            assert (QA / 'recovered.wav').read_bytes().startswith(b'RIFF')
            report['checks']['connection_error_preserves_unsaved_audio'] = True
            await context.close()

            context = await browser.new_context()
            await context.add_init_script("navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('QA denied','NotAllowedError'); };")
            page = await context.new_page()
            await page.goto(BASE)
            await page.wait_for_function("!document.querySelector('#live-record').disabled")
            await page.locator('#live-tab').click()
            await page.locator('#live-record').click()
            await page.wait_for_function("document.querySelector('#message').textContent.includes('마이크 권한이 필요')")
            assert await page.locator('#live-record').is_enabled()
            assert not await page.evaluate('LiveRecognition.active')
            report['checks']['permission_denial_is_recoverable'] = True
            await context.close()
            report['peak_parallel_inference_requests'] = peak_requests
            assert peak_requests == 1, peak_requests
            assert not report['page_errors'], report['page_errors']
            report['passed'] = True
            await browser.close()
    finally:
        records = (ROOT / '03_Recordings/원문자모_성능테스트').resolve()
        destination = (QA / 'records').resolve()
        destination.mkdir(exist_ok=True)
        assert records.is_relative_to(ROOT) and destination.is_relative_to(ROOT)
        assert records.drive.upper() == destination.drive.upper() == 'C:'
        for record_id in owned:
            assert len(record_id) == 32 and all(c in '0123456789abcdef' for c in record_id)
            for suffix in ('.json', '.wav'):
                source, target = records / (record_id + suffix), destination / (record_id + suffix)
                assert source.parent == records and target.parent == destination
                if source.exists():
                    source.rename(target)
        (QA / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'passed': report.get('passed', False), 'checks': report['checks'],
                          'previews': len(report['previews']), 'finals': len(report['results']),
                          'peak_parallel_requests': peak_requests, 'artifacts': str(QA)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
