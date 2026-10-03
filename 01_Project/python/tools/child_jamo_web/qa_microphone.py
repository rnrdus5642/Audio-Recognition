"""Real browser capture (file-backed microphone) -> current GPU model -> playback.

Use .venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe.
Only the exact records created by this run are moved out of the user's history.
"""
from pathlib import Path
import io
import json
import re
import struct
import urllib.request
import wave

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.venv/child-jamo-web-20260916/microphone'
QA = WORK / 'qa'
BASE = 'http://127.0.0.1:8776'


def main():
    QA.mkdir(parents=True, exist_ok=True)
    report = {'capture': 'Chromium file-backed fake microphone; physical microphone not claimed',
              'model_mocked': False, 'checks': {}, 'records': [], 'page_errors': []}
    owned = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True, args=[
                '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
                f'--use-file-for-fake-audio-capture={(QA / "fake-microphone.wav").resolve()}',
                '--autoplay-policy=no-user-gesture-required'])
            context = browser.new_context(viewport={'width': 1100, 'height': 1100}, permissions=['microphone'])
            page = context.new_page()
            page.on('pageerror', lambda exc: report['page_errors'].append(str(exc)))
            page.goto(BASE)
            page.wait_for_function("document.querySelector('#model-status').textContent.includes('준비 완료')")
            assert page.locator('#record').is_enabled()
            page.screenshot(path=str(QA / 'ready-desktop.png'), full_page=True)
            for automatic in (False, True):
                page.locator('#record').click()
                page.wait_for_function("document.querySelector('#record-label').textContent === '녹음 끝내기'")
                page.wait_for_function("Number(document.querySelector('.meter').getAttribute('aria-valuenow')) > 0")
                with page.expect_response(lambda r: r.url.endswith('/api/transcribe') and r.request.method == 'POST', timeout=45000) as response:
                    if not automatic:
                        page.wait_for_timeout(2800)
                        page.locator('#record').click()
                answer = response.value
                assert answer.status == 200, answer.text()
                row = answer.json()
                owned.append(row['id'])
                report['records'].append(row)
                page.wait_for_function("document.querySelector('#record-label').textContent === '녹음 시작' && !document.querySelector('#record').disabled")
                assert page.locator('#result-text').inner_text() == row['prediction']['text']
                assert row['checkpoint_sha256'] == '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7'
                assert .5 < row['seconds'] <= 15
                if automatic:
                    assert row['seconds'] >= 14.8
                data = urllib.request.urlopen(BASE + row['audio_url']).read()
                with wave.open(io.BytesIO(data)) as audio:
                    assert audio.getframerate() == 16000 and audio.getnchannels() == 1
                    pcm = audio.readframes(audio.getnframes())
                    assert max(abs(x[0]) for x in struct.iter_unpack('<h', pcm)) > 100
                page.locator('#playback').evaluate('(audio)=>audio.play()')
                page.wait_for_function("document.querySelector('#playback').currentTime > .15")
                page.locator('#playback').evaluate('(audio)=>audio.pause()')
                report['checks']['automatic_stop' if automatic else 'manual_capture_to_real_model'] = True
            page.reload()
            page.wait_for_function("document.querySelectorAll('.history-item').length >= 2")
            for record_id in owned:
                assert page.locator(f'.history-item[data-id="{record_id}"]').count() == 1
            report['checks']['history_survives_reload'] = True
            page.locator(f'.history-item[data-id="{owned[0]}"]').click()
            assert page.locator('#result-text').inner_text() == report['records'][0]['prediction']['text']
            with page.expect_download() as download:
                page.locator('#export').click()
            download.value.save_as(str(QA / 'export.json'))
            exported = json.loads((QA / 'export.json').read_text(encoding='utf-8'))
            assert set(owned) <= {r['id'] for r in exported['items']}
            report['checks']['export_and_history_selection'] = True
            page.screenshot(path=str(QA / 'result-desktop.png'), full_page=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(QA / 'result-mobile.png'), full_page=True)
            report['checks']['mobile_no_horizontal_overflow'] = True
            # Exercise denial handling without asking for a physical microphone.
            page.evaluate("() => { navigator.mediaDevices.getUserMedia = async () => { throw new DOMException('denied', 'NotAllowedError'); }; }")
            page.locator('#record').click()
            page.wait_for_function("document.querySelector('#error').textContent.includes('마이크 권한')")
            assert page.locator('#record').is_enabled()
            report['checks']['simulated_permission_denial_recovers'] = True
            assert not report['page_errors'], report['page_errors']
            browser.close()
        report['complete'] = True
    finally:
        # Preserve QA artifacts, remove only our explicitly tracked UUID records.
        recording_root = (WORK / 'recordings').resolve()
        for record_id in owned:
            assert re.fullmatch('[0-9a-f]{32}', record_id)
            for suffix in ('.wav', '.json'):
                source = (recording_root / (record_id + suffix)).resolve()
                assert source.parent == recording_root
                if source.exists():
                    (QA / source.name).write_bytes(source.read_bytes())
                    source.unlink()
        (QA / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'complete': report['complete'], 'checks': report['checks'],
                      'predictions': [r['prediction']['text'] for r in report['records']],
                      'inference_seconds': [r['prediction']['inference_seconds'] for r in report['records']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
