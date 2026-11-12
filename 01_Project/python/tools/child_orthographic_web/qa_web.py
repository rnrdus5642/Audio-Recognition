"""Real browser upload/recording checks against the running, unmocked model."""
import gzip
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
QA = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/qa'
BASE = 'http://127.0.0.1:8782'


def main():
    QA.mkdir(parents=True, exist_ok=True)
    with gzip.open(ROOT / '02_Datasets/학습준비/자모_원문라벨_3-6세/manifests-20261027/monitor.jsonl.gz', 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    cases = []
    for age in (3, 4, 5, 6):
        row = next(row for row in rows if row['age'] == age and 1.5 < row['seconds'] < 3)
        source = (ROOT / row['audio_path']).resolve()
        assert source.drive.upper() == 'C:' and source.is_relative_to(ROOT)
        path = QA / f'age-{age}.wav'
        shutil.copyfile(source, path)
        cases.append({'age': age, 'reference': row['text'], 'path': str(path)})
    report = {'model_mocked': False, 'microphone': 'Browser file-backed fake microphone; physical microphone not tested',
              'results': [], 'checks': {}, 'page_errors': []}
    owned = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True, args=[
                '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
                f'--use-file-for-fake-audio-capture={cases[0]["path"]}',
                '--autoplay-policy=no-user-gesture-required'])
            context = browser.new_context(viewport={'width': 1440, 'height': 1100}, permissions=['microphone'])
            page = context.new_page()
            page.on('pageerror', lambda exc: report['page_errors'].append(str(exc)))
            page.goto(BASE)
            page.wait_for_function("document.querySelector('#model-status').textContent.includes('사용 준비 완료')", timeout=120000)
            expected_device = page.request.get(BASE + '/api/status').json()['device']
            assert page.locator('#record').is_enabled()
            page.screenshot(path=str(QA / 'desktop-ready.png'), full_page=True)
            page.locator('#file-tab').click()
            for case in cases:
                page.locator('#reference').fill(case['reference'])
                page.locator('#age').select_option(str(case['age']))
                page.locator('#audio-file').set_input_files(case['path'])
                with page.expect_response(lambda r: r.url.endswith('/api/transcribe') and r.request.method == 'POST', timeout=120000) as response:
                    page.locator('#upload').click()
                answer = response.value
                assert answer.status == 200, answer.text()
                result = answer.json()
                owned.append(result['id'])
                report['results'].append(result)
                page.wait_for_function("!document.querySelector('#upload').disabled")
                assert page.locator('#prediction').inner_text() == (result['prediction']['text'] or '(인식된 문장 없음)')
                assert result['device'] == expected_device
                assert result['model_id'] == '3695260b526df829e75901904e14dd95dbb450fb2f9116d15438cdd8ddd06bdc'
                assert page.locator('#playback').get_attribute('src') == result['audio_url']
            report['checks']['real_wav_upload_ages_3_to_6'] = True
            page.screenshot(path=str(QA / 'desktop-result.png'), full_page=True)
            page.locator('#mic-tab').click()
            page.locator('#reference').fill('')
            page.locator('#record').click()
            page.wait_for_function("document.querySelector('#record-label').textContent === '녹음 끝내고 분석'")
            page.wait_for_function("document.querySelector('#record-time').textContent.startsWith('00:03')", timeout=15000)
            with page.expect_response(lambda r: r.url.endswith('/api/transcribe') and r.request.method == 'POST', timeout=120000) as response:
                page.locator('#record').click()
            answer = response.value
            assert answer.status == 200, answer.text()
            result = answer.json()
            owned.append(result['id'])
            report['results'].append(result)
            assert result['source'] == 'microphone' and result['score'] is None
            assert 2.8 < result['prediction']['audio_seconds'] < 4.5
            page.wait_for_function("!document.querySelector('#record').disabled")
            report['checks']['browser_audio_worklet_capture'] = True
            audio = page.request.get(BASE + result['audio_url'])
            assert audio.status == 200 and audio.body().startswith(b'RIFF')
            report['checks']['playback_wav'] = True
            page.set_viewport_size({'width': 390, 'height': 844})
            page.screenshot(path=str(QA / 'mobile.png'), full_page=True)
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            report['checks']['mobile_no_horizontal_overflow'] = True
            with page.expect_download() as download:
                page.locator('#export').click()
            download.value.save_as(str(QA / 'downloaded-history.json'))
            assert json.loads((QA / 'downloaded-history.json').read_text(encoding='utf-8'))['total'] >= 5
            report['checks']['history_download'] = True
            bad = page.request.post(BASE + '/api/transcribe', multipart={'file': {'name': 'invalid.wav', 'mimeType': 'audio/wav', 'buffer': b'not a waveform'}})
            assert bad.status == 400, bad.text()
            report['checks']['invalid_audio_rejected'] = True
            assert not report['page_errors'], report['page_errors']
            browser.close()
        report['passed'] = True
    finally:
        # Move only this QA run's own records, leaving all user recordings untouched.
        recordings = (ROOT / '03_Recordings/원문자모_성능테스트').resolve()
        artifacts = (QA / 'records').resolve()
        artifacts.mkdir(exist_ok=True)
        assert recordings.is_relative_to(ROOT) and artifacts.is_relative_to(ROOT)
        for record_id in owned:
            assert len(record_id) == 32 and all(c in '0123456789abcdef' for c in record_id)
            for suffix in ('.wav', '.json'):
                source = recordings / (record_id + suffix)
                destination = artifacts / (record_id + suffix)
                assert source.parent == recordings and destination.parent == artifacts
                if source.exists():
                    source.rename(destination)
        (QA / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': report.get('passed', False), 'checks': report['checks'],
                      'samples': [{'reference': r['reference'], 'prediction': r['prediction']['text'],
                                   'audio_seconds': r['prediction']['audio_seconds'], 'inference_seconds': r['prediction']['inference_seconds']} for r in report['results']]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
