"""Exercise real WAV playback; all test answers go to a dedicated QA database."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

from server import STATE, WORKSPACE, make_server, prepare_catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packages', type=Path)
    parser.add_argument('--output', type=Path, default=WORKSPACE / '99_Organization/age3-listening-20261027')
    args = parser.parse_args()
    if args.packages:
        sys.path.insert(0, str(args.packages.resolve()))
    from playwright.sync_api import sync_playwright

    output = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=False)
    server = make_server(prepare_catalog(), output / 'qa-reviews.sqlite3', port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    report = {'passed': False, 'user_review_database_used': False, 'checks': {}, 'browser_errors': []}

    def request(path, body=None, headers=None):
        req = urllib.request.Request(base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            response = urllib.request.urlopen(req)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read()
            data = json.loads(raw) if 'application/json' in response.headers.get('Content-Type', '') else raw
            return response.status, data, dict(response.headers)

    try:
        _, listing, _ = request('/api/clips?limit=60&sort=diverse')
        assert listing['total'] == 25852
        assert len({r['speaker_key'] for r in listing['items']}) == 60
        assert request('/api/clips?limit=60&sort=diverse')[1]['items'] == listing['items']
        report['checks']['25852_clips_74_speakers_and_diverse_stable_order'] = request('/api/stats')[1]['speakers'] == 74
        first = listing['items'][0]
        source = server.room.clip(first['id'])
        source_bytes = (WORKSPACE / source['audio_path']).read_bytes()
        status, raw, _ = request(first['audio_url'])
        assert status == 200 and hashlib.sha256(raw).hexdigest() == source['audio_sha256']
        status, raw, headers = request(first['audio_url'], headers={'Range': 'bytes=0-99'})
        assert status == 206 and raw == source_bytes[:100] and headers['Content-Range'].startswith('bytes 0-99/')
        assert request(first['audio_url'], headers={'Range': 'bytes=-10'})[1] == source_bytes[-10:]
        assert request(first['audio_url'], headers={'Range': 'bytes=999999999-'})[0] == 416
        assert request(first['audio_url'], headers={'Origin': 'https://example.invalid'})[0] == 403
        assert request('/audio/../../README.md')[0] == 404
        report['checks']['audio_exact_sha_range_requests_and_local_access'] = True
        for partition in ('train', 'dev', 'test', 'review'):
            clips = request('/api/clips?partition=' + partition)[1]
            assert clips['items'] and all(r['partition'] == partition for r in clips['items'])
        assert request('/api/clips?speaker=1')[1]['items'][0]['speaker_no'] == 1
        report['checks']['partition_and_speaker_filters'] = True

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='msedge', headless=True,
                args=['--autoplay-policy=no-user-gesture-required'])
            page = browser.new_page(viewport={'width':1440, 'height':1120})
            page.on('pageerror', lambda error: report['browser_errors'].append(str(error)))
            page.goto(base)
            page.wait_for_selector('#review-workspace:not([hidden])')
            page.wait_for_function('document.querySelector("#wave-svg").children.length===120')
            assert page.locator('#label-text').is_hidden()
            assert page.locator('.clip-row strong').first.inner_text().startswith('녹음 ')
            assert page.locator('#relation').is_disabled()
            page.screenshot(path=str(output / 'desktop-blind.png'), full_page=True)
            page.locator('#player').evaluate('(a)=>a.play()')
            page.wait_for_function('document.querySelector("#player").currentTime>0.25')
            page.locator('#player').evaluate('(a)=>a.pause()')
            assert page.locator('#player').evaluate('(a)=>!a.error && a.duration>0')
            page.locator('#speed').select_option('0.8')
            assert page.locator('#player').evaluate('(a)=>a.playbackRate') == .8
            page.locator('#reveal-label').click()
            assert page.locator('#label-text').inner_text() == first['label_original']
            page.locator('.clarity-choices label').filter(has_text='또렷하게 들림').click()
            page.locator('#relation').select_option('variation')
            page.locator('#heard').fill('QA_ONLY 들리는 말')
            page.locator('#note').fill('QA_ONLY <img src=x onerror=alert(1)>')
            page.locator('#save').click()
            page.wait_for_function('document.querySelector("#saved-state").textContent==="저장 완료"')
            page.wait_for_function('document.querySelector("#reviewed-count").textContent==="1"')
            page.reload()
            page.wait_for_selector('#review-workspace:not([hidden])')
            assert page.locator('#note').input_value().startswith('QA_ONLY')
            assert page.locator('input[name=clarity][value=clear]').is_checked()
            page.locator('#reveal-label').click()
            page.screenshot(path=str(output / 'desktop-label-review.png'), full_page=True)
            report['checks']['real_browser_audio_blind_mode_speed_and_persistent_review'] = True

            page.locator('#next-clip').click()
            previous_title = f"녹음 {first['number']:05d}"
            page.wait_for_function('(previous)=>document.querySelector("#clip-title").textContent!==previous', arg=previous_title)
            page.locator('#note').fill('QA_ONLY unsaved draft')
            page.reload()
            page.wait_for_selector('#review-workspace:not([hidden])')
            assert page.locator('#note').input_value() == 'QA_ONLY unsaved draft'
            assert request('/api/stats')[1]['reviewed'] == 1
            report['checks']['unsaved_draft_survives_reload_without_counting_as_review'] = True

            page.locator('#search').fill('책')
            page.wait_for_function('document.querySelector("#match-count").textContent!=="25,852개"')
            page.locator('#blind').uncheck()
            page.wait_for_function('Array.from(document.querySelectorAll(".clip-row strong")).every(e=>e.textContent.includes("책"))')
            assert page.locator('.clip-row').count() > 0
            page.locator('#search').fill('__QA_NO_MATCH_7f9e__')
            page.wait_for_selector('#empty-state:not([hidden])')
            page.locator('#reset-filters').click()
            page.wait_for_selector('#review-workspace:not([hidden])')
            page.wait_for_function('document.querySelector("#match-count").textContent==="25,852개"')
            page.locator('#speaker').select_option('1')
            page.wait_for_function('Array.from(document.querySelectorAll(".row-copy>span")).every(e=>e.textContent.startsWith("화자 01"))')
            assert page.locator('.clip-row').count() > 0
            report['checks']['browser_search_empty_reset_and_speaker_filter'] = True

            page.set_viewport_size({'width':390, 'height':844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(output / 'mobile.png'), full_page=True)
            assert not report['browser_errors'], report['browser_errors']
            report['checks']['mobile_no_overflow_and_no_browser_errors'] = True
            browser.close()

        export = request('/api/export')[1]
        assert len(export['reviews']) == 1 and len(export['history']) == 1
        saved = export['reviews'][0]
        payload = {'id':saved['clip_id'], 'audio_sha256':saved['audio_sha256'], 'revision':0,
                   'clarity':'clear','relation':'same','heard':'QA','note':'','label_seen':True}
        assert request('/api/review', payload)[0] == 409
        payload.update(revision=1, label_seen=False)
        assert request('/api/review', payload)[0] == 400
        assert request('/api/stats')[1]['reviewed'] == 1
        report['checks']['json_export_revision_conflict_and_invalid_save_rejection'] = True
        assert hashlib.sha256((WORKSPACE / source['audio_path']).read_bytes()).hexdigest() == source['audio_sha256']
        report['passed'] = True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'output':str(output), **report}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
