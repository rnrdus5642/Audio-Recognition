"""Play real clips and save synthetic answers ONLY to a disposable QA database."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright
from server import ROOT, make_server

DIRECTORY = ROOT / '.venv/child-jamo-web-20260916/phonetic-labels-20260929'
PAGE = Path(__file__).with_name('phonetic.html')


def main():
    qa = DIRECTORY / 'qa' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    qa.mkdir(parents=True)
    server = make_server(DIRECTORY, port=0, db_path=qa / 'qa-answers.sqlite3', page_path=PAGE)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    report = {'user_answer_database_used': False, 'qa_directory': str(qa), 'checks': {}, 'errors': []}

    def request(path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(base + path, data=body, headers={'Content-Type': 'application/json'})
        try:
            response = urllib.request.urlopen(req)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    try:
        review = json.loads((DIRECTORY / 'review.json').read_text(encoding='utf-8'))
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True, args=['--autoplay-policy=no-user-gesture-required'])
            page = browser.new_page(viewport={'width':1100, 'height':1000})
            page.on('pageerror', lambda error: report['errors'].append(str(error)))
            page.goto(base)
            page.wait_for_selector('#review:not([hidden])')
            assert page.locator('#reveal').is_disabled() and page.locator('#comparison').is_hidden()
            assert page.locator('input[name=choice]').count() == 0
            page.screenshot(path=str(qa / 'before-listening.png'), full_page=True)
            public = request('/api/state')[1]
            for forbidden in ('reference', 'prediction', 'surface', 'role', 'transcript'):
                assert forbidden not in json.dumps(public)
            assert request('/evidence.json')[0] == 404
            report['checks']['blind_before_listening_no_model_or_split_display'] = True
            originals_played = 0
            for index in range(15):
                page.wait_for_function('(n)=>document.querySelector("#position").textContent.startsWith(`${n}번째`)', arg=index+1)
                if index == 0:
                    page.locator('#first-heard').fill('QA only, not a listener answer')
                page.locator('#clip').evaluate('(a)=>{a.playbackRate=8; return a.play();}')
                page.wait_for_function('!document.querySelector("#reveal").disabled', timeout=15000)
                assert page.locator('#clip').evaluate('(a)=>a.ended && !a.error')
                page.locator('#reveal').click()
                page.wait_for_selector('#comparison:not([hidden])')
                assert page.locator('input[name=choice]:checked').count() == 0
                if index == 0:
                    item = review['items'][0]
                    first_form = item['reference']['choices'][0]['value']
                    payload = {'id':item['id'], 'binding':item['binding'], 'revision':0, 'choice':first_form,
                               'note':'QA, must be rejected without whole-form confirmation'}
                    assert request('/api/answer', payload)[0] == 400
                    assert request('/api/state')[1]['answered'] == 0
                    page.locator(f'input[name=choice][value={first_form}]').check()
                    assert page.locator('#save').is_disabled()
                    page.locator('#whole-confirmed').check()
                    assert page.locator('#save').is_enabled()
                    page.locator('#note').fill('QA whole-form selection, not a human label')
                    page.screenshot(path=str(qa / 'whole-form-desktop.png'), full_page=True)
                    page.set_viewport_size({'width':390, 'height':844})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    page.screenshot(path=str(qa / 'whole-form-mobile.png'), full_page=True)
                    page.set_viewport_size({'width':1100, 'height':1000})
                    report['checks']['whole_form_requires_explicit_confirmation_in_browser_and_api'] = True
                    report['checks']['mobile_no_horizontal_overflow'] = True
                else:
                    page.locator('input[name=choice][value=ambiguous]').check()
                    assert page.locator('#whole-check').is_hidden()
                if page.locator('#original').is_visible():
                    page.locator('#original summary').click()
                    page.locator('#full').evaluate('(a)=>{a.playbackRate=8; return a.play();}')
                    page.wait_for_function('!document.querySelector("#crop-choices").disabled', timeout=15000)
                    assert page.locator('#full').evaluate('(a)=>a.ended && !a.error')
                    originals_played += 1
                page.locator('#save').click()
                page.wait_for_function('(n)=>document.querySelector("#progress").value===n', arg=index+1)
                if index == 0:
                    page.reload()
                    page.wait_for_function('document.querySelector("#position").textContent.startsWith("2번째")')
                    report['checks']['reload_resumes_unanswered_question'] = True
            page.wait_for_selector('#done:not([hidden])')
            page.reload()
            page.wait_for_selector('#done:not([hidden])')
            page.screenshot(path=str(qa / 'complete.png'), full_page=True)
            page.locator('#review-again').click()
            page.wait_for_selector('#comparison:not([hidden])')
            assert page.locator('#whole-confirmed').is_checked()
            assert page.locator('#first-heard').input_value() == 'QA only, not a listener answer'
            assert page.locator('#note').input_value() == 'QA whole-form selection, not a human label'
            page.locator('input[name=choice][value=partial]').check()
            assert page.locator('#whole-check').is_hidden() and not page.locator('#whole-confirmed').is_checked()
            page.locator('#note').fill('QA revision, partial only')
            page.locator('#save').click()
            page.wait_for_selector('#done:not([hidden])')
            export = request('/api/export')[1]
            assert export['answered'] == 15 and len(export['answer_history']) == 16
            assert export['items'][0]['answer']['choice'] == 'partial'
            assert not report['errors'], report['errors']
            report['checks']['all_15_saved_and_completed_state_persisted'] = True
            report['checks']['revision_preserved_and_whole_confirmation_resets_on_choice_change'] = True
            report['checks']['real_clip_playbacks'] = 15
            report['checks']['real_full_source_playbacks'] = originals_played
            assert originals_played == 9
            browser.close()
        report['complete'] = True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        report['page_sha256'] = hashlib.sha256(PAGE.read_bytes()).hexdigest()
        report['server_sha256'] = hashlib.sha256(Path(__file__).with_name('server.py').read_bytes()).hexdigest()
        (qa / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
