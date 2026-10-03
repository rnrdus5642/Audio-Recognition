"""Real playback and browser persistence checks in an isolated QA database."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import urllib.request

from playwright.sync_api import sync_playwright
from server import DEFAULT_DIRECTORY, make_server


def main():
    qa = DEFAULT_DIRECTORY / 'qa' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    qa.mkdir(parents=True)
    server = make_server(port=0, db_path=qa / 'qa-answers.sqlite3')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    report = {'user_answer_database_used': False, 'qa_directory': str(qa), 'errors': [], 'checks': {}}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel='msedge', headless=True,
                                         args=['--autoplay-policy=no-user-gesture-required'])
            page = browser.new_page(viewport={'width': 1100, 'height': 1000})
            page.on('pageerror', lambda error: report['errors'].append(str(error)))
            page.goto(base)
            page.wait_for_selector('#review:not([hidden])')
            assert page.locator('#reveal').is_disabled()
            assert page.locator('#comparison').is_hidden()
            assert page.locator('input[name=choice]').count() == 0
            page.screenshot(path=str(qa / 'before-listening-desktop.png'), full_page=True)
            report['checks']['comparison_hidden_before_playback'] = True
            for index in range(20):
                page.wait_for_function('(n)=>document.querySelector("#position").textContent.startsWith(`${n}번째`)', arg=index+1)
                if index == 0:
                    page.locator('#first-heard').fill('QA only: approximate first impression')
                page.locator('#train').evaluate('(a)=>{a.playbackRate=8; return a.play();}')
                page.wait_for_function('!document.querySelector("#reveal").disabled', timeout=30000)
                assert page.locator('#train').evaluate('(a)=>a.ended && !a.error')
                page.locator('#reveal').click()
                page.wait_for_selector('#comparison:not([hidden])')
                assert page.locator('input[name=choice]:checked').count() == 0
                if index == 0:
                    page.locator('#note').fill('QA only: this is not a human judgment')
                if index == 0 and page.locator('#original').is_visible():
                    page.locator('#original summary').click()
                    page.locator('#full').evaluate('(a)=>{a.playbackRate=8; return a.play();}')
                    page.wait_for_function('!document.querySelector("#crop-choices").disabled', timeout=30000)
                    page.locator('input[name=crop][value=uncertain]').check()
                    report['checks']['original_audio_playback_and_crop_choice'] = True
                if page.locator('.option.form').count() and 'contrasts_rendered' not in report['checks']:
                    page.screenshot(path=str(qa / 'comparison-desktop.png'), full_page=True)
                    page.set_viewport_size({'width':390,'height':844})
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    page.screenshot(path=str(qa / 'comparison-mobile.png'), full_page=True)
                    page.set_viewport_size({'width':1100,'height':1000})
                    report['checks']['contrasts_rendered'] = True
                    report['checks']['mobile_no_horizontal_overflow'] = True
                page.locator('input[name=choice][value=ambiguous]').check()
                page.locator('#save').click()
                page.wait_for_function('(n)=>document.querySelector("#progress").value===n', arg=index+1)
                if index == 0:
                    page.reload()
                    page.wait_for_function('document.querySelector("#position").textContent.startsWith("2번째")')
                    report['checks']['refresh_resumes_at_next_unanswered'] = True
            page.wait_for_selector('#done:not([hidden])')
            page.reload()
            page.wait_for_selector('#done:not([hidden])')
            page.screenshot(path=str(qa / 'complete.png'), full_page=True)
            report['checks']['all_20_saved_and_completion_survives_reload'] = True
            page.locator('#review-again').click()
            page.wait_for_selector('#comparison:not([hidden])')
            assert page.locator('#note').input_value() == 'QA only: this is not a human judgment'
            assert page.locator('#first-heard').input_value() == 'QA only: approximate first impression'
            page.locator('#note').fill('QA revised')
            page.locator('#save').click()
            page.wait_for_selector('#done:not([hidden])')
            report['checks']['saved_answer_can_be_revised'] = True
            with urllib.request.urlopen(base+'/api/export') as response:
                exported = json.load(response)
            assert exported['answered'] == 20 and len(exported['answer_history']) == 21
            assert not report['errors'], report['errors']
            report['checks']['all_audio_played_without_decode_error'] = True
            browser.close()
        report['complete'] = True
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        (qa / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
