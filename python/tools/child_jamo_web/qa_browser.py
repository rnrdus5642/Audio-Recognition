"""Drive the running viewer in a real browser, against the real model.

Run with the isolated QA interpreter, not the training environment:

    .venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe \
        python/tools/child_jamo_web/qa_browser.py

The assertions here are behavioural, not cosmetic. The two that matter
most:

  playback     the audio element is actually played and `currentTime`
               observed to advance with no media error - a decoded
               duration alone does not prove a sound came out
  association  every inference response is delayed by a fixed amount, so
               a reply for the previous sample is guaranteed to arrive
               after the user has moved on. The panel is then checked by
               the sample id it recorded, not by reading its text, and it
               must never show the abandoned sample.

Expected 4xx responses (the deliberately malformed upload) are separated
from unexpected page exceptions and network failures rather than being
ignored wholesale.
"""
from pathlib import Path
import json
import sys
import time

from playwright.sync_api import sync_playwright

BASE = 'http://127.0.0.1:8767/'
WORK = Path(__file__).resolve().parents[3] / '.venv/child-jamo-web-20260916'
SHOTS = WORK / 'screenshots'


class Hold:
    """Park matching requests unfulfilled until the test releases them.

    Sleeping inside a sync Playwright route handler blocks the driver
    itself, so the second click could not even be issued until the first
    reply had already been delivered - the overlap it was supposed to
    create never reliably happened. Here the handler stores the Route and
    returns immediately, so the request is genuinely still open while the
    test goes on clicking. Nothing about the response is altered: when
    released, the real server answer is passed through untouched.
    """

    def __init__(self, page, pattern, predicate=None):
        self.page, self.pattern, self.predicate = page, pattern, predicate
        self.held = []
        page.route(pattern, self._handler)

    def _handler(self, route):
        if self.predicate is None or self.predicate(route.request):
            self.held.append(route)
        else:
            route.continue_()

    def wait_until_pending(self, count=1, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.held) >= count:
                return True
            self.page.wait_for_timeout(100)
        return False

    @property
    def pending(self):
        return len(self.held)

    def release(self):
        released, self.held = self.held, []
        for route in released:
            route.continue_()
        return len(released)

    def close(self):
        self.release()
        self.page.unroute(self.pattern, self._handler)


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    report = {'base': BASE, 'checks': {}, 'console_errors': [],
              'page_errors': [], 'failed_requests': [], 'screenshots': [],
              'overlap_method': 'requests are held unfulfilled and released '
                                'explicitly; server responses are unaltered',
              'at': time.strftime('%Y-%m-%d %H:%M:%S')}

    with sync_playwright() as p:
        browser = None
        for channel in ('msedge', 'chrome'):
            try:
                browser = p.chromium.launch(channel=channel, headless=True)
                report['browser'] = channel
                break
            except Exception:
                continue
        if browser is None:
            raise SystemExit('설치된 Edge/Chrome 을 찾지 못했습니다')

        page = browser.new_page(viewport={'width': 1440, 'height': 950})
        page.on('console', lambda m: m.type == 'error'
                and report['console_errors'].append(m.text))
        page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
        page.on('requestfailed', lambda r: report['failed_requests'].append(
            f'{r.method} {r.url} {r.failure}'))

        # 1. first paint: a real sample, already transcribed
        page.goto(BASE, wait_until='networkidle')
        page.wait_for_selector('#result:not([hidden])', timeout=180_000)
        report['model_status'] = page.inner_text('#model-status-text')
        report['first_transcript'] = page.inner_text('#ref-transcript')
        report['first_surface'] = page.inner_text('#ref-surface')
        report['first_prediction'] = page.inner_text('#pred-text')
        report['first_meta'] = page.inner_text('#sample-meta')
        report['checks']['default_sample_transcribed_on_load'] = bool(
            report['first_prediction'].strip())
        report['checks']['model_reports_ready'] = '준비 완료' in report['model_status']
        report['checks']['selected_sample_shows_its_split'] = \
            report['first_meta'].startswith('테스트')
        report['checks']['dataset_label_not_truncated'] = page.evaluate(
            """() => {
                const s = document.getElementById('split');
                return s.options[s.selectedIndex].text.includes('개');
            }""")

        # 2. playback really happens
        playback = page.evaluate("""() => new Promise(resolve => {
            const a = document.getElementById('player');
            const out = {error: null, duration: 0, start: 0, end: 0};
            a.onerror = () => { out.error = 'media error ' + (a.error && a.error.code); resolve(out); };
            const go = () => {
                out.duration = a.duration; out.start = a.currentTime;
                a.play().then(() => {
                    setTimeout(() => {
                        out.end = a.currentTime; a.pause(); resolve(out);
                    }, 1200);
                }).catch(e => { out.error = String(e); resolve(out); });
            };
            if (a.readyState >= 2) go();
            else a.addEventListener('canplay', go, {once: true});
            setTimeout(() => resolve(out), 9000);
        })""")
        report['playback'] = playback
        report['checks']['audio_actually_plays'] = (
            playback['error'] is None and playback['duration'] > 0
            and playback['end'] > playback['start'])

        # 3. search narrows the list, and selecting transcribes (the UX)
        page.fill('#query', '사과')
        page.wait_for_timeout(700)
        report['search_result_count'] = page.inner_text('#result-count')
        rows = page.locator('#sample-list button')
        report['checks']['search_filters_list'] = rows.count() > 0
        rows.nth(0).click()
        page.wait_for_selector('#result:not([hidden])', timeout=120_000)
        report['searched_transcript'] = page.inner_text('#ref-transcript')
        report['searched_prediction'] = page.inner_text('#pred-text')
        report['checks']['selecting_a_sample_transcribes_it'] = bool(
            report['searched_prediction'].strip())
        page.screenshot(path=str(SHOTS / '02-desktop-search-result.png'),
                        full_page=True)
        report['screenshots'].append('02-desktop-search-result.png')

        page.fill('#query', '')
        page.wait_for_timeout(800)

        # 4. true overlap: A's request is held open, B is chosen while it is
        #    still pending, then A is released. B must end up with its own
        #    visible result - not merely "A was hidden".
        hold = Hold(page, '**/api/transcribe')
        rows = page.locator('#sample-list button')
        first_id = rows.nth(0).get_attribute('data-id')
        second_id = rows.nth(3).get_attribute('data-id')
        report['association'] = {'first': first_id, 'second': second_id,
                                 'distinct_ids': first_id != second_id}
        rows.nth(0).click()
        report['association']['a_request_pending_before_switch'] = \
            hold.wait_until_pending(1)
        rows.nth(3).click()                       # B chosen, A still open
        page.wait_for_timeout(300)
        midway = page.evaluate("""() => {
            const r = document.getElementById('result');
            return {hidden: r.hidden, sampleId: r.dataset.sampleId || ''};
        }""")
        report['association']['midway'] = midway
        report['association']['pending_when_b_chosen'] = hold.pending
        report['checks']['a_request_still_open_when_b_selected'] = (
            report['association']['a_request_pending_before_switch']
            and midway['sampleId'] != first_id)
        released = hold.release()                 # let A's real answer through
        report['association']['released'] = released
        page.wait_for_timeout(600)
        after_release = page.get_attribute('#result', 'data-sample-id') or ''
        report['association']['after_release'] = after_release
        report['checks']['abandoned_reply_never_painted'] = \
            after_release != first_id
        # B's own inference must still happen and be displayed.
        hold.close()
        page.wait_for_function(
            "id => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId === id; }",
            arg=second_id, timeout=120_000)
        report['association']['final'] = page.get_attribute('#result', 'data-sample-id')
        report['checks']['b_selected_during_a_gets_its_own_result'] = \
            report['association']['final'] == second_id

        # 4b. a real click on the button must produce a new request/result
        before_attempt = page.get_attribute('#result', 'data-attempt')
        page.click('#transcribe')
        page.wait_for_function(
            "prev => { const r = document.getElementById('result');"
            "          return !r.hidden && r.dataset.attempt !== prev; }",
            arg=before_attempt, timeout=120_000)
        report['manual_retranscribe'] = {
            'attempt_before': before_attempt,
            'attempt_after': page.get_attribute('#result', 'data-attempt'),
            'sample_id': page.get_attribute('#result', 'data-sample-id'),
            'note': page.inner_text('#action-note'),
        }
        report['checks']['manual_button_issues_a_new_transcription'] = (
            report['manual_retranscribe']['attempt_after'] != before_attempt
            and report['manual_retranscribe']['sample_id'] == second_id)

        # 5. jamo detail
        page.locator('details.details').first.click()
        page.wait_for_timeout(300)
        report['diff_segments'] = page.locator('#diff span').count()
        report['checks']['jamo_diff_rendered'] = report['diff_segments'] > 0
        report['metrics_text'] = page.inner_text('#metrics')
        report['checks']['metrics_say_boundaries_excluded'] = \
            '띄어쓰기 제외' in report['metrics_text']
        page.screenshot(path=str(SHOTS / '03-desktop-jamo-diff.png'), full_page=True)
        report['screenshots'].append('03-desktop-jamo-diff.png')

        # 6. batch, and its association guard
        page.click('#batch')
        page.wait_for_selector('#batch-result:not([hidden])', timeout=300_000)
        report['batch_rows'] = page.locator('.batch-row').count()
        report['batch_aggregate'] = page.inner_text('#batch-aggregate')
        report['checks']['batch_runs'] = report['batch_rows'] > 0
        report['checks']['batch_reports_scope'] = '인식 성공 / 요청' in report['batch_aggregate']
        page.screenshot(path=str(SHOTS / '04-desktop-batch.png'), full_page=True)
        report['screenshots'].append('04-desktop-batch.png')

        # selecting a sample must clear the batch panel, not leave it stale
        page.locator('#sample-list button').nth(1).click()
        page.wait_for_timeout(500)
        report['checks']['batch_panel_cleared_on_new_selection'] = page.evaluate(
            "() => document.getElementById('batch-result').hidden")
        page.wait_for_selector('#result:not([hidden])', timeout=120_000)

        # 6b. a batch still running when the user picks a sample must not
        #     paint itself afterwards
        batch_hold = Hold(page, '**/api/batch')
        page.click('#batch')
        report['batch_race'] = {'pending': batch_hold.wait_until_pending(1)}
        target = page.locator('#sample-list button').nth(2)
        batch_target_id = target.get_attribute('data-id')
        target.click()
        page.wait_for_timeout(300)
        batch_hold.release()
        page.wait_for_timeout(900)
        batch_hold.close()
        report['batch_race'].update({
            'selected': batch_target_id,
            'batch_panel_hidden': page.evaluate(
                "() => document.getElementById('batch-result').hidden"),
        })
        page.wait_for_function(
            "id => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId === id; }",
            arg=batch_target_id, timeout=120_000)
        report['checks']['late_batch_does_not_override_selection'] = (
            report['batch_race']['pending']
            and report['batch_race']['batch_panel_hidden'])

        # 6c. a filter reply that arrives after a newer filter must lose
        filter_hold = Hold(page, '**/api/samples**')
        page.fill('#query', '사과')
        report['filter_race'] = {'first_pending': filter_hold.wait_until_pending(1)}
        page.wait_for_timeout(400)
        page.fill('#query', '')
        page.wait_for_timeout(400)
        filter_hold.release()
        page.wait_for_timeout(1200)
        filter_hold.close()
        page.wait_for_timeout(1200)
        report['filter_race']['final_count'] = page.inner_text('#result-count')
        report['checks']['late_filter_reply_does_not_win'] = \
            '2,999' in report['filter_race']['final_count']

        # 7. layout
        report['desktop_overflow'] = page.evaluate(
            '() => document.documentElement.scrollWidth - document.documentElement.clientWidth')
        report['checks']['no_desktop_horizontal_overflow'] = \
            report['desktop_overflow'] <= 0
        page.screenshot(path=str(SHOTS / '01-desktop-first-load.png'), full_page=True)
        report['screenshots'].append('01-desktop-first-load.png')

        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(600)
        report['mobile_overflow'] = page.evaluate(
            '() => document.documentElement.scrollWidth - document.documentElement.clientWidth')
        report['checks']['no_mobile_horizontal_overflow'] = \
            report['mobile_overflow'] <= 0
        page.screenshot(path=str(SHOTS / '05-mobile.png'), full_page=True)
        report['screenshots'].append('05-mobile.png')
        page.set_viewport_size({'width': 1440, 'height': 950})

        # 8. upload: refusal, then a real file, then its association guard
        bad = WORK / 'qa-bad-upload.txt'
        bad.write_text('not audio', encoding='utf-8')
        page.locator('details.upload').click()
        page.wait_for_timeout(200)
        page.set_input_files('#file', str(bad))
        page.click('#upload')
        page.wait_for_timeout(1500)
        report['upload_rejection_note'] = page.inner_text('#upload-note')
        report['checks']['upload_rejects_wrong_format'] = \
            '형식만' in report['upload_rejection_note']

        import urllib.parse
        import urllib.request
        with urllib.request.urlopen(BASE + 'api/samples?split=test&size=1') as fh:
            sample_id = json.load(fh)['items'][0]['id']
        wav = WORK / 'qa-upload.wav'
        with urllib.request.urlopen(
                BASE + 'api/audio/' + urllib.parse.quote(sample_id)) as fh:
            wav.write_bytes(fh.read())
        page.set_input_files('#file', str(wav))
        page.click('#upload')
        page.wait_for_function(
            "() => document.getElementById('upload-note').textContent.includes('인식했습니다')",
            timeout=120_000)
        report['upload_note'] = page.inner_text('#upload-note')
        report['upload_prediction'] = page.inner_text('#pred-text')
        report['checks']['upload_transcribes'] = bool(
            report['upload_prediction'].strip())
        page.screenshot(path=str(SHOTS / '06-desktop-upload.png'), full_page=True)
        report['screenshots'].append('06-desktop-upload.png')

        # a sample chosen while an upload is genuinely still open must win
        upload_hold = Hold(page, '**/api/upload')
        page.set_input_files('#file', str(wav))
        page.click('#upload')
        upload_pending = upload_hold.wait_until_pending(1)
        target = page.locator('#sample-list button').nth(2)
        target_id = target.get_attribute('data-id')
        target.click()
        page.wait_for_timeout(300)
        upload_hold.release()
        page.wait_for_timeout(900)
        upload_hold.close()
        page.wait_for_function(
            "id => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId === id; }",
            arg=target_id, timeout=120_000)
        report['upload_race'] = {
            'upload_pending_when_sample_chosen': upload_pending,
            'selected': target_id,
            'result_sample_id': page.get_attribute('#result', 'data-sample-id'),
            'title': page.inner_text('#sample-title'),
        }
        report['checks']['late_upload_does_not_override_selection'] = (
            upload_pending
            and report['upload_race']['result_sample_id'] == target_id)

        # 8b. the lazily parsed train split must be browsable, and switching
        #     to it must not auto-select a row left over from the old split
        report['splits_browsed'] = {}
        for split in ('valid', 'train'):
            page.select_option('#split', split)
            page.wait_for_function(
                "s => document.getElementById('sample-meta').textContent"
                "      .startsWith(s)",
                arg={'valid': '검증', 'train': '학습'}[split], timeout=300_000)
            page.wait_for_function(
                "() => { const r = document.getElementById('result');"
                "        return !r.hidden && r.dataset.sampleId; }",
                timeout=300_000)
            report['splits_browsed'][split] = {
                'meta': page.inner_text('#sample-meta')[:60],
                'count': page.inner_text('#result-count'),
                'result_id': page.get_attribute('#result', 'data-sample-id'),
                'prediction': page.inner_text('#pred-text')[:40],
            }
        report['checks']['valid_split_browsable'] = \
            report['splits_browsed']['valid']['result_id'].startswith('valid:')
        report['checks']['train_split_browsable'] = \
            report['splits_browsed']['train']['result_id'].startswith('train:')
        page.select_option('#split', 'test')
        page.wait_for_function(
            "() => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId.startsWith('test:'); }",
            timeout=300_000)
        report['checks']['returns_to_test_split'] = \
            page.get_attribute('#result', 'data-sample-id').startswith('test:')

        # 9. console hygiene, with the deliberate refusal separated out
        expected = [e for e in report['console_errors']
                    if '400 (Bad Request)' in e]
        unexpected = [e for e in report['console_errors'] if e not in expected]
        report['expected_console_errors'] = expected
        report['unexpected_console_errors'] = unexpected
        report['checks']['no_unexpected_console_errors'] = not unexpected
        report['checks']['no_page_exceptions'] = not report['page_errors']

        # Selecting another sample replaces the <audio> src, and the browser
        # cancels the download that is already in flight. That abort is the
        # old request being superseded, not a failure: the server logs no
        # error for it and `audio_actually_plays` proves playback works.
        # Narrow on purpose - any other failed request still fails the run.
        aborted_media = [r for r in report['failed_requests']
                         if '/api/audio/' in r and 'ERR_ABORTED' in r]
        other_failures = [r for r in report['failed_requests']
                          if r not in aborted_media]
        report['superseded_media_requests'] = aborted_media
        report['unexpected_failed_requests'] = other_failures
        report['checks']['no_unexpected_failed_requests'] = not other_failures
        browser.close()

    report['passed'] = all(report['checks'].values())
    report['failed_checks'] = [k for k, v in report['checks'].items() if not v]
    (WORK / 'qa-browser.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for key, value in report['checks'].items():
        print(f'  {"ok  " if value else "FAIL"} {key}')
    print('passed:', report['passed'])
    if report.get('superseded_media_requests'):
        print('superseded media fetches (expected):',
              len(report['superseded_media_requests']))
    for label in ('unexpected_console_errors', 'page_errors',
                  'unexpected_failed_requests'):
        if report.get(label):
            print(label, report[label][:4])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
