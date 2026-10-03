"""The two source pages, in a real browser.

The app now has one page per source, at `/aihub` and `/recordings`. What
this pass has to establish:

  separation    each page offers only its own datasets, with only the
                filters that mean something there, and the corpus's
                recorded aggregate appears on the corpus page only
  addressing    deep links, refresh, back and forward, the links that
                predate the split, and a split that belongs to the other
                source - which must never show that source's data
  history       what each page lists is its own runs, not the other's
  and that a real recording from each source still plays and transcribes.

The heavier evidence - provenance, the 83-clip run, whole-recording
coverage - was established before and is not repeated here.

    .venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe \
        python/tools/child_jamo_web/qa_sources.py
"""
from pathlib import Path
import json
import sys
import time

from playwright.sync_api import sync_playwright

BASE = 'http://127.0.0.1:8767/'
WORK = Path(__file__).resolve().parents[3] / '.venv/child-jamo-web-20260916'
SHOTS = WORK / 'screenshots'
OUT = WORK / 'personal-recordings/qa-sources.json'

PLAY = """() => new Promise(resolve => {
    const a = document.getElementById('player');
    const out = {error: null, duration: 0, start: 0, end: 0};
    a.onerror = () => { out.error = 'media error ' + (a.error && a.error.code); resolve(out); };
    const go = () => {
        out.duration = a.duration; out.start = a.currentTime;
        a.play().then(() => {
            setTimeout(() => { out.end = a.currentTime; a.pause(); resolve(out); }, 1400);
        }).catch(e => { out.error = String(e); resolve(out); });
    };
    if (a.readyState >= 2) go();
    else a.addEventListener('canplay', go, {once: true});
    setTimeout(() => resolve(out), 12000);
})"""

OPTIONS = """() => Array.from(document.getElementById('split').options)
                      .map(o => o.value)"""


def played(result):
    return (result['error'] is None and result['duration'] > 0
            and result['end'] > result['start'])


def wait_ready(page, prefix, timeout=300_000):
    page.wait_for_function(
        "p => { const r = document.getElementById('result');"
        "       const o = document.getElementById('original-result');"
        "       return (!r.hidden && r.dataset.sampleId.startsWith(p))"
        "           || (!o.hidden && o.dataset.sampleId.startsWith(p)); }",
        arg=prefix, timeout=timeout)


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {'base': BASE, 'checks': {}, 'console_errors': [],
              'page_errors': [], 'failed_requests': [], 'screenshots': [],
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

        # --- 1. the corpus page ------------------------------------------
        page.goto(BASE + 'aihub', wait_until='networkidle')
        wait_ready(page, 'test:')
        report['aihub'] = {
            'url': page.url,
            'title': page.inner_text('#page-title'),
            'document_title': page.title(),
            'options': page.evaluate(OPTIONS),
            'split_value': page.input_value('#split'),
            'count': page.inner_text('#result-count'),
            'age_filter_visible': not page.is_hidden('#age-filter'),
            'child_filter_hidden': page.is_hidden('#child-filter'),
            'dataset_note_hidden': page.is_hidden('#dataset-note'),
            'facts': page.inner_text('#model-facts'),
            'active_tab': page.get_attribute('#tab-aihub', 'aria-current'),
            'other_tab': page.get_attribute('#tab-recordings', 'aria-current'),
            'transcript': page.inner_text('#ref-transcript'),
            'prediction': page.inner_text('#pred-text'),
            'metrics': page.inner_text('#metrics'),
        }
        a = report['aihub']
        report['checks']['aihub_offers_only_corpus_datasets'] = \
            a['options'] == ['test', 'valid', 'train']
        report['checks']['aihub_defaults_to_test'] = (
            a['split_value'] == 'test' and a['count'].startswith('2,999'))
        report['checks']['aihub_shows_age_not_child'] = (
            a['age_filter_visible'] and a['child_filter_hidden'])
        report['checks']['aihub_keeps_the_recorded_aggregate'] = \
            '2.89%' in a['facts'] and '전체 테스트 성적' in a['facts']
        report['checks']['aihub_tab_is_marked_current'] = (
            a['active_tab'] == 'page' and a['other_tab'] is None)
        report['checks']['aihub_recording_transcribes'] = bool(
            a['prediction'].strip()) and '자모 오류율' in a['metrics']
        report['checks']['aihub_page_has_its_own_title'] = \
            '말뭉치' in a['title'] and a['document_title'] == a['title']
        report['aihub']['playback'] = page.evaluate(PLAY)
        report['checks']['aihub_recording_plays'] = \
            played(report['aihub']['playback'])
        page.screenshot(path=str(SHOTS / '11-source-aihub.png'), full_page=True)
        report['screenshots'].append('11-source-aihub.png')

        # --- 2. the recordings page, reached by the tab -------------------
        page.click('#tab-recordings')
        page.wait_for_load_state('networkidle')
        wait_ready(page, 'personal:')
        report['recordings'] = {
            'url': page.url,
            'title': page.inner_text('#page-title'),
            'options': page.evaluate(OPTIONS),
            'split_value': page.input_value('#split'),
            'count': page.inner_text('#result-count'),
            'age_filter_hidden': page.is_hidden('#age-filter'),
            'child_filter_visible': not page.is_hidden('#child-filter'),
            'dataset_note': page.inner_text('#dataset-note'),
            'facts': page.inner_text('#model-facts'),
            'active_tab': page.get_attribute('#tab-recordings', 'aria-current'),
            'other_tab': page.get_attribute('#tab-aihub', 'aria-current'),
            'source': page.inner_text('#sample-source'),
            'prediction': page.inner_text('#pred-text'),
        }
        r = report['recordings']
        report['checks']['recordings_offers_only_personal_datasets'] = \
            r['options'] == ['personal', 'original']
        report['checks']['recordings_defaults_to_the_83_clips'] = (
            r['split_value'] == 'personal' and r['count'].startswith('83'))
        report['checks']['recordings_shows_child_not_age'] = (
            r['age_filter_hidden'] and r['child_filter_visible'])
        report['checks']['recordings_tab_is_marked_current'] = (
            r['active_tab'] == 'page' and r['other_tab'] is None)
        # The corpus aggregate is a corpus fact and stays on the corpus page.
        report['checks']['recordings_carries_no_corpus_aggregate'] = (
            '2.89' not in r['facts'] and '%' not in r['facts'])
        report['checks']['recordings_says_where_that_score_lives'] = (
            'AIHub' in r['facts'] and '정확도가 아니' in r['facts'])
        report['checks']['recordings_keeps_its_provenance_note'] = (
            '전사' in r['dataset_note'] and '출처' in r['source'])
        report['checks']['recordings_recording_transcribes'] = bool(
            r['prediction'].strip())
        report['recordings']['playback'] = page.evaluate(PLAY)
        report['checks']['recordings_recording_plays'] = \
            played(report['recordings']['playback'])
        report['checks']['tab_navigation_lands_on_the_right_url'] = \
            r['url'].startswith(BASE + 'recordings')
        page.screenshot(path=str(SHOTS / '12-source-recordings.png'),
                        full_page=True)
        report['screenshots'].append('12-source-recordings.png')

        # --- 3. the url follows the dataset, and survives a refresh -------
        page.select_option('#split', 'original')
        page.wait_for_function(
            "() => document.getElementById('result-count').textContent.startsWith('4')",
            timeout=120_000)
        page.select_option('#child', 'child2')
        page.wait_for_function(
            "() => document.getElementById('result-count').textContent.startsWith('3')",
            timeout=60_000)
        report['url_follows'] = {'after_dataset_change': page.url}
        report['checks']['url_follows_the_chosen_dataset'] = (
            'split=original' in page.url and 'child=child2' in page.url)

        page.reload(wait_until='networkidle')
        wait_ready(page, 'original:', timeout=600_000)
        report['url_follows']['after_reload'] = {
            'url': page.url,
            'split': page.input_value('#split'),
            'child': page.input_value('#child'),
            'count': page.inner_text('#result-count'),
            'segments': page.locator('.segment').count(),
        }
        after = report['url_follows']['after_reload']
        report['checks']['refresh_reopens_the_same_dataset'] = (
            after['split'] == 'original' and after['child'] == 'child2'
            and after['count'].startswith('3'))
        report['checks']['whole_recording_still_chunked_after_refresh'] = \
            after['segments'] >= 5

        # the segment button still seeks the whole recording
        target = page.locator('.segment').nth(2)
        seek_to = float(target.locator('button').get_attribute('data-at'))
        target.locator('button').click()
        page.wait_for_timeout(1500)
        moved = page.evaluate("""() => {
            const a = document.getElementById('player');
            const out = {at: a.currentTime, error: a.error && a.error.code};
            a.pause(); return out;
        }""")
        report['url_follows']['seek'] = {'asked': seek_to, **moved}
        report['checks']['segment_seeking_still_works'] = (
            moved['error'] is None and moved['at'] >= seek_to
            and moved['at'] < seek_to + 8)

        # --- 4. deep links, old links, and a conflicting split ------------
        report['links'] = {}
        for label, url, expect_prefix, expect_split in [
                ('recordings_original', BASE + 'recordings?split=original',
                 'original:', 'original'),
                ('aihub_valid', BASE + 'aihub?split=valid', 'valid:', 'valid'),
                # Predates the split into two pages; must still land on the
                # personal recordings, and say so in the address bar.
                ('legacy_root_personal', BASE + '?split=personal',
                 'personal:', 'personal'),
                ('legacy_root_original', BASE + '?split=original',
                 'original:', 'original'),
                # A bare root is the corpus, as it always was.
                ('legacy_root_bare', BASE, 'test:', 'test'),
                # The split belongs to the other source: this page must show
                # its own default, never the other source's recordings.
                ('conflicting_aihub_personal', BASE + 'aihub?split=personal',
                 'test:', 'test'),
                ('conflicting_recordings_test',
                 BASE + 'recordings?split=test', 'personal:', 'personal')]:
            page.goto(url, wait_until='networkidle')
            wait_ready(page, expect_prefix, timeout=600_000)
            report['links'][label] = {
                'asked': url, 'landed': page.url,
                'split': page.input_value('#split'),
                'options': page.evaluate(OPTIONS),
                'count': page.inner_text('#result-count'),
            }
            got = report['links'][label]
            report['checks'][f'link_{label}'] = got['split'] == expect_split
        # and the conflicting ones must not even be able to reach the other
        # source's datasets from the picker
        report['checks']['conflicting_split_cannot_reach_other_source'] = (
            'personal' not in report['links']['conflicting_aihub_personal']['options']
            and 'test' not in
            report['links']['conflicting_recordings_test']['options'])

        # --- 5. back and forward across the two pages ---------------------
        page.goto(BASE + 'aihub', wait_until='networkidle')
        wait_ready(page, 'test:')
        page.click('#tab-recordings')
        page.wait_for_load_state('networkidle')
        wait_ready(page, 'personal:')
        page.go_back()
        page.wait_for_load_state('networkidle')
        wait_ready(page, 'test:')
        report['back_forward'] = {'after_back': page.url,
                                  'back_split': page.input_value('#split')}
        page.go_forward()
        page.wait_for_load_state('networkidle')
        wait_ready(page, 'personal:')
        report['back_forward'].update({'after_forward': page.url,
                                       'forward_split': page.input_value('#split')})
        bf = report['back_forward']
        report['checks']['back_returns_to_the_corpus_page'] = (
            'aihub' in bf['after_back'] and bf['back_split'] == 'test')
        report['checks']['forward_returns_to_the_recordings_page'] = (
            'recordings' in bf['after_forward'] and bf['forward_split'] == 'personal')

        # --- 6. each page lists its own runs only -------------------------
        # A corpus run and a personal run have both happened by now.
        personal_history = page.evaluate(
            """async () => (await (await fetch('/api/history?source=recordings')).json()).items""")
        corpus_history = page.evaluate(
            """async () => (await (await fetch('/api/history?source=aihub')).json()).items""")
        report['history'] = {
            'recordings_splits': sorted({i['split'] for i in personal_history}),
            'corpus_splits': sorted({i['split'] for i in corpus_history}),
            'recordings_rows': len(personal_history),
            'corpus_rows': len(corpus_history),
        }
        h = report['history']
        report['checks']['recordings_history_holds_only_its_own'] = (
            h['recordings_rows'] > 0
            and set(h['recordings_splits']) <= {'personal', 'original'})
        report['checks']['corpus_history_holds_only_its_own'] = (
            h['corpus_rows'] > 0
            and set(h['corpus_splits']) <= {'test', 'valid', 'train'})

        page.locator('details.details', has_text='지금까지 인식한 기록').click()
        page.wait_for_timeout(600)
        shown = page.inner_text('#history')
        report['history']['shown_on_recordings'] = shown[:400]
        report['checks']['recordings_page_shows_no_corpus_rows'] = (
            '테스트' not in shown and '검증' not in shown)

        # --- 7. the tabs on a phone --------------------------------------
        page.set_viewport_size({'width': 390, 'height': 844})
        page.wait_for_timeout(600)
        report['mobile'] = {
            'overflow': page.evaluate(
                '() => document.documentElement.scrollWidth'
                '    - document.documentElement.clientWidth'),
            'tabs_visible': page.is_visible('#tab-aihub')
            and page.is_visible('#tab-recordings'),
        }
        report['checks']['no_mobile_horizontal_overflow'] = \
            report['mobile']['overflow'] <= 0
        report['checks']['both_tabs_reachable_on_a_phone'] = \
            report['mobile']['tabs_visible']
        # the tab works at that size, not merely renders
        page.click('#tab-aihub')
        page.wait_for_load_state('networkidle')
        wait_ready(page, 'test:')
        report['mobile']['navigated_to'] = page.url
        report['checks']['mobile_tab_navigates'] = 'aihub' in page.url
        page.screenshot(path=str(SHOTS / '13-source-tabs-mobile.png'),
                        full_page=True)
        report['screenshots'].append('13-source-tabs-mobile.png')
        page.set_viewport_size({'width': 1440, 'height': 950})

        # --- 8. the tabs are reachable by keyboard -----------------------
        page.goto(BASE + 'recordings', wait_until='networkidle')
        wait_ready(page, 'personal:')
        focus = page.evaluate("""() => {
            const tab = document.getElementById('tab-aihub');
            tab.focus();
            const style = getComputedStyle(tab, ':focus-visible');
            return {id: document.activeElement.id,
                    outline: style.outlineWidth, tag: document.activeElement.tagName};
        }""")
        report['keyboard'] = focus
        report['checks']['tabs_take_keyboard_focus'] = (
            focus['id'] == 'tab-aihub' and focus['tag'] == 'A')

        report['checks']['no_page_exceptions'] = not report['page_errors']
        report['checks']['no_console_errors'] = not report['console_errors']
        aborted = [r for r in report['failed_requests']
                   if '/api/audio/' in r and 'ERR_ABORTED' in r]
        other = [r for r in report['failed_requests'] if r not in aborted]
        report['superseded_media_requests'] = aborted
        report['unexpected_failed_requests'] = other
        report['checks']['no_unexpected_failed_requests'] = not other
        browser.close()

    report['passed'] = all(report['checks'].values())
    report['failed_checks'] = [k for k, v in report['checks'].items() if not v]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    for key, value in report['checks'].items():
        print(f'  {"ok  " if value else "FAIL"} {key}')
    print('passed:', report['passed'], '->', OUT)
    for label in ('console_errors', 'page_errors', 'unexpected_failed_requests'):
        if report.get(label):
            print(label, report[label][:4])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
