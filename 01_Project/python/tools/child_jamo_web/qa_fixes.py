"""The three review fixes, checked in a real browser.

Narrow on purpose. The provenance, the 83-clip run and the whole-recording
coverage were established elsewhere and are not repeated here. What this
pass has to establish is the three things that were wrong:

  filters       a filter the page has hidden must also stop being applied,
                in both directions - a stale AIHub age made the personal
                clips come up empty with nothing on screen to explain it
  upload        an uploaded file must not inherit the previous clip's
                source recording, review flags or expected-answer note
  coverage      the whole-recording card must say a duration a listener
                can compare with the player, not a sample counter

and then that the whole-recording view still plays, still seeks and still
reports full coverage after those edits.

    .venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe \
        python/tools/child_jamo_web/qa_fixes.py
"""
from pathlib import Path
import json
import sys
import time
import urllib.parse
import urllib.request

from playwright.sync_api import sync_playwright

BASE = 'http://127.0.0.1:8767/'
WORK = Path(__file__).resolve().parents[3] / '.venv/child-jamo-web-20260916'
SHOTS = WORK / 'screenshots'
OUT = WORK / 'personal-recordings/qa-final-fixes.json'

PLAY = """(seekTo) => new Promise(resolve => {
    const a = document.getElementById('player');
    const out = {error: null, duration: 0, start: 0, end: 0};
    a.onerror = () => { out.error = 'media error ' + (a.error && a.error.code); resolve(out); };
    const go = () => {
        if (seekTo != null) a.currentTime = seekTo;
        out.duration = a.duration; out.start = a.currentTime;
        a.play().then(() => {
            setTimeout(() => { out.end = a.currentTime; a.pause(); resolve(out); }, 1400);
        }).catch(e => { out.error = String(e); resolve(out); });
    };
    if (a.readyState >= 2) go();
    else a.addEventListener('canplay', go, {once: true});
    setTimeout(() => resolve(out), 12000);
})"""


def played(result):
    return (result['error'] is None and result['duration'] > 0
            and result['end'] > result['start'])


def upload_fixture():
    """A real recording's WAV, fetched from the server, to upload back."""
    path = WORK / 'qa-upload.wav'
    if not path.exists():
        with urllib.request.urlopen(BASE + 'api/samples?split=test&size=1') as fh:
            sample_id = json.load(fh)['items'][0]['id']
        with urllib.request.urlopen(
                BASE + 'api/audio/' + urllib.parse.quote(sample_id)) as fh:
            path.write_bytes(fh.read())
    return path


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report = {'base': BASE, 'checks': {}, 'console_errors': [],
              'page_errors': [], 'failed_requests': [], 'screenshots': [],
              'scope': '리뷰에서 지적된 세 가지 수정만 확인합니다. 출처 확인, '
                       '83개 클립 추론, 원본 전체 포함은 이전 기록을 씁니다.',
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

        # --- 1. a corpus age filter must not follow us to the clips -----
        #
        # The two sources are separate pages now, so crossing between them
        # is a real navigation and the filters reset by themselves. The
        # guard is still worth holding: the age select exists on the
        # recordings page, hidden, and a value forced into it must not
        # quietly filter the clips - which is what emptied the list before.
        page.goto(BASE + 'aihub', wait_until='networkidle')
        page.wait_for_selector('#result:not([hidden])', timeout=300_000)
        page.select_option('#age', '5')
        page.wait_for_function(
            "() => !document.getElementById('result-count').textContent.startsWith('2,999')",
            timeout=60_000)
        aihub_filtered = page.inner_text('#result-count')
        report['aihub'] = {'count_with_age5': aihub_filtered,
                           'child_filter_hidden': page.is_hidden('#child-filter')}
        report['checks']['corpus_page_applies_its_visible_age_filter'] = (
            not aihub_filtered.startswith('2,999')
            and report['aihub']['child_filter_hidden'])

        page.click('#tab-recordings')
        page.wait_for_load_state('networkidle')
        page.wait_for_function(
            "() => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId.startsWith('personal:'); }",
            timeout=300_000)
        report['age_to_personal'] = {
            'url': page.url,
            'age_filter_hidden': page.is_hidden('#age-filter'),
            'personal_count': page.inner_text('#result-count'),
            'rows': page.locator('#sample-list button').count(),
            'result_id': page.get_attribute('#result', 'data-sample-id'),
            'prediction': page.inner_text('#pred-text'),
        }
        entry = report['age_to_personal']
        report['checks']['navigating_to_the_clips_leaves_the_age_filter_behind'] = (
            entry['age_filter_hidden'] and entry['personal_count'].startswith('83')
            and entry['rows'] > 0)
        report['checks']['personal_still_transcribes_after_the_switch'] = bool(
            entry['prediction'].strip())

        # Force the hidden select and make the list reload: a hidden filter
        # must still not be sent.
        page.evaluate("() => { document.getElementById('age').value = '5'; }")
        page.select_option('#child', 'child2')
        page.wait_for_function(
            "() => document.getElementById('result-count').textContent.startsWith('35')",
            timeout=60_000)
        report['forced_hidden_age'] = {
            'age_value': page.input_value('#age'),
            'count': page.inner_text('#result-count'),
        }
        report['checks']['hidden_age_filter_is_not_applied'] = (
            report['forced_hidden_age']['age_value'] == '5'
            and report['forced_hidden_age']['count'].startswith('35'))

        # --- 2. the child filter is this page's own, and stays visible ---
        personal_child2 = page.inner_text('#result-count')
        page.select_option('#split', 'original')
        page.wait_for_function(
            "() => document.getElementById('result-count').textContent.startsWith('3')"
            "   || document.getElementById('result-count').textContent.startsWith('4')",
            timeout=120_000)
        report['child_to_original'] = {
            'count': page.inner_text('#result-count'),
            'child_filter_visible': not page.is_hidden('#child-filter'),
        }
        # The child filter is visible here, so it is meant to apply: child2
        # owns three of the four recordings.
        report['checks']['child_filter_stays_visible_across_personal_datasets'] = (
            report['child_to_original']['child_filter_visible']
            and report['child_to_original']['count'].startswith('3'))
        page.select_option('#child', '')
        page.wait_for_function(
            "() => document.getElementById('result-count').textContent.startsWith('4')",
            timeout=60_000)

        page.select_option('#child', 'child2')
        page.wait_for_timeout(500)
        page.click('#tab-aihub')
        page.wait_for_load_state('networkidle')
        page.wait_for_function(
            "() => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId.startsWith('test:'); }",
            timeout=300_000)
        report['child_to_aihub'] = {
            'personal_child2_count': personal_child2,
            'url': page.url,
            'child_filter_hidden': page.is_hidden('#child-filter'),
            'child_select_value': page.input_value('#child'),
            'age_filter_visible': not page.is_hidden('#age-filter'),
            'count': page.inner_text('#result-count'),
        }
        back = report['child_to_aihub']
        report['checks']['hidden_child_filter_is_not_applied'] = (
            back['child_filter_hidden'] and back['count'].startswith('2,999'))
        report['checks']['corpus_filters_are_the_ones_on_offer_again'] = (
            back['age_filter_visible'] and page.input_value('#age') == '')
        report['checks']['navigation_resets_the_other_sources_filters'] = \
            back['count'].startswith('2,999')

        # --- 3. an upload must not wear the previous clip's provenance ---
        page.goto(BASE + 'recordings', wait_until='networkidle')
        page.wait_for_function(
            "() => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId.startsWith('personal:'); }",
            timeout=300_000)
        # The child select still holds child2 from the step above and is
        # visible again here, so it applies - clear it, or the one clip
        # that carries an expected-answer annotation is filtered out.
        page.select_option('#child', '')
        page.fill('#query', '야옹이')
        page.wait_for_timeout(800)
        # By id, not by position: only child1_06 has the annotation.
        annotated_id = 'personal:child1_06'
        row = page.locator(f'#sample-list button[data-id="{annotated_id}"]')
        report['annotated_row_found'] = row.count()
        row.click()
        page.wait_for_function(
            "id => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId === id; }",
            arg=annotated_id, timeout=300_000)
        before = {
            'id': annotated_id,
            'title': page.inner_text('#sample-title'),
            'source': page.inner_text('#sample-source'),
            'meta': page.inner_text('#sample-meta'),
        }
        report['before_upload'] = before
        # The clip this is worth testing against is the one carrying an
        # expected-answer annotation and a review flag.
        report['checks']['annotated_clip_shows_its_own_provenance'] = (
            '기대 정답' in before['source'] and '출처' in before['source'])

        fixture = upload_fixture()
        page.locator('details.upload').click()
        page.wait_for_timeout(200)
        page.set_input_files('#file', str(fixture))
        page.click('#upload')
        page.wait_for_function(
            "() => document.getElementById('upload-note').textContent.includes('인식했습니다')",
            timeout=300_000)
        after = {
            'title': page.inner_text('#sample-title'),
            'meta': page.inner_text('#sample-meta'),
            'source_hidden': page.is_hidden('#sample-source'),
            'source_text': page.inner_text('#sample-source'),
            'prediction': page.inner_text('#pred-text'),
            'transcribe_label': page.inner_text('#transcribe'),
            'prev_disabled': page.is_disabled('#prev-sample'),
            'next_disabled': page.is_disabled('#next-sample'),
        }
        report['after_upload'] = after
        report['checks']['upload_drops_the_previous_clip_provenance'] = (
            after['source_hidden'] and not after['source_text'].strip())
        report['checks']['upload_shows_its_own_file_not_the_clip'] = (
            after['title'] == fixture.name
            and '내가 올린 파일' in after['meta']
            and before['title'] not in after['title'])
        report['checks']['upload_carries_no_expected_answer_note'] = (
            '기대 정답' not in after['source_text']
            and '기대 정답' not in after['meta'])
        report['checks']['upload_clears_stale_navigation'] = (
            after['prev_disabled'] and after['next_disabled'])
        report['checks']['upload_resets_the_action_label'] = \
            after['transcribe_label'].strip() == '이 음성 인식하기'
        report['checks']['upload_still_transcribes'] = bool(
            after['prediction'].strip())

        # --- 4. the whole recordings still play, seek and cover it all ---
        page.fill('#query', '')
        page.wait_for_timeout(600)
        page.select_option('#split', 'original')
        page.wait_for_function(
            "() => { const r = document.getElementById('original-result');"
            "        return !r.hidden && r.dataset.sampleId; }",
            timeout=600_000)
        segments = page.locator('.segment')
        coverage = page.inner_text('#original-coverage')
        report['original'] = {
            'coverage': coverage,
            'segments': segments.count(),
            'first_segment': segments.nth(0).inner_text(),
            'source_shown': page.inner_text('#sample-source'),
        }
        report['checks']['coverage_reads_as_a_duration'] = (
            '전체 포함' in coverage and '빠짐없이' in coverage
            and '샘플' not in coverage)
        report['checks']['coverage_has_no_raw_sample_counter'] = \
            '1,840,912' not in coverage and '918' not in coverage
        report['checks']['whole_recording_still_chunked'] = \
            report['original']['segments'] >= 5

        playback = page.evaluate(PLAY, None)
        report['original']['playback'] = playback
        report['checks']['whole_recording_still_plays'] = played(playback)

        target = segments.nth(min(3, report['original']['segments'] - 1))
        seek_to = float(target.locator('button').get_attribute('data-at'))
        target.locator('button').click()
        page.wait_for_timeout(1500)
        moved = page.evaluate("""() => {
            const a = document.getElementById('player');
            const out = {at: a.currentTime, error: a.error && a.error.code};
            a.pause(); return out;
        }""")
        report['original']['seek'] = {'asked': seek_to, **moved}
        report['checks']['segment_button_still_seeks_the_original'] = (
            moved['error'] is None and moved['at'] >= seek_to
            and moved['at'] < seek_to + 8)
        page.screenshot(path=str(SHOTS / '10-final-original-segments.png'),
                        full_page=True)
        report['screenshots'].append('10-final-original-segments.png')

        # --- 5. final look at the clips, as the user will open them ------
        page.goto(BASE + 'recordings', wait_until='networkidle')
        page.wait_for_selector('#result:not([hidden])', timeout=300_000)
        report['final_personal'] = {
            'count': page.inner_text('#result-count'),
            'meta': page.inner_text('#sample-meta'),
            'source': page.inner_text('#sample-source'),
            'age_filter_hidden': page.is_hidden('#age-filter'),
        }
        report['checks']['personal_opens_clean_from_its_own_page'] = (
            report['final_personal']['count'].startswith('83')
            and report['final_personal']['age_filter_hidden'])
        page.screenshot(path=str(SHOTS / '09-final-personal-clips.png'),
                        full_page=True)
        report['screenshots'].append('09-final-personal-clips.png')

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
