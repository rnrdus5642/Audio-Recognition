"""Drive the personally recorded datasets in a real browser.

Companion to `qa_browser.py`, which covers the corpus splits. What this one
has to establish is narrow and specific:

  the page can be opened straight onto the personal clips by URL
  each child can be picked out, and a clip of each is played for real
  a whole recording plays, is recognised in timed chunks, and a chunk
    button actually seeks the whole-file player to that time
  a recording with no reference says so, and shows no error rate

    .venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe \
        python/tools/child_jamo_web/qa_personal.py
"""
from pathlib import Path
import json
import sys
import time

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
#: The same request-holding helper the corpus pass uses: a route handler
#: that parks the Route and returns, so the request is genuinely still open
#: while the test keeps clicking. Sleeping in the handler would block the
#: driver instead and never create the overlap.
from qa_browser import Hold

BASE = 'http://127.0.0.1:8767/'
WORK = Path(__file__).resolve().parents[3] / '.venv/child-jamo-web-20260916'
SHOTS = WORK / 'screenshots'
OUT = WORK / 'personal-recordings/qa-personal.json'

#: Playing an audio element and watching `currentTime` move is the only
#: way to know a sound came out; a decoded duration proves nothing.
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

        # 1. the old link still opens the personal clips, already recognised
        #    (it predates the split into two pages, so it is kept working)
        page.goto(BASE + '?split=personal', wait_until='networkidle')
        page.wait_for_selector('#result:not([hidden])', timeout=300_000)
        report['opened'] = {
            'split_value': page.input_value('#split'),
            'count': page.inner_text('#result-count'),
            'meta': page.inner_text('#sample-meta'),
            'source': page.inner_text('#sample-source'),
            'note': page.inner_text('#dataset-note'),
            'transcript': page.inner_text('#ref-transcript'),
            'surface': page.inner_text('#ref-surface'),
            'prediction': page.inner_text('#pred-text'),
            'result_id': page.get_attribute('#result', 'data-sample-id'),
        }
        report['checks']['url_opens_personal_dataset'] = (
            report['opened']['split_value'] == 'personal'
            and '83' in report['opened']['count']
            and report['opened']['result_id'].startswith('personal:'))
        report['checks']['personal_clip_transcribed_on_open'] = bool(
            report['opened']['prediction'].strip())
        report['checks']['dataset_note_explains_provenance'] = (
            '전사' in report['opened']['note']
            and '검수' in report['opened']['note'])
        report['checks']['source_line_shows_origin_and_review_state'] = (
            '출처' in report['opened']['source']
            and '검수' in report['opened']['source'])
        report['checks']['age_shown_as_unknown_not_invented'] = \
            '나이 미상' in report['opened']['meta']
        report['checks']['child_filter_is_offered'] = not page.is_hidden('#child-filter')
        report['checks']['historical_score_scoped_to_the_corpus'] = (
            'AIHub' in page.inner_text('#model-facts')
            and '정확도가 아니'  in page.inner_text('#model-facts'))
        page.screenshot(path=str(SHOTS / '07-personal-clips.png'), full_page=True)
        report['screenshots'].append('07-personal-clips.png')

        # 2. each child in turn: filter, select, play, recognise
        report['children'] = {}
        for child, expected in (('child1', '48'), ('child2', '35')):
            page.select_option('#child', child)
            page.wait_for_function(
                "n => document.getElementById('result-count').textContent.startsWith(n)",
                arg=expected, timeout=60_000)
            row = page.locator('#sample-list button').nth(1)
            row_id = row.get_attribute('data-id')
            row.click()
            page.wait_for_function(
                "id => { const r = document.getElementById('result');"
                "        return !r.hidden && r.dataset.sampleId === id; }",
                arg=row_id, timeout=300_000)
            playback = page.evaluate(PLAY, None)
            report['children'][child] = {
                'count': page.inner_text('#result-count'),
                'selected': row_id,
                'meta': page.inner_text('#sample-meta'),
                'transcript': page.inner_text('#ref-transcript'),
                'prediction': page.inner_text('#pred-text'),
                'metrics': page.inner_text('#metrics'),
                'playback': playback,
            }
            entry = report['children'][child]
            report['checks'][f'{child}_filtered_and_transcribed'] = (
                entry['count'].startswith(expected)
                and row_id.startswith(f'personal:{child}')
                and bool(entry['prediction'].strip()))
            report['checks'][f'{child}_audio_actually_plays'] = played(playback)
            report['checks'][f'{child}_is_labelled_in_the_panel'] = \
                ('아동 1' if child == 'child1' else '아동 2') in entry['meta']
        page.select_option('#child', '')
        page.wait_for_timeout(600)

        # 3. the whole recordings: no reference, chunked text, real seeking
        #    (a dataset of this same source, so the picker still does it)
        page.select_option('#split', 'original')
        page.wait_for_function(
            "() => { const r = document.getElementById('original-result');"
            "        return !r.hidden && r.dataset.sampleId; }",
            timeout=600_000)
        segments = page.locator('.segment')
        report['original'] = {
            'count': page.inner_text('#result-count'),
            'title': page.inner_text('#sample-title'),
            'meta': page.inner_text('#sample-meta'),
            'coverage': page.inner_text('#original-coverage'),
            'note': page.inner_text('#original-note'),
            'segments': segments.count(),
            'first_segment': segments.nth(0).inner_text(),
            'last_segment': segments.nth(segments.count() - 1).inner_text(),
            'batch_button_hidden': page.is_hidden('#batch'),
            'result_panel_hidden': page.evaluate(
                "() => document.getElementById('result').hidden"),
        }
        report['checks']['original_dataset_browsable'] = \
            report['original']['count'].startswith('4')
        report['checks']['whole_recording_transcribed_in_chunks'] = \
            report['original']['segments'] >= 5
        report['checks']['coverage_says_every_sample_included'] = \
            '전체 포함' in report['original']['coverage']
        report['checks']['no_error_rate_without_a_reference'] = (
            '비교 불가' in report['original']['coverage']
            and '%' not in report['original']['coverage'])
        report['checks']['whole_recording_not_offered_to_batch'] = \
            report['original']['batch_button_hidden']

        # the timestamps are the recording's own, and the player is whole
        times = page.evaluate("""() => Array.from(
            document.querySelectorAll('.segment .at')).map(n => n.textContent)""")
        report['original']['times'] = times
        report['checks']['segments_are_timestamped'] = (
            len(times) == report['original']['segments']
            and times[0].startswith('00:00'))

        duration = page.evaluate("() => document.getElementById('player').duration")
        report['original']['player_duration'] = duration
        report['checks']['player_holds_the_whole_recording'] = duration > 50

        # clicking a later chunk seeks the whole-file player and it plays on
        target = segments.nth(min(3, report['original']['segments'] - 1))
        seek_to = float(target.locator('button').get_attribute('data-at'))
        target.locator('button').click()
        page.wait_for_timeout(1500)
        moved = page.evaluate("""() => {
            const a = document.getElementById('player');
            const out = {at: a.currentTime, paused: a.paused, error: a.error && a.error.code};
            a.pause();
            return out;
        }""")
        report['original']['seek'] = {'asked': seek_to, **moved}
        report['checks']['segment_button_seeks_the_original_player'] = (
            moved['error'] is None and moved['at'] >= seek_to
            and moved['at'] < seek_to + 8)
        page.screenshot(path=str(SHOTS / '08-original-segments.png'), full_page=True)
        report['screenshots'].append('08-original-segments.png')

        # 3b. a whole-recording run is the longest request the page makes,
        #     so the latest-selection rule matters most here. A's request is
        #     held genuinely open while B is chosen, then released: A's
        #     answer must never be painted, and B must still get its own.
        hold = Hold(page, '**/api/transcribe-original')
        rows = page.locator('#sample-list button')
        first_id = rows.nth(0).get_attribute('data-id')
        second_id = rows.nth(2).get_attribute('data-id')
        rows.nth(0).click()
        pending = hold.wait_until_pending(1)
        rows.nth(2).click()                       # B chosen, A still open
        page.wait_for_timeout(400)
        midway = page.get_attribute('#original-result', 'data-sample-id') or ''
        hold.release()
        page.wait_for_timeout(800)
        after_release = page.get_attribute('#original-result', 'data-sample-id') or ''
        hold.close()
        page.wait_for_function(
            "id => { const r = document.getElementById('original-result');"
            "        return !r.hidden && r.dataset.sampleId === id; }",
            arg=second_id, timeout=600_000)
        report['original_race'] = {
            'first': first_id, 'second': second_id,
            'a_pending_when_b_chosen': pending, 'midway': midway,
            'after_release': after_release,
            'final': page.get_attribute('#original-result', 'data-sample-id'),
        }
        report['checks']['late_whole_recording_reply_never_painted'] = (
            pending and midway != first_id and after_release != first_id)
        report['checks']['recording_chosen_during_a_run_gets_its_own_result'] = \
            report['original_race']['final'] == second_id

        # 4. history must carry the unscored run without a made-up rate
        page.locator('details.details', has_text='지금까지 인식한 기록').click()
        page.wait_for_timeout(500)
        report['history'] = page.inner_text('#history')[:400]
        report['checks']['history_shows_not_comparable_not_zero'] = \
            '비교 불가' in report['history']

        # 5. back to the corpus: its own score and its own filters return.
        #    The corpus is a different page now, so this is a real
        #    navigation through the tab rather than a dropdown change.
        page.click('#tab-aihub')
        page.wait_for_load_state('networkidle')
        page.wait_for_function(
            "() => { const r = document.getElementById('result');"
            "        return !r.hidden && r.dataset.sampleId.startsWith('test:'); }",
            timeout=300_000)
        report['back_to_test'] = {
            'count': page.inner_text('#result-count'),
            'meta': page.inner_text('#sample-meta'),
            'metrics': page.inner_text('#metrics'),
            'facts': page.inner_text('#model-facts'),
            'child_filter_hidden': page.is_hidden('#child-filter'),
            'dataset_note_hidden': page.is_hidden('#dataset-note'),
        }
        report['checks']['corpus_split_unchanged'] = (
            '2,999' in report['back_to_test']['count']
            and '자모 오류율' in report['back_to_test']['metrics'])
        report['checks']['corpus_score_reads_as_its_own_again'] = (
            '전체 테스트 성적' in report['back_to_test']['facts'])
        report['checks']['child_filter_only_for_personal'] = \
            report['back_to_test']['child_filter_hidden']

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
