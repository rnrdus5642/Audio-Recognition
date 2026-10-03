"""The listening review page, driven in a real browser.

Destructive by nature - it saves answers - so it runs against a server
started on a spare port with `CHILD_REVIEW_DB` pointing at a throwaway
file. The reviewer's real answers are never opened. That server also skips
warming the model: the review page reads stored baseline predictions and
serves audio, neither of which needs the checkpoint on the GPU.

    .venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe \
        python/tools/child_jamo_web/qa_review.py
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
#: Holds a request genuinely open while the test keeps clicking, rather
#: than sleeping inside the handler (which would block the driver and
#: never create the overlap).
from qa_browser import Hold                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / '.venv/child-jamo-web-20260916'
SHOTS = WORK / 'screenshots'
OUT = WORK / 'consonant-review/qa-review.json'
PORT = 8791
BASE = f'http://127.0.0.1:{PORT}/'
LIVE = 'http://127.0.0.1:8767/'

PLAY = """(id) => new Promise(resolve => {
    const a = document.getElementById(id);
    const out = {error: null, duration: 0, start: 0, end: 0};
    a.onerror = () => { out.error = 'media error ' + (a.error && a.error.code); resolve(out); };
    const go = () => {
        out.duration = a.duration; out.start = a.currentTime;
        a.play().then(() => {
            setTimeout(() => { out.end = a.currentTime; a.pause(); resolve(out); }, 1200);
        }).catch(e => { out.error = String(e); resolve(out); });
    };
    if (a.readyState >= 2) go();
    else a.addEventListener('canplay', go, {once: true});
    setTimeout(() => resolve(out), 12000);
})"""


def played(result):
    return (result['error'] is None and result['duration'] > 0
            and result['end'] > result['start'])


def wait_for(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as fh:
                return json.load(fh)
        except Exception:
            time.sleep(1)
    raise SystemExit(f'{url} 이 뜨지 않았습니다')


def start_server(db_path):
    """A throwaway server on a spare port, with its own annotation file."""
    env = {**os.environ, 'CHILD_REVIEW_DB': str(db_path),
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
    return subprocess.Popen(
        [str(ROOT / '.venv-gpu/Scripts/python.exe'),
         str(Path(__file__).resolve().parent / 'server.py'),
         '--port', str(PORT),            # no --warm: the page needs no GPU
         # Its own ownership file, so this throwaway instance cannot
         # overwrite the record the launcher uses for the real server.
         '--pid-file', str(db_path.parent / 'qa-server.pid.json')],
        cwd=str(ROOT), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    SHOTS.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='child-review-qa-'))
    db = temp / 'reviews.sqlite3'
    report = {'base': BASE, 'isolated_db': str(db), 'checks': {},
              'console_errors': [], 'page_errors': [], 'failed_requests': [],
              'screenshots': [], 'at': time.strftime('%Y-%m-%d %H:%M:%S')}
    real_db = ROOT / '.venv/child-jamo-web-20260916/consonant-review/reviews.sqlite3'

    def live_snapshot():
        """The live answers themselves, read-only.

        SQLite in WAL mode can hold committed rows outside the main file,
        so the file's mtime proves nothing. The rows are what matter.
        """
        if not real_db.exists():
            return {'exists': False, 'rows': 0, 'ids': []}
        import sqlite3
        uri = f'file:{real_db.as_posix()}?mode=ro'
        with sqlite3.connect(uri, uri=True, timeout=10) as db_ro:
            rows = db_ro.execute(
                'SELECT revision, record_id, at FROM revision '
                'ORDER BY revision').fetchall()
        return {'exists': True, 'rows': len(rows), 'ids': [tuple(r) for r in rows]}

    before = live_snapshot()

    server = start_server(db)
    try:
        wait_for(BASE + 'api/review/state')
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
            page = browser.new_page(viewport={'width': 1440, 'height': 980})
            page.on('console', lambda m: m.type == 'error'
                    and report['console_errors'].append(m.text))
            page.on('pageerror', lambda e: report['page_errors'].append(str(e)))
            page.on('requestfailed', lambda r: report['failed_requests'].append(
                f'{r.method} {r.url} {r.failure}'))

            # 1. the page opens on the first recording, with nothing preselected
            page.goto(BASE + 'review', wait_until='networkidle')
            page.wait_for_selector('#choice-list input', timeout=60_000)
            report['first'] = {
                'title': page.inner_text('#item-title'),
                'meta': page.inner_text('#item-meta'),
                'queue': page.locator('#queue button').count(),
                'choices': page.locator('#choice-list input').count(),
                'checked': page.locator('#choice-list input:checked').count(),
                'progress': page.inner_text('#progress-text'),
                'tab_current': page.get_attribute(
                    'a[href="/review"]', 'aria-current'),
            }
            first = report['first']
            report['checks']['review_page_opens_with_a_shortlist'] = (
                15 <= first['queue'] <= 20 and first['choices'] >= 3)
            report['checks']['no_answer_is_preselected'] = first['checked'] == 0
            report['checks']['review_tab_is_marked_current'] = \
                first['tab_current'] == 'page'
            report['checks']['progress_is_shown'] = '우선 확인' in first['progress']

            # 2. the model-input clip actually plays
            report['playback'] = page.evaluate(PLAY, 'player')
            report['checks']['clip_audio_actually_plays'] = played(report['playback'])
            # 3. and so does the original-context clip, labelled as such
            report['context_visible'] = not page.is_hidden('#context-block')
            if report['context_visible']:
                report['context_playback'] = page.evaluate(PLAY, 'context-player')
                report['checks']['context_audio_plays'] = \
                    played(report['context_playback'])
                report['checks']['context_is_labelled_as_context'] = '참고용' in \
                    page.inner_text('#context-block')

            # 4. the reference panel starts collapsed and opens on request.
            #    Named for what it is: the panel is collapsed, NOT that the
            #    reviewer is listening blind - the expected and predicted
            #    text are on the answer buttons either way, and the page
            #    says so rather than claiming otherwise.
            report['reveal_open_initially'] = page.evaluate(
                "() => document.getElementById('reveal').open")
            report['checks']['reference_panel_starts_collapsed'] = \
                report['reveal_open_initially'] is False
            report['choices_text'] = page.inner_text('#choice-list')
            # text_content, not inner_text: the note lives inside the
            # collapsed panel, so inner_text would not see it at all.
            report['checks']['page_does_not_claim_blind_listening'] = (
                '가려 두고 듣는 방식이 아니라'
                in page.locator('#reveal').text_content())
            page.locator('#reveal summary').click()
            page.wait_for_timeout(200)
            report['revealed'] = page.inner_text('#reveal-expected')
            report['checks']['reference_panel_can_be_opened'] = bool(
                report['revealed'].strip())
            page.screenshot(path=str(SHOTS / '14-review-page.png'), full_page=True)
            report['screenshots'].append('14-review-page.png')

            # 4b. conditional controls must actually disappear, not merely
            #     carry the attribute: a class with its own `display` beats
            #     the browser's default for `hidden`.
            visibility = page.evaluate("""() => {
                const shown = (id) => {
                    const el = document.getElementById(id);
                    return getComputedStyle(el).display !== 'none';
                };
                return {heard_before: shown('heard-row'),
                        pager_priority: shown('flagged-pager')};
            }""")
            page.locator('#choice-list input[value="other"]').check()
            page.wait_for_timeout(150)
            visibility['heard_after'] = page.evaluate(
                "() => getComputedStyle(document.getElementById('heard-row'))"
                "        .display !== 'none'")
            report['visibility'] = visibility
            report['checks']['heard_box_appears_only_for_a_custom_answer'] = (
                visibility['heard_before'] is False
                and visibility['heard_after'] is True)
            report['checks']['pager_is_really_hidden_in_the_priority_list'] = \
                visibility['pager_priority'] is False

            # 5. saving without choosing is refused, not silently accepted
            page.reload(wait_until='networkidle')
            page.wait_for_selector('#choice-list input', timeout=60_000)
            page.click('#save')
            page.wait_for_timeout(400)
            report['no_choice_note'] = page.inner_text('#answer-note')
            report['checks']['saving_without_an_answer_is_refused'] = \
                '골라' in report['no_choice_note']

            # 6. 'heard differently' needs the text, then saves and advances
            first_id = page.locator('#queue button').first.get_attribute('data-id')
            page.locator('#choice-list input[value="other"]').check()
            page.click('#save')
            page.wait_for_timeout(400)
            report['checks']['free_text_is_required_when_chosen'] = \
                '적어' in page.inner_text('#answer-note')
            page.fill('#heard-text', 'ㅅㅐㅂ 비슷하게 들림')
            page.click('#save')
            page.wait_for_function(
                "() => document.getElementById('answer-note').textContent"
                "        .includes('저장')", timeout=30_000)
            page.wait_for_timeout(800)
            report['after_save'] = {
                'note': page.inner_text('#answer-note'),
                'moved_to': page.locator(
                    '#queue button[aria-current="true"]').first.get_attribute('data-id'),
                'progress': page.inner_text('#progress-text'),
            }
            report['checks']['save_and_next_advances'] = (
                report['after_save']['moved_to'] != first_id)
            report['checks']['progress_updates_after_saving'] = \
                '1 / 18' in report['after_save']['progress'] or \
                '1 /' in report['after_save']['progress']

            # 7. the saved answer is exactly what was typed, and comes back
            saved = json.loads(urllib.request.urlopen(
                BASE + 'api/review/item/' + first_id).read())
            report['saved_answer'] = saved['annotation']
            report['checks']['free_text_saved_verbatim'] = \
                saved['annotation']['heard_text'] == 'ㅅㅐㅂ 비슷하게 들림'
            page.locator(f'#queue button[data-id="{first_id}"]').click()
            page.wait_for_timeout(700)
            report['revisit'] = {
                'heard': page.input_value('#heard-text'),
                'checked': page.locator('#choice-list input:checked').count(),
            }
            report['checks']['revisiting_shows_the_saved_answer'] = (
                report['revisit']['heard'] == 'ㅅㅐㅂ 비슷하게 들림'
                and report['revisit']['checked'] == 1)

            # 8. editing keeps a history
            page.locator('#choice-list input[value="expected"]').check()
            page.click('#save-stay')
            page.wait_for_function(
                "() => document.getElementById('answer-note').textContent"
                "        .includes('저장했습니다')", timeout=30_000)
            history = json.loads(urllib.request.urlopen(
                BASE + 'api/review/history/' + first_id).read())['items']
            report['history_revisions'] = len(history)
            report['checks']['editing_keeps_a_history'] = len(history) >= 2

            # 9. 'cannot tell' and 'unusable'
            page.locator('#queue button').nth(2).click()
            page.wait_for_timeout(700)
            unclear_id = page.locator(
                '#queue button[aria-current="true"]').first.get_attribute('data-id')
            page.locator('#choice-list input[value="unclear"]').check()
            page.click('#save-stay')
            page.wait_for_function(
                "() => document.getElementById('answer-note').textContent"
                "        .includes('저장했습니다')", timeout=30_000)
            unclear = json.loads(urllib.request.urlopen(
                BASE + 'api/review/item/' + unclear_id).read())['annotation']
            report['checks']['unclear_is_saved_as_its_own_status'] = \
                unclear['status'] == 'unclear' and unclear['choice'] is None

            page.locator('#queue button').nth(3).click()
            page.wait_for_timeout(700)
            unusable_id = page.locator(
                '#queue button[aria-current="true"]').first.get_attribute('data-id')
            page.click('#unusable')
            page.wait_for_timeout(500)
            report['checks']['unusable_needs_a_reason'] = \
                '이유' in page.inner_text('#answer-note')
            page.locator('details').filter(
                has_text='이 녹음 자체에 문제가').locator('summary').click()
            page.wait_for_timeout(200)
            page.locator('#flag-list input[value="boundary"]').check()
            page.click('#unusable')
            page.wait_for_function(
                "() => document.getElementById('answer-note').textContent"
                "        .includes('저장했습니다')", timeout=30_000)
            unusable = json.loads(urllib.request.urlopen(
                BASE + 'api/review/item/' + unusable_id).read())['annotation']
            report['checks']['unusable_saves_with_a_flag'] = (
                unusable['status'] == 'unusable'
                and unusable['flags'] == ['boundary'])

            # 9b. a save held open must not lose a newer answer.
            #
            # The save request is parked, the user moves to another
            # recording and types there, then the first reply is released.
            # The answer on screen has to survive, and the reply must not
            # be applied to the recording now showing.
            page.locator('#queue button').nth(4).click()
            page.wait_for_timeout(700)
            slow_id = page.locator(
                '#queue button[aria-current="true"]').first.get_attribute('data-id')
            hold = Hold(page, '**/api/review/annotation')
            page.locator('#choice-list input[value="expected"]').check()
            page.click('#save-stay')
            pending = hold.wait_until_pending(1)
            # While it is in flight the form is inert, so nothing else can
            # be answered underneath it.
            frozen = page.evaluate("""() => ({
                save: document.getElementById('save').disabled,
                stay: document.getElementById('save-stay').disabled,
                unusable: document.getElementById('unusable').disabled,
                next: document.getElementById('next-item').disabled,
                scope: document.getElementById('scope').disabled,
                queue: [...document.querySelectorAll('#queue button')]
                        .every(b => b.disabled),
                choices: [...document.querySelectorAll('#choice-list input')]
                        .every(b => b.disabled),
            })""")
            hold.release()
            hold.close()
            page.wait_for_timeout(900)
            report['save_race'] = {
                'pending': pending, 'frozen': frozen, 'held_item': slow_id,
                'note_after': page.inner_text('#answer-note'),
                'still_on': page.locator(
                    '#queue button[aria-current="true"]').first.get_attribute('data-id'),
            }
            report['checks']['form_is_inert_while_saving'] = (
                pending and all(frozen.values()))
            report['checks']['a_held_save_lands_on_its_own_recording'] = \
                report['save_race']['still_on'] == slow_id

            # 9c. a slow item request must not paint under a newer one
            item_hold = Hold(page, '**/api/review/item/**')
            first_row = page.locator('#queue button').nth(6)
            first_row_id = first_row.get_attribute('data-id')
            first_row.click()
            item_pending = item_hold.wait_until_pending(1)
            second_row = page.locator('#queue button').nth(7)
            second_row_id = second_row.get_attribute('data-id')
            second_row.click()
            page.wait_for_timeout(300)
            item_hold.release()
            item_hold.close()
            page.wait_for_timeout(1200)
            report['item_race'] = {
                'pending': item_pending, 'first': first_row_id,
                'second': second_row_id,
                'showing': page.locator(
                    '#queue button[aria-current="true"]').first.get_attribute('data-id'),
            }
            report['checks']['a_superseded_item_never_paints'] = (
                item_pending
                and report['item_race']['showing'] == second_row_id)

            # 9d. a slow *item load* must not swallow an answer.
            #
            # The earlier races hold a save, or race two item loads against
            # each other. Neither touches this one: the form belonging to
            # the recording being left stayed live and editable for as long
            # as the next item took to arrive, and was then rebuilt on top
            # of whatever had been typed into it - no warning, no save.
            #
            # Two halves, because "an answer cannot disappear" needs both:
            # an unsaved answer stops the switch and survives it, and once
            # the switch is under way there is no live form left to type a
            # lost answer into.
            page.locator('#queue button').nth(8).click()
            page.wait_for_timeout(700)
            leaving_id = page.locator(
                '#queue button[aria-current="true"]').first.get_attribute('data-id')
            page.locator('#choice-list input[value="other"]').check()
            page.fill('#heard-text', '바꾸기 전에 적어 둔 답')
            warned = []
            page.once('dialog', lambda d: (warned.append(d.message), d.dismiss()))
            next_row = page.locator('#queue button').nth(9)
            arriving_id = next_row.get_attribute('data-id')
            next_row.click()
            page.wait_for_timeout(600)
            kept = {
                'warned': warned,
                'still_on': page.locator(
                    '#queue button[aria-current="true"]').first.get_attribute('data-id'),
                'heard': page.input_value('#heard-text'),
                'checked': page.locator('#choice-list input:checked').count(),
            }

            # Now allow the switch, but hold the item request open and try
            # to answer the old recording while it is in flight.
            get_hold = Hold(page, '**/api/review/item/**')
            page.once('dialog', lambda d: d.accept())
            next_row.click()
            get_pending = get_hold.wait_until_pending(1)
            during = page.evaluate("""() => ({
                choices: document.querySelectorAll('#choice-list input').length,
                flags: document.querySelectorAll('#flag-list input').length,
                heard: document.getElementById('heard-text').value,
                heard_disabled: document.getElementById('heard-text').disabled,
                note_disabled: document.getElementById('note').disabled,
                save: document.getElementById('save').disabled,
                stay: document.getElementById('save-stay').disabled,
                unusable: document.getElementById('unusable').disabled,
                prev_page: document.getElementById('prev-page').disabled,
                next_page: document.getElementById('next-page').disabled,
            })""")
            get_hold.release()
            get_hold.close()
            page.wait_for_timeout(1200)
            left_behind = json.loads(urllib.request.urlopen(
                BASE + 'api/review/item/' + leaving_id).read())
            report['item_load_race'] = {
                'leaving': leaving_id, 'arriving': arriving_id,
                'kept': kept, 'pending': get_pending, 'during': during,
                'showing': page.locator(
                    '#queue button[aria-current="true"]').first.get_attribute('data-id'),
                'left_behind_annotation': left_behind['annotation'],
            }
            race = report['item_load_race']
            report['checks']['an_unsaved_answer_blocks_an_item_switch'] = (
                bool(kept['warned']) and kept['still_on'] == leaving_id
                and kept['heard'] == '바꾸기 전에 적어 둔 답'
                and kept['checked'] == 1)
            report['checks']['no_answer_can_be_typed_while_an_item_loads'] = (
                get_pending and during['choices'] == 0 and during['flags'] == 0
                and during['heard'] == '' and during['heard_disabled']
                and during['note_disabled'] and during['save']
                and during['stay'] and during['unusable'])
            report['checks']['a_delayed_item_load_loses_no_answer'] = (
                race['showing'] == arriving_id
                and left_behind['annotation'] is None)

            # 10. source filtering, and the optional wider list
            page.select_option('#source', 'valid')
            page.wait_for_timeout(900)
            report['valid_only'] = page.inner_text('#queue-count')
            valid_ids = [b.get_attribute('data-id') for b in
                         page.locator('#queue button').all()]
            report['checks']['source_filter_shows_one_source'] = bool(
                valid_ids) and all(i.startswith('valid:') for i in valid_ids)
            page.select_option('#source', 'personal')
            page.wait_for_timeout(900)
            personal_ids = [b.get_attribute('data-id') for b in
                            page.locator('#queue button').all()]
            report['checks']['personal_filter_shows_personal_only'] = bool(
                personal_ids) and all(i.startswith('personal:')
                                      for i in personal_ids)
            page.select_option('#source', '')
            page.select_option('#scope', 'flagged')
            page.wait_for_timeout(900)
            report['flagged_count'] = page.inner_text('#queue-count')
            report['flagged_hint'] = page.inner_text('#scope-hint')
            report['checks']['wider_list_is_optional'] = \
                '필요는 없습니다' in report['flagged_hint']

            # 10b. the pager belongs in the save guard as well. Its handlers
            #      change the generation, which is precisely what must not
            #      happen underneath an answer that is still in flight. And
            #      releasing the guard has to restore the pager's own state
            #      - page 1 has no previous page - not enable everything.
            pager_state = ("""() => ({
                prev: document.getElementById('prev-page').disabled,
                next: document.getElementById('next-page').disabled})""")
            # The corpus list is the one long enough to page through; the
            # personal one fits on a single page, where "next is disabled"
            # would prove nothing about the guard.
            page.select_option('#source', 'valid')
            page.wait_for_timeout(900)
            report['flagged_pages'] = page.inner_text('#page-label')
            page.locator('#queue button').first.click()
            page.wait_for_timeout(800)
            pager_before = page.evaluate(pager_state)
            pager_hold = Hold(page, '**/api/review/annotation')
            page.locator('#choice-list input[value="expected"]').check()
            page.click('#save-stay')
            pager_pending = pager_hold.wait_until_pending(1)
            pager_during = page.evaluate(pager_state)
            pager_hold.release()
            pager_hold.close()
            page.wait_for_function(
                "() => document.getElementById('answer-note').textContent"
                "        .includes('저장')", timeout=30_000)
            page.wait_for_timeout(500)
            pager_after = page.evaluate(pager_state)
            report['pager_guard'] = {'pending': pager_pending,
                                     'before': pager_before,
                                     'during': pager_during,
                                     'after': pager_after}
            report['checks']['the_pager_is_frozen_while_saving'] = (
                pager_pending and pager_during['prev'] and pager_during['next'])
            report['checks']['the_pager_returns_to_its_own_state'] = (
                pager_before['prev'] is True and pager_before['next'] is False
                and pager_after == pager_before)

            page.select_option('#scope', 'priority')
            page.wait_for_timeout(900)

            # 11. answers survive a refresh
            page.reload(wait_until='networkidle')
            page.wait_for_selector('#choice-list input', timeout=60_000)
            report['after_refresh_progress'] = page.inner_text('#progress-text')
            report['checks']['answers_survive_a_refresh'] = \
                '0 / 18' not in report['after_refresh_progress']
            page.screenshot(path=str(SHOTS / '15-review-answered.png'),
                            full_page=True)
            report['screenshots'].append('15-review-answered.png')

            # 12. mobile
            page.set_viewport_size({'width': 390, 'height': 844})
            page.wait_for_timeout(600)
            report['mobile_overflow'] = page.evaluate(
                '() => document.documentElement.scrollWidth'
                '    - document.documentElement.clientWidth')
            report['checks']['no_mobile_horizontal_overflow'] = \
                report['mobile_overflow'] <= 0
            report['checks']['choices_usable_on_a_phone'] = page.evaluate(
                """() => {
                    const box = document.querySelector('.choice input');
                    const r = box.getBoundingClientRect();
                    return r.width >= 16 && r.height >= 16;
                }""")
            page.screenshot(path=str(SHOTS / '16-review-mobile.png'),
                            full_page=True)
            report['screenshots'].append('16-review-mobile.png')
            page.set_viewport_size({'width': 1440, 'height': 980})

            report['checks']['no_page_exceptions'] = not report['page_errors']
            unexpected = [e for e in report['console_errors']
                          if '400 (Bad Request)' not in e]
            report['unexpected_console_errors'] = unexpected
            report['checks']['no_unexpected_console_errors'] = not unexpected
            browser.close()

        # 12b. answers must outlive the server, not just the page. The
        #      process is stopped and started again against the same file.
        before_restart = json.loads(urllib.request.urlopen(
            BASE + 'api/review/export').read())['counts']
        server.terminate()
        try:
            server.wait(timeout=20)
        except Exception:
            server.kill()
        time.sleep(1.0)
        server = start_server(db)
        wait_for(BASE + 'api/review/state')
        after_restart = json.loads(urllib.request.urlopen(
            BASE + 'api/review/export').read())['counts']
        restored = json.loads(urllib.request.urlopen(
            BASE + 'api/review/item/' + first_id).read())['annotation']
        report['restart'] = {'before': before_restart, 'after': after_restart,
                             'heard_after': restored['heard_text'],
                             'status_after': restored['status']}
        report['checks']['answers_survive_a_server_restart'] = (
            before_restart == after_restart and restored is not None)

        # 13. the export, and the live app still being itself
        export = json.loads(urllib.request.urlopen(BASE + 'api/review/export').read())
        report['export_counts'] = export['counts']
        report['checks']['export_has_accepted_and_set_aside'] = (
            export['counts']['accepted'] >= 1
            and export['counts']['set_aside'] >= 2)
        report['checks']['export_says_it_is_not_ground_truth'] = \
            '전문가' in export['what_this_is']

        live = json.loads(urllib.request.urlopen(LIVE + 'api/status').read())
        report['live_checkpoint'] = live['model']['checkpoint_sha256']
        report['checks']['live_server_uses_the_same_checkpoint'] = (
            live['model']['checkpoint_sha256'] ==
            '5a330f1bea3a17d1d063fdfd71a0cc25394695b0b797de01b293f6f688257aa7')
        for path in ('aihub', 'recordings', 'review'):
            with urllib.request.urlopen(LIVE + path) as fh:
                report['checks'][f'live_{path}_page_ok'] = fh.status == 200
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except Exception:
            server.kill()

    after = live_snapshot()
    report['live_store_before'] = before
    report['live_store_after'] = after
    report['checks']['no_fake_answers_in_the_real_store'] = (
        before['rows'] == after['rows'] and before['ids'] == after['ids'])
    report['passed'] = all(report['checks'].values())
    report['failed_checks'] = [k for k, v in report['checks'].items() if not v]
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding='utf-8')
    for key, value in report['checks'].items():
        print(f'  {"ok  " if value else "FAIL"} {key}')
    print('passed:', report['passed'], '->', OUT)
    for label in ('unexpected_console_errors', 'page_errors'):
        if report.get(label):
            print(label, report[label][:4])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
