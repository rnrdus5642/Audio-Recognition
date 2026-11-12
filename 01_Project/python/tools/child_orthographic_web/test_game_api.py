"""Word-game decisions, stale frames and persistence; inherited live regression tests."""
import io
import unittest
import unicodedata

import numpy as np
import soundfile as sf

import test_live_api


class WordGameTests(test_live_api.LiveApiTests):
    def setUp(self):
        super().setUp()
        self.voice = self.wav(0.03)
        self.silence = self.wav(0.0)

    @staticmethod
    def wav(level):
        buffer = io.BytesIO()
        sf.write(buffer, np.full(40000, level, dtype=np.float32), 16000, format='WAV', subtype='PCM_16')
        return buffer.getvalue()

    def start(self, **kwargs):
        payload = kwargs if 'targets' in kwargs else {'reference': '책', **kwargs}
        response = self.client.post('/api/game/start', json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['session_id']

    def frame(self, session, index, elapsed=None, audio=None):
        response = self.post(f'/api/game/{session}/frame',
                             data={'frame_index': str(index), 'elapsed_seconds': str(elapsed or index * 0.5)},
                             audio=self.voice if audio is None else audio)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def finish(self, session, reason):
        response = self.client.post(f'/api/game/{session}/finish', json={'reason': reason})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_retry_does_not_count_twice_and_confirmation_is_frozen(self):
        session = self.start()
        first = self.frame(session, 1)
        self.assertEqual(first['streak'], 1)
        self.assertEqual(self.frame(session, 1), first)
        self.assertEqual(len(self.engine.calls), 1)
        accepted = self.frame(session, 2)
        self.assertEqual(accepted['state'], 'accepted')
        self.assertEqual(self.frame(session, 2), accepted)
        self.engine.text = '자동차'
        self.assertEqual(self.frame(session, 3), accepted)
        self.assertEqual(self.finish(session, 'stopped'), accepted)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(self.client.get('/api/history').json()['total'], 1)
        record = accepted['record']
        self.assertEqual(record['game']['outcome'], 'accepted')
        self.assertEqual(record['prediction']['text'], '책')
        self.assertEqual(self.client.get(record['audio_url']).content, self.voice)

    def test_wrong_result_and_missing_interval_break_streak(self):
        session = self.start()
        self.frame(session, 1)
        self.engine.text = '자동차'
        self.assertEqual(self.frame(session, 2)['streak'], 0)
        self.engine.text = '책'
        self.assertEqual(self.frame(session, 3)['streak'], 1)
        self.assertEqual(self.frame(session, 4, elapsed=3.0)['streak'], 1)
        self.assertEqual(self.frame(session, 5, elapsed=3.5)['state'], 'accepted')

    def test_silence_cannot_pass_even_if_asr_hallucinates_target(self):
        session = self.start()
        self.frame(session, 1, audio=self.silence)
        self.assertEqual(self.frame(session, 2, audio=self.silence)['streak'], 0)
        finished = self.finish(session, 'timeout')
        self.assertEqual(finished['state'], 'retry')
        self.assertEqual(finished['record']['game']['outcome'], 'retry')

    def test_near_pronunciation_allowed_but_larger_word_is_not_a_substring_hit(self):
        session = self.start()
        self.engine.text = '짹'
        self.frame(session, 1)
        self.assertEqual(self.frame(session, 2)['state'], 'accepted')
        session = self.start()
        self.engine.text = '책상'
        self.frame(session, 1)
        self.assertEqual(self.frame(session, 2)['streak'], 0)
        self.assertEqual(self.finish(session, 'timeout')['state'], 'retry')
        session = self.start(threshold=1.0)
        self.engine.text = '짹'
        self.frame(session, 1)
        self.assertEqual(self.frame(session, 2)['streak'], 0)

    def test_stop_freezes_decision_and_new_session_does_not_inherit_streak(self):
        session = self.start()
        self.frame(session, 1)
        stopped = self.finish(session, 'stopped')
        self.assertEqual(stopped['state'], 'stopped')
        self.assertEqual(self.frame(session, 2), stopped)
        other = self.start()
        self.assertNotEqual(session, other)
        self.assertEqual(self.frame(other, 1)['streak'], 1)

    def test_selects_nonfirst_candidate_without_using_it_as_ground_truth(self):
        session = self.start(targets=['자동차', '출발', '책'])
        self.engine.text = '출발'
        first = self.frame(session, 1)
        self.assertEqual(first['candidate'], '출발')
        self.assertEqual(first['streak'], 1)
        accepted = self.frame(session, 2)
        self.assertEqual(accepted['matched_target'], '출발')
        record = accepted['record']
        self.assertEqual(record['game']['matched_target'], '출발')
        self.assertEqual(record['reference'], '')
        self.assertIsNone(record['score'])
        self.assertEqual(record['game']['settings']['targets'], ['자동차', '출발', '책'])
        self.assertEqual(record['game']['candidate_scores'][0]['target'], '출발')
        self.assertEqual(self.client.get('/api/history').json()['scored_count'], 0)

    def test_alternating_candidates_do_not_accumulate_confirmation(self):
        session = self.start(targets=['책', '자동차'])
        for index, word in enumerate(['책', '자동차', '책', '자동차'], 1):
            self.engine.text = word
            answer = self.frame(session, index)
            self.assertEqual(answer['state'], 'listening')
            self.assertEqual(answer['streak'], 1)
            self.assertEqual(answer['candidate'], word)
            self.assertEqual(self.frame(session, index), answer)
        self.assertEqual(len(self.engine.calls), 4)
        accepted = self.frame(session, 5)
        self.assertEqual(accepted['matched_target'], '자동차')
        self.assertEqual(accepted['record']['game']['streak'], 2)
        self.engine.text = '책'
        self.assertEqual(self.frame(session, 6), accepted)

    def test_homophone_ties_never_choose_by_candidate_order(self):
        self.engine.text = '낫'
        for targets in [['낫', '낮', '책'], ['낮', '낫', '책']]:
            session = self.start(targets=targets)
            for index in [1, 2, 3]:
                answer = self.frame(session, index)
                self.assertTrue(answer['ambiguous'])
                self.assertEqual(set(answer['tied_targets']), {'낫', '낮'})
                self.assertIsNone(answer['candidate'])
                self.assertEqual(answer['streak'], 0)
            result = self.finish(session, 'timeout')
            self.assertEqual(result['state'], 'retry')
            self.assertIsNone(result['record']['game']['matched_target'])
            self.assertTrue(result['record']['game']['ambiguous'])

    def test_duplicates_normalize_and_single_target_keeps_reference_score(self):
        session = self.start(targets=[' 책 ', unicodedata.normalize('NFD', '책'), '책'])
        self.frame(session, 1)
        result = self.frame(session, 2)
        self.assertEqual(result['settings']['targets'], ['책'])
        self.assertEqual(result['matched_target'], '책')
        self.assertTrue(result['record']['score']['exact'])

    def test_candidate_lists_and_age_do_not_change_recognizer_input(self):
        for targets, age in [(['책', '출발'], '3'), (['자동차', '사과'], 'adult')]:
            session = self.start(targets=targets, age=age)
            answer = self.frame(session, 1)
            self.assertEqual(answer['prediction']['text'], '책')
        np.testing.assert_array_equal(*self.engine.calls)

    def test_multicandidate_silence_and_invalid_lists(self):
        session = self.start(targets=['책', '출발'])
        for index in [1, 2]:
            answer = self.frame(session, index, audio=self.silence)
            self.assertEqual(answer['streak'], 0)
            self.assertTrue(all(row['similarity'] == 0 for row in answer['candidate_scores']))
            self.assertIsNone(answer['matched_target'])
        for payload in [{}, {'targets': ['책', '두 단어']}, {'reference': '책', 'targets': ['책']}]:
            self.assertEqual(self.client.post('/api/game/start', json=payload).status_code, 400)
        for payload in [{'targets': []}, {'targets': ['책'] * 21}, {'targets': '책,사과'}]:
            self.assertEqual(self.client.post('/api/game/start', json=payload).status_code, 422)

    def test_bad_target_settings_and_frame_data_rejected(self):
        self.assertEqual(self.client.post('/api/game/start', json={'reference': '두 단어'}).status_code, 400)
        self.assertEqual(self.client.post('/api/game/start', json={'reference': '책', 'consecutive': 1}).status_code, 422)
        session = self.start()
        response = self.post(f'/api/game/{session}/frame', data={'frame_index': '1', 'elapsed_seconds': 'nan'}, audio=self.voice)
        self.assertEqual(response.status_code, 400)
        response = self.post(f'/api/game/{session}/frame', data={'frame_index': '1', 'elapsed_seconds': '.5'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.engine.calls), 0)
        self.assertEqual(list(self.records.iterdir()), [])

    def test_preceding_speech_does_not_penalize_complete_answer(self):
        for text in ['브라이어 들어야지', '아ᄀ 들어야지', '안녕하세요 들 거에요']:
            with self.subTest(text=text):
                session = self.start(targets=['들어야지', '들거에요'])
                self.engine.text = text
                self.frame(session, 1)
                answer = self.frame(session, 2)
                self.assertEqual(answer['state'], 'accepted')
                self.assertEqual(answer['prediction']['text'], text)
                self.assertEqual(answer['record']['prediction']['text'], text)
                self.assertEqual(answer['matched_text'], '들 거에요' if '들 거' in text else '들어야지')
                self.assertEqual(answer['record']['game']['matched_text'], answer['matched_text'])

    def test_context_does_not_cut_inside_word_or_delete_unfinished_jamo(self):
        for text in ['책상', '안녕 책상', '책ᄉ', '안녕 책ᄉ', '책 아니야']:
            with self.subTest(text=text):
                session = self.start()
                self.engine.text = text
                self.frame(session, 1)
                self.assertEqual(self.frame(session, 2)['streak'], 0)

    def test_context_homophones_remain_ambiguous(self):
        session = self.start(targets=['낫', '낮'])
        self.engine.text = '안녕하세요 낫'
        self.frame(session, 1)
        answer = self.frame(session, 2)
        self.assertTrue(answer['ambiguous'])
        self.assertEqual(answer['streak'], 0)

    def test_sentence_middle_and_listed_noun_affixes_keep_raw_transcript(self):
        for text in ['이건 사과인거 같아요', '이건 사과인 거 같아요', '이건 사과 인 거 같아요',
                     '이건 사과인것같아요', '이건 사과라고 생각해요', '사과를 먹었어요',
                     '저기 사과가 있어요', '저기 사과 있어요', '이건 사과예요',
                     '이건 "사과" 같아요']:
            with self.subTest(text=text):
                session = self.start(targets=['배', '사과', '자동차'])
                self.engine.text = text
                self.assertEqual(self.frame(session, 1)['state'], 'listening')
                answer = self.frame(session, 2)
                self.assertEqual(answer['matched_target'], '사과')
                self.assertEqual(answer['prediction']['text'], text)
                self.assertEqual(answer['record']['prediction']['text'], text)
                self.assertEqual(answer['comparison_text'], '사과')
                self.assertEqual(answer['settings']['policy'], 'orthographic-word-context-v5')
                self.assertEqual(answer['record']['game']['comparison_text'], '사과')

    def test_context_affixes_do_not_turn_compounds_or_actions_into_answers(self):
        for target, text in [('물', '여기 물고기 있어요'), ('책', '저기 책상 있어요'),
                             ('들어', '아주 힘들어 보여요'), ('파', '나는 파랑 좋아해요'),
                             ('사과', '여기 사과주스 있어요'), ('알', '그건 알고 있어요'),
                             ('사과', '사과ᄀ 같아요'), ('실로폰', '그 실로폰채를 주세요')]:
            with self.subTest(target=target, text=text):
                session = self.start(reference=target)
                self.engine.text = text
                self.frame(session, 1)
                self.assertEqual(self.frame(session, 2)['state'], 'listening')

    def test_middle_short_words_require_identical_sound_but_tail_keeps_tolerance(self):
        for text, target, expected in [('간식도 다 먹어요', '따', 'listening'),
                                       ('빨래가 다 되었나봐요', '따', 'listening'),
                                       ('여기 물 주세요', '물', 'accepted'),
                                       ('여기 물을 주세요', '물', 'accepted'),
                                       ('짹', '책', 'accepted')]:
            with self.subTest(text=text, target=target):
                session = self.start(reference=target)
                self.engine.text = text
                self.frame(session, 1)
                self.assertEqual(self.frame(session, 2)['state'], expected)

    def test_middle_same_action_tolerance_and_known_action_contrast(self):
        for text, target, expected in [('이제 흔들어 해 봐', '흔들어요', 'accepted'),
                                       ('지금 쳐요 해 보자', '줘요', 'listening')]:
            with self.subTest(text=text, target=target):
                session = self.start(reference=target)
                self.engine.text = text
                self.frame(session, 1)
                answer = self.frame(session, 2)
                self.assertEqual(answer['state'], expected)
                if expected == 'listening':
                    self.assertEqual(answer['decision_reason'], 'different_action')

    def test_longer_listed_form_does_not_tie_with_its_prefix(self):
        for targets, text, expected in [(['꺼내', '꺼내요'], '자 꺼내요 해 보자', '꺼내요'),
                                        (['사과', '사과나무'], '저기 사과나무를 봐요', '사과나무'),
                                        (['들거야', '들거에요'], '들거에요라고 말했어요', '들거에요'),
                                        (['들거야', '들거에요'], '들거예요라고 말했어요', '들거에요')]:
            with self.subTest(text=text):
                session = self.start(targets=targets)
                self.engine.text = text
                self.frame(session, 1)
                answer = self.frame(session, 2)
                self.assertFalse(answer['ambiguous'])
                self.assertEqual(answer['matched_target'], expected)

    def test_middle_homophones_and_explicit_negation_remain_unconfirmed(self):
        session = self.start(targets=['낫', '낮'])
        self.engine.text = '저기 낫 있어요'
        self.frame(session, 1)
        self.assertTrue(self.frame(session, 2)['ambiguous'])
        for text in ['이건 사과 아니야', '사과가 아니에요', '사과 말고 배 주세요', '사과라고 아니야']:
            with self.subTest(text=text):
                session = self.start(reference='사과')
                self.engine.text = text
                self.frame(session, 1)
                self.assertEqual(self.frame(session, 2)['state'], 'listening')

    def test_two_separate_exact_answers_choose_latest_independent_of_list_order(self):
        for targets in [['먹이', '주자'], ['주자', '먹이']]:
            session = self.start(targets=targets)
            self.engine.text = '이제 먹이를 주자'
            self.frame(session, 1)
            answer = self.frame(session, 2)
            self.assertEqual(answer['matched_target'], '주자')
            self.assertFalse(answer['ambiguous'])
            self.assertEqual(answer['selection_reason'], 'latest_distinct_exact_word')
            self.assertEqual(answer['record']['game']['selection_reason'], 'latest_distinct_exact_word')

    def test_unlimited_session_keeps_listening_and_bounds_history(self):
        import word_game
        session = self.start()
        self.engine.text = '자동차'
        for index in range(1, word_game.TRACE_LIMIT + 4):
            answer = self.frame(session, index)
            self.assertEqual(answer['state'], 'listening')
            self.engine.calls.clear()
        self.assertIsNone(answer['settings']['timeout_seconds'])
        self.assertGreater(answer['elapsed_seconds'], 120)
        state = self.app.state.games.get(session)
        self.assertEqual(len(state['trace']), word_game.TRACE_LIMIT)
        self.assertEqual(state['trace_dropped_frames'], 3)
        self.engine.text = '책'
        self.frame(session, index + 1, elapsed=3600)
        answer = self.frame(session, index + 2, elapsed=3600.5)
        self.assertEqual(answer['state'], 'accepted')
        self.assertEqual(answer['record']['game']['trace_dropped_frames'], 5)
        self.assertEqual(len(answer['record']['game']['trace']), word_game.TRACE_LIMIT)

    def test_explicit_legacy_timeout_still_validates_elapsed(self):
        session = self.start(timeout_seconds=5)
        response = self.post(f'/api/game/{session}/frame',
                             data={'frame_index': '1', 'elapsed_seconds': '6'}, audio=self.voice)
        self.assertEqual(response.status_code, 400)

    def test_repeated_complete_answer_keeps_original_transcript(self):
        session = self.start(targets=['들어', '들어요'])
        self.engine.text = '들어'
        self.frame(session, 1)
        self.engine.text = '들어들어'
        answer = self.frame(session, 2)
        self.assertEqual(answer['state'], 'accepted')
        self.assertEqual(answer['matched_target'], '들어')
        self.assertEqual(answer['prediction']['text'], '들어들어')
        self.assertEqual(answer['matched_text'], '들어들어')
        self.assertEqual(answer['comparison_text'], '들어')
        self.assertEqual(answer['normalization'], 'repeated_word')
        self.assertEqual(answer['record']['game']['comparison_text'], '들어')

    def test_repetition_cannot_extract_internal_or_single_syllable_word(self):
        for text, target in [('책상책상', '책'), ('안녕들어들어', '들어'), ('하하', '하')]:
            with self.subTest(text=text):
                session = self.start(reference=target)
                self.engine.text = text
                self.frame(session, 1)
                self.assertEqual(self.frame(session, 2)['state'], 'listening')

    def test_one_uncertain_check_is_not_positive_but_can_bridge_confirmation(self):
        for text in ['', 'ᄎ', '들어ᅯ']:
            with self.subTest(text=text):
                session = self.start()
                self.engine.text = '책'
                self.frame(session, 1)
                self.engine.text = text
                uncertain = self.frame(session, 2)
                self.assertEqual(uncertain['streak'], 1)
                self.assertFalse(uncertain['passed'])
                self.assertTrue(uncertain['confirmation_waiting'])
                self.assertEqual(uncertain['state'], 'listening')
                self.engine.text = '책'
                self.assertEqual(self.frame(session, 3)['state'], 'accepted')

    def test_near_threshold_same_candidate_can_bridge_but_two_gaps_reset(self):
        session = self.start(reference='들어')
        self.engine.text = '들어'
        self.frame(session, 1)
        self.engine.text = '블어'
        answer = self.frame(session, 2)
        self.assertTrue(answer['confirmation_waiting'])
        self.assertEqual(answer['streak'], 1)
        self.assertEqual(self.frame(session, 3)['streak'], 0)
        self.engine.text = '들어'
        self.assertEqual(self.frame(session, 4)['state'], 'listening')
        self.assertEqual(self.frame(session, 5)['state'], 'accepted')

    def test_known_different_action_is_not_accepted_for_shared_ending(self):
        for text, target in [('넣을거에요', '놓을거에요'), ('넣을 거예요', '놓을거에요'),
                             ('쳐요', '줘요'), ('담을거에요', '닦을거에요')]:
            with self.subTest(text=text):
                session = self.start(reference=target)
                self.engine.text = text
                self.frame(session, 1)
                answer = self.frame(session, 2)
                self.assertGreaterEqual(answer['similarity'], .85)
                self.assertFalse(answer['passed'])
                self.assertEqual(answer['state'], 'listening')
                self.assertTrue(answer['blocked_by'])
                self.assertEqual(answer['decision_reason'], 'different_action')

    def test_same_action_tolerance_and_explicit_correct_candidate_remain(self):
        self.engine.text = '흔들어'
        session = self.start(reference='흔들어요')
        self.frame(session, 1)
        self.assertEqual(self.frame(session, 2)['state'], 'accepted')
        self.engine.text = '넣을거에요'
        session = self.start(targets=['놓을거에요', '넣을거에요'])
        self.frame(session, 1)
        self.assertEqual(self.frame(session, 2)['matched_target'], '넣을거에요')

    def test_other_candidate_or_long_gap_clears_pending_confirmation(self):
        session = self.start(targets=['책', '자동차'])
        for index, text in enumerate(['책', '', '자동차', '책'], 1):
            self.engine.text = text
            self.assertEqual(self.frame(session, index)['state'], 'listening')
        self.engine.text = ''
        self.assertEqual(self.frame(session, 5, elapsed=4)['streak'], 0)
        self.engine.text = '책'
        self.assertEqual(self.frame(session, 6, elapsed=4.5)['state'], 'listening')

    def test_higher_confirmation_count_allows_only_one_uncertain_check(self):
        for texts, expected in [(['책', '', '책', '책'], 'accepted'),
                                (['책', '', '책', '', '책'], 'listening')]:
            session = self.start(consecutive=3)
            for index, text in enumerate(texts, 1):
                self.engine.text = text
                answer = self.frame(session, index)
                self.assertEqual(self.frame(session, index), answer)
            self.assertEqual(answer['state'], expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
