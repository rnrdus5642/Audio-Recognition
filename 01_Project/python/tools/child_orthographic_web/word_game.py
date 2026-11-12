"""Target-independent ASR followed by phoneme matching and session confirmation."""
import hashlib
import math
from pathlib import Path
import re
import sys
import threading
import time
import unicodedata
import uuid

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from answer_policy import ActionGuard, context_spans, confirm, spelling_key

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from python.build.g2p.ko import KoreanG2P
from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import Matcher

WINDOW_SECONDS = 2.5
HOP_SECONDS = 0.5
RATE = 16000
POLICY = 'orthographic-word-context-v5'
TRACE_LIMIT = 240  # Recent two minutes at the normal 0.5-second cadence.


class StartWordGame(BaseModel):
    reference: str | None = Field(default=None, min_length=1, max_length=80)
    targets: list[str] | None = Field(default=None, min_length=1, max_length=20)
    age: Literal['unknown', '3', '4', '5', '6', 'adult'] = 'unknown'
    threshold: float = Field(default=0.85, ge=0.7, le=1)
    consecutive: int = Field(default=2, ge=2, le=4)
    timeout_seconds: int | None = Field(default=None, ge=5, le=15)


class FinishWordGame(BaseModel):
    reason: Literal['stopped', 'timeout']


class WordGames:
    def __init__(self, engine, infer, save):
        self.engine, self.infer, self.save = engine, infer, save
        self.g2p = KoreanG2P()
        self.action_guard = ActionGuard()
        # Compare bounded words throughout the sentence and explicit affixes;
        # "이건 책인 거 같아" can match "책", while "책상" cannot be cut.
        self.matcher = Matcher(ConfusionMatrix.from_json(
            PROJECT / 'shared/confusion_matrices/ko_child_v2.json'), mode='exact')
        self.sessions = {}
        self.lock = threading.Lock()

    def start(self, params):
        if not self.engine.ready:
            raise HTTPException(503, '모델 준비가 끝난 뒤 시작해 주세요.')
        if (params.reference is None) == (params.targets is None):
            raise HTTPException(400, '정답 단어 또는 후보 목록 중 하나를 입력해 주세요.')
        targets = list(dict.fromkeys(re.sub(r'[?？]+$', '', unicodedata.normalize('NFC', word.strip())).strip()
                                    for word in (params.targets if params.targets is not None else [params.reference])))
        if any(not re.fullmatch('[가-힣]{1,20}', word) for word in targets):
            raise HTTPException(400, '후보마다 띄어쓰기 없이 한글 1~20글자로 입력해 주세요. 끝의 물음표는 생략해 비교합니다.')
        settings = params.model_dump()
        settings.update(reference=targets[0] if len(targets) == 1 else '', targets=targets,
                        window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS,
                        policy=POLICY, threshold_calibrated=False,
                        ambiguity_policy='homophones_rejected_distinct_exact_words_choose_latest',
                        context_policy='bounded_words_anywhere_with_listed_affixes',
                        short_context_policy='one_syllable_interior_requires_identical_phonemes',
                        affix_policy='listed_nouns_particles_copula_and_quotation',
                        negation_policy='immediate_following_ani_or_malgo',
                        repetition_policy='complete_listed_word_only_min_2_syllables',
                        confirmation_policy='same_candidate_with_one_uncertain_check',
                        confirmation_window_checks=params.consecutive + 1,
                        action_guard='known_different_action',
                        trace_limit=TRACE_LIMIT,
                        similarity='weighted_IPA_from_recognized_spelling')
        session_id = uuid.uuid4().hex
        state = {'id': session_id, 'settings': settings,
                 'candidates': [(word, self.g2p.to_ipa(word)) for word in targets],
                 'lock': threading.Lock(), 'touched': time.monotonic(), 'index': 0,
                 'elapsed': 0.0, 'streak': 0, 'streak_target': None, 'outcome': 'listening', 'trace': [],
                 'trace_dropped_frames': 0,
                 'evidence': [], 'uncertain_checks': 0, 'confirmation_waiting': False,
                 'last': None, 'best': None, 'record': None, 'request_hash': None}
        with self.lock:
            for key in list(self.sessions):
                if time.monotonic() - self.sessions[key]['touched'] > 600:
                    del self.sessions[key]
            if len(self.sessions) >= 128:
                finished = [key for key, value in self.sessions.items() if value['outcome'] != 'listening']
                if finished:
                    del self.sessions[finished[0]]
                else:
                    raise HTTPException(503, '진행 중인 테스트를 마친 뒤 다시 시작해 주세요.')
            self.sessions[session_id] = state
        return self.response(state)

    def get(self, session_id):
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None or time.monotonic() - state['touched'] > 600:
                raise HTTPException(410, '테스트 세션이 만료되었습니다. 다시 시작해 주세요.')
            state['touched'] = time.monotonic()
            return state

    @staticmethod
    def voice_fraction(waveform):
        # Energy gate only; this is not a speech/noise classifier.
        frames = waveform[:len(waveform) // 320 * 320].reshape(-1, 320)
        return float(np.mean(np.sqrt(np.mean(frames * frames, axis=1)) >= 0.006)) if len(frames) else 0.0

    def score_candidates(self, text, candidates, fraction):
        # An unfinished jamo in an earlier word must not invalidate a later
        # complete word. Keep invalid characters inside their token, however;
        # deleting them could turn an unfinished "책ᄉ" into a false "책" hit.
        spans = [(span, self.g2p.to_ipa(span.normalized)) for span in
                 context_spans(text, {word for word, _ in candidates}, self.action_guard)]
        scores = []
        for target, phonemes in candidates:
            best, matched, compared, normalization = 0.0, None, None, None
            best_span = None
            if fraction >= 0.04:
                for span, user in spans:
                    # A listed affix cannot create fuzzy matches to unrelated
                    # roots. Keep genuine homophones tied, as before.
                    if span.owner and span.owner != target and user != phonemes:
                        continue
                    # "다 먹어요" must not create a new middle-of-sentence
                    # match for the one-syllable answer "따" (90% similarity).
                    if not span.at_end and len(target) == 1 and user != phonemes:
                        continue
                    value = self.matcher.score_against(user, phonemes)[1]
                    if value > best:
                        best, matched, compared, normalization = value, span.raw, span.normalized, span.reason
                        best_span = span
            conflict = self.action_guard.conflict(compared or '', target)
            scores.append({'target': target, 'similarity': best, 'matched_text': matched,
                           'comparison_text': compared, 'normalization': normalization, 'blocked_by': conflict,
                           'match_start': best_span.start if best_span else None,
                           'match_end': best_span.end if best_span else None,
                           'exact_spelling': bool(compared and spelling_key(compared) == spelling_key(target))})
        scores.sort(key=lambda item: item['similarity'], reverse=True)
        return scores

    def frame(self, session_id, index, elapsed, waveform, canonical, original_rate):
        state = self.get(session_id)
        timeout = state['settings']['timeout_seconds']
        if (not math.isfinite(elapsed) or elapsed < HOP_SECONDS - 0.02 or
                (timeout is not None and elapsed > timeout + 0.05)):
            raise HTTPException(400, '녹음 시각이 올바르지 않습니다.')
        if abs(len(waveform) / RATE - WINDOW_SECONDS) > 0.02:
            raise HTTPException(400, '판정 음성은 최근 2.5초여야 합니다.')
        digest = hashlib.sha256(canonical + repr(elapsed).encode()).hexdigest()
        with state['lock']:
            if index == state['index']:
                if digest != state['request_hash']:
                    raise HTTPException(409, '이미 처리한 검사와 음성 내용이 다릅니다.')
                return self.response(state)
            if state['outcome'] != 'listening':
                return self.response(state)
            if index != state['index'] + 1 or elapsed - state['elapsed'] < HOP_SECONDS - 0.05:
                raise HTTPException(409, '새로운 음성을 순서대로 보내 주세요.')
            fraction = self.voice_fraction(waveform)
            # The recognizer never receives the target, age or scoring settings.
            prediction = self.infer(waveform)
            text = prediction['text']
            scores = self.score_candidates(text, state['candidates'], fraction)
            similarity = scores[0]['similarity']
            leaders = [item['target'] for item in scores
                       if math.isclose(item['similarity'], similarity, rel_tol=0, abs_tol=1e-9)]
            selection_reason = None
            if len(leaders) > 1 and math.isclose(similarity, 1.0, rel_tol=0, abs_tol=1e-9):
                tied = [item for item in scores if item['target'] in leaders]
                # "먹이를 주자" explicitly contains two permitted answers. This
                # differs from one sound matching two candidates (낫/낮).
                distinct_exact = all(item.get('exact_spelling') and not item['blocked_by'] for item in tied)
                separate = distinct_exact and all(
                    left['match_end'] <= right['match_start'] or right['match_end'] <= left['match_start']
                    for i, left in enumerate(tied) for right in tied[i + 1:])
                if separate:
                    chosen = max(tied, key=lambda item: item['match_end'])
                    scores = [chosen] + [item for item in scores if item is not chosen]
                    leaders = [chosen['target']]
                    selection_reason = 'latest_distinct_exact_word'
            ambiguous = similarity >= state['settings']['threshold'] and len(leaders) > 1
            candidate = leaders[0] if similarity > 0 and len(leaders) == 1 else None
            matched_text = scores[0]['matched_text'] if candidate else None
            blocked_by = scores[0]['blocked_by'] if candidate else None
            passes = bool(candidate and not blocked_by and similarity >= state['settings']['threshold'])
            accepted = confirm(state, candidate, passes, similarity, text, elapsed,
                               blocked=bool(blocked_by), ambiguous=ambiguous)
            reason = ('different_action' if blocked_by else 'ambiguous' if ambiguous else
                      'uncertain_check' if state['confirmation_waiting'] else
                      'confirming' if passes else 'no_voice' if fraction < .04 else 'below_threshold')
            frame = {'prediction': prediction, 'similarity': similarity, 'passed': passes,
                     'candidate': candidate, 'candidate_scores': scores, 'ambiguous': ambiguous,
                     'matched_text': matched_text,
                     'comparison_text': scores[0]['comparison_text'] if candidate else None,
                     'normalization': scores[0]['normalization'] if candidate else None,
                     'blocked_by': blocked_by, 'decision_reason': reason,
                     'selection_reason': selection_reason,
                     'confirmation_waiting': state['confirmation_waiting'],
                     'tied_targets': leaders if ambiguous else [],
                     'voice_fraction': fraction, 'elapsed_seconds': elapsed, 'frame_index': index,
                     'streak': state['streak'], 'canonical': canonical, 'source_sample_rate': original_rate}
            state['last'] = frame
            if state['best'] is None or (fraction >= 0.04 and
                    (state['best']['voice_fraction'] < 0.04 or similarity >= state['best']['similarity'])):
                state['best'] = frame
            state['trace'].append({'frame_index': index, 'elapsed_seconds': elapsed,
                                   'text': text, 'similarity': similarity, 'passed': passes,
                                   'candidate': candidate, 'ambiguous': ambiguous,
                                   'matched_text': matched_text,
                                   'comparison_text': frame['comparison_text'],
                                   'normalization': frame['normalization'],
                                   'blocked_by': blocked_by, 'decision_reason': reason,
                                   'selection_reason': selection_reason,
                                   'confirmation_waiting': state['confirmation_waiting'],
                                   'candidate_scores': scores,
                                   'streak': state['streak'], 'voice_fraction': fraction})
            if len(state['trace']) > TRACE_LIMIT:
                state['trace'].pop(0)
                state['trace_dropped_frames'] += 1
            state['index'], state['elapsed'], state['request_hash'] = index, elapsed, digest
            if accepted:
                state['outcome'] = 'accepted'
            return self.response(state)

    def finish(self, session_id, reason):
        state = self.get(session_id)
        with state['lock']:
            if state['outcome'] == 'listening':
                state['outcome'] = 'retry' if reason == 'timeout' else 'stopped'
            return self.response(state)

    def response(self, state):
        if state['outcome'] != 'listening' and state['record'] is None:
            frame = state['last'] if state['outcome'] == 'accepted' else state['best']
            if frame is not None:
                # Save the exact inference window and prediction that drove the
                # decision, never a different post-confirmation re-transcription.
                state['record'] = self.save(state, frame)
        last = state['last']
        return {'session_id': state['id'], 'state': state['outcome'],
                'settings': state['settings'], 'streak': state['streak'],
                'frames': state['index'], 'elapsed_seconds': state['elapsed'],
                'similarity': last['similarity'] if last else 0.0,
                'candidate': last['candidate'] if last else None,
                'matched_text': last['matched_text'] if last else None,
                'comparison_text': last['comparison_text'] if last else None,
                'normalization': last['normalization'] if last else None,
                'blocked_by': last['blocked_by'] if last else None,
                'decision_reason': last['decision_reason'] if last else 'waiting',
                'selection_reason': last.get('selection_reason') if last else None,
                'confirmation_waiting': state['confirmation_waiting'],
                'candidate_scores': last['candidate_scores'] if last else [],
                'ambiguous': last['ambiguous'] if last else False,
                'tied_targets': last['tied_targets'] if last else [],
                'matched_target': state['streak_target'] if state['outcome'] == 'accepted' else None,
                'passed': last['passed'] if last else False,
                'prediction': last['prediction'] if last else None,
                'record': state['record']}
