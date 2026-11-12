"""Word boundaries, known action contrasts, and bounded confirmation history."""
import json
from pathlib import Path
import re
import unicodedata
from dataclasses import dataclass


def compact(text):
    return ''.join(unicodedata.normalize('NFC', text).split())


def spelling_key(text):
    # Both spellings occur in the supplied answer list / model transcripts.
    return re.sub('거에요$', '거예요', compact(text))


class ActionGuard:
    def __init__(self):
        data = json.loads(Path(__file__).with_name('answer_families.json').read_text(encoding='utf-8'))
        self.families = {spelling_key(word): family for family, words in data['families'].items() for word in words}

    def conflict(self, heard, target):
        actual, expected = self.families.get(spelling_key(heard)), self.families.get(spelling_key(target))
        return compact(heard) if actual and expected and actual != expected else None


def comparison_spans(text, targets):
    """Complete suffixes; collapse only a complete repetition of a listed word.

    Do not extract arbitrary internal substrings, complete unfinished jamo, or
    shorten one-syllable reduplications such as 하하. ASR text stays untouched.
    """
    tokens = unicodedata.normalize('NFC', text).split()
    for start in range(len(tokens) - 1, -1, -1):
        words = tokens[start:]
        if not all(re.fullmatch('[가-힣]+', word) for word in words):
            break
        raw, joined = ' '.join(words), ''.join(words)
        yield raw, joined, None
        for count in (2, 3, 4):
            width, remainder = divmod(len(joined), count)
            root = joined[:width]
            if not remainder and width >= 2 and root in targets and root * count == joined:
                yield raw, root, 'repeated_word'
                break


@dataclass(frozen=True)
class ContextSpan:
    raw: str
    normalized: str
    reason: str | None
    at_end: bool
    owner: str | None = None
    start: int = 0
    end: int = 0


# Deliberately a finite set of surface forms, not a Korean morphological analyzer.
# Only a complete listed noun at the START of a complete token/phrase may be
# recovered. A known game action keeps its listed inflection (except quotation).
_PARTICLES = frozenset(('은 는 이 가 을 를 도 만 과 와 랑 이랑 하고 의 에 에서 '
                        '에게 한테 로 으로 부터 까지 처럼 보다 마다 밖에 '
                        '에도 에는 에서도 에서는 에서만 에만 으로도 로도 으로는 로는 '
                        '으로만 로만 에게도 한테도 부터는 까지는 만은 만을 만이 만도 '
                        '와도 과도 랑도 이랑도 하고도 도요 만요').split())
_COPULAS = frozenset(('야 이야 예요 이에요 에요 요 이요 입니다 입니다만 인 인거 인것 '
                      '인가 인가요 이네 이네요 네요 이죠 죠 이지 지 이고 이라서 라서 '
                      '이거든 이거든요 거든 거든요 이었어 이었어요 였어 였어요').split())
_QUOTATIONS = frozenset(('라고 이라고 라는 이라는 라며 이라며 라니 이라니 라니까 이라니까').split())
_ATTACHED_GUESS = re.compile(r'인(?:거|것)(?:같(?:아|아요|네|네요|은데|은데요|죠|지))?')
_EDGE_PUNCTUATION = '.,!?;:…。，？！"\'“”‘’()[]{}<>〈〉《》'


def _following_negation(tokens, end):
    # Preserve the existing rejection of an explicit "책 아니야". This is a
    # local guard, not general sentence-meaning or speaker-intent recognition.
    following = tokens[end:end + 2]
    if following and following[0] in ('은', '는', '이', '가'):
        following = following[1:]
    return bool(following and following[0].startswith(('아니', '말고')))


def context_spans(text, targets, action_guard):
    """Whole words/phrases anywhere, plus explicit listed-word affixes.

    An incomplete jamo stays in its token and makes that token ineligible. It
    does not invalidate other complete tokens. No internal character windows.
    """
    tokens = [word.strip(_EDGE_PUNCTUATION) for word in unicodedata.normalize('NFC', text).split()]
    keys = {target: spelling_key(target) for target in targets}
    seen = set()
    # Tail-first ordering preserves the previous choice when scores are equal.
    for end in range(len(tokens), 0, -1):
        if _following_negation(tokens, end):
            continue
        for start in range(end - 1, -1, -1):
            if not re.fullmatch('[가-힣]+', tokens[start]):
                break
            words = tokens[start:end]
            raw, joined = ' '.join(words), ''.join(words)
            at_end = end == len(tokens)
            spans = [ContextSpan(raw, joined, None if at_end else 'context_word', at_end, start=start, end=end)]
            for count in (2, 3, 4):
                width, remainder = divmod(len(joined), count)
                stem = joined[:width]
                if not remainder and width >= 2 and stem in targets and stem * count == joined:
                    spans.append(ContextSpan(raw, stem, 'repeated_word', at_end, start=start, end=end))
                    break
            canonical = spelling_key(joined)
            owners = []
            for target, key in keys.items():
                # Both spellings must still work when a quotation follows;
                # spelling_key only replaces a terminal 거에요.
                for prefix in {compact(target), key}:
                    if not canonical.startswith(prefix):
                        continue
                    ending = canonical[len(prefix):]
                    is_action = key in action_guard.families
                    if (not ending or ending in _QUOTATIONS or
                            (not is_action and (ending in _PARTICLES or ending in _COPULAS or
                                                _ATTACHED_GUESS.fullmatch(ending)))):
                        owners.append((target, len(prefix), ending))
            # If an entire longer candidate is present, do not also manufacture
            # a shorter candidate by treating its ending as a particle.
            longest = max((length for _, length, _ in owners), default=0)
            for target, length, ending in owners:
                if ending and length == longest:
                    spans.append(ContextSpan(raw, target, 'listed_word_affix', at_end, target, start, end))
            for span in spans:
                identity = (span.raw, span.normalized, span.reason, span.at_end, span.owner, span.start, span.end)
                if identity not in seen:
                    seen.add(identity)
                    yield span


def confirm(state, candidate, passes, score, text, elapsed, *, blocked=False, ambiguous=False):
    """Require N positive checks, with at most one uncertain check in between.

    A different positive candidate or clearly different word resets evidence.
    The gap itself never counts as positive. Old evidence expires by audio time.
    """
    required = state['settings']['consecutive']
    threshold = state['settings']['threshold']
    evidence = state.get('evidence', [])
    previous = state.get('streak_target')
    if elapsed - state['elapsed'] > 1.25:
        evidence = []
    evidence = [time for time in evidence if elapsed - time <= (required + .5) * .5]
    gaps = state.get('uncertain_checks', 0)
    if not evidence:
        previous, gaps = None, 0
    uncertain = (not text.strip() or any(0x1100 <= ord(ch) <= 0x11ff for ch in unicodedata.normalize('NFC', text)) or
                 (candidate == previous and score >= max(.7, threshold - .05)))
    waiting = False
    if passes:
        if candidate != previous:
            evidence, gaps = [], 0
        evidence.append(elapsed)
        previous = candidate
    elif evidence and uncertain and not blocked and not ambiguous and gaps == 0:
        gaps += 1
        waiting = True
    else:
        evidence, previous, gaps = [], None, 0
    # Once old evidence has expired, a prior gap must not burden a new trial.
    if not evidence:
        previous, gaps, waiting = None, 0, False
    state.update(evidence=evidence, uncertain_checks=gaps, streak=len(evidence),
                 streak_target=previous, confirmation_waiting=waiting)
    return passes and len(evidence) >= required
