"""Turn predicted jamo into readable Hangul without inventing anything.

The model emits 40 compatibility jamo plus a word boundary. Those are
phonological targets, not orthography, so the job here is only to stack
jamo into syllables where the sequence genuinely allows it - never to
guess at a word, consult a dictionary, or lean on the reference text.

A run that cannot be composed stays visible as the jamo it actually was.
That matters more than a tidy line of Hangul: a stray 'ㅅ' the model
produced is evidence, and silently dropping it or rounding it into a
nearby syllable would misrepresent what the model did.

The silent onset is explicit. `labels.jamo` writes every syllable as
initial + medial (+ final), so a vowel-initial syllable arrives as
'ㅇ' + vowel and composes back to the same syllable. A bare vowel with
no onset is therefore not a silent-onset syllable - it is an orphan, and
is shown as one.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_jamo_finetune'))
import labels
from labels import FINALS, INITIALS, MEDIALS

BASE = 0xAC00
INITIAL_INDEX = {c: i for i, c in enumerate(INITIALS)}
MEDIAL_INDEX = {c: i for i, c in enumerate(MEDIALS)}
#: FINALS[0] is the empty coda, so the real codas start at 1.
FINAL_INDEX = {c: i for i, c in enumerate(FINALS) if i and c}
VOWELS = set(MEDIALS)

#: Bumped whenever this file changes how a prediction is rendered, so a
#: cached result rendered by older logic is not reused.
DECODE_VERSION = 'compose-1'


def collapse(ids):
    """CTC greedy decode, identical to the training decoder.

    Merge runs of equal frames, then drop blanks - in that order. The
    order is the whole point: two of the same jamo separated by a blank
    survive as two, which is how the model spells a genuine double.
    """
    out = []
    for i in ids:
        if not out or out[-1] != i:
            out.append(i)
    return [i for i in out if i != labels.BLANK]


def tokens(ids):
    return [labels.TOKENS[i] for i in ids]


def _syllable(initial, medial, final):
    return chr(BASE + (INITIAL_INDEX[initial] * 21 + MEDIAL_INDEX[medial]) * 28
               + (FINAL_INDEX[final] if final else 0))


def compose_word(jamo):
    """One word's jamo into segments: composed syllables and orphans.

    Greedy and local. A consonant becomes a coda only when the next jamo
    is not a vowel; if it is, that consonant is the onset of the next
    syllable instead. Nothing here looks beyond that one lookahead, so
    the output is a function of the prediction alone.
    """
    segments = []
    i, n = 0, len(jamo)
    while i < n:
        head = jamo[i]
        if (head in INITIAL_INDEX and i + 1 < n and jamo[i + 1] in VOWELS):
            initial, medial = head, jamo[i + 1]
            i += 2
            final = None
            if i < n and jamo[i] in FINAL_INDEX:
                # Only a coda if it is not the onset of the next syllable.
                if not (i + 1 < n and jamo[i + 1] in VOWELS):
                    final = jamo[i]
                    i += 1
            used = [initial, medial] + ([final] if final else [])
            segments.append({'kind': 'syllable',
                             'text': _syllable(initial, medial, final),
                             'jamo': used})
        else:
            segments.append({'kind': 'orphan', 'text': head, 'jamo': [head]})
            i += 1
    return segments


def compose(ids):
    """Predicted token ids to a readable rendering plus the raw evidence.

    Returns the composed text, the segments behind it, the raw jamo, and
    a count of what could not be composed. Callers show the text; the raw
    jamo stays one click away so nothing is hidden behind the tidy view.
    """
    ids = [i for i in ids if i != labels.BLANK]
    words, current = [], []
    for i in ids:
        if i == labels.BOUNDARY:
            words.append(current)
            current = []
        else:
            current.append(labels.TOKENS[i])
    words.append(current)

    segments, pieces = [], []
    for w, word in enumerate(words):
        if w:
            segments.append({'kind': 'space', 'text': ' ', 'jamo': ['|']})
            pieces.append(' ')
        for segment in compose_word(word):
            segments.append(segment)
            pieces.append(segment['text'])
    orphans = sum(1 for s in segments if s['kind'] == 'orphan')
    return {
        'text': ''.join(pieces),
        'segments': segments,
        'jamo': ' '.join(''.join(w) for w in words),
        'jamo_tokens': [labels.TOKENS[i] for i in ids],
        'orphan_jamo': orphans,
        'syllables': sum(1 for s in segments if s['kind'] == 'syllable'),
        'fully_composed': orphans == 0,
        'decode_version': DECODE_VERSION,
    }


def scored_ids(ids):
    """The token list the training metric scores: blanks and boundaries out."""
    return [i for i in ids if i not in (labels.BLANK, labels.BOUNDARY)]


def edits(reference, hypothesis):
    """Levenshtein ops between two token-id lists, as training measures it."""
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(reference, hypothesis)


def compare(reference_ids, hypothesis_ids):
    """Jamo error rate for one utterance, on the training convention.

    Boundaries are excluded, the denominator is the reference length, and
    the ratio is reported as measured. A model that inserts freely can
    exceed 1.0; clamping that would turn an error rate into a made-up
    confidence.
    """
    reference = scored_ids(reference_ids)
    hypothesis = scored_ids(hypothesis_ids)
    errors = edits(reference, hypothesis)
    return {
        'errors': errors,
        'reference_jamo': len(reference),
        'hypothesis_jamo': len(hypothesis),
        'jer': errors / len(reference) if reference else None,
        'exact': errors == 0,
        'convention': 'training: boundaries ignored, edit ops / reference '
                      'jamo, not clamped',
    }


def alignment(reference_ids, hypothesis_ids):
    """Per-jamo opcodes, so a difference can be seen rather than inferred."""
    from rapidfuzz.distance import Levenshtein
    reference = scored_ids(reference_ids)
    hypothesis = scored_ids(hypothesis_ids)
    out = []
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(reference, hypothesis):
        out.append({
            'op': tag,
            'reference': [labels.TOKENS[t] for t in reference[i1:i2]],
            'hypothesis': [labels.TOKENS[t] for t in hypothesis[j1:j2]],
        })
    return out
