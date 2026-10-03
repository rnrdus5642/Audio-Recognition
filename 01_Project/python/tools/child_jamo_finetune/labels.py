"""Transcript -> jamo CTC targets, with an explicit, versioned contract.

The 42 outputs are the same as the encoder comparison: blank, a word
boundary, and the 40 compatibility jamo that the project's
`KoreanG2P.apply_rules` surface form decomposes into.

A target is the *expected surface form* of the corpus transcript
(`Transcription.LabelText`) under the project's phonological rules.
LabelText is documented as a transcription, and some rows carry
realisation or insertion notation, so it cannot be asserted that every
row is simply the prompt a child was given. What can be said is narrow
and sufficient: the target is what these rules derive from the recorded
transcript. It is not an expert annotation of the pronunciation the
child produced, and nothing computed from it measures pronunciation-error
diagnosis.

Normalisation is deliberate rather than incidental, and every decision
below was checked against the actual corpus rather than assumed:

  removed, because the corpus uses them as ordinary sentence punctuation
  and they are not spoken - `. , ? !`, the prolongation `~` ("아~"),
  the trailing ellipsis `…`, and quotation marks around quoted speech.

  rejected as unsupported, because in this corpus they delimit or join
  rather than decorate, and stripping them would silently promote
  annotation to speech or fuse two spoken words:
    `#`  trailing marker ("좋아 #", "좋아 #13") - 12 rows
    `:`  acrostic notation ("솔: 솔직히...") - 24 rows
    `·`  stands for "and" between two spoken words ("아빠·엄마") - 16 rows
    `-`  joins words ("학습자-교수자간") - 11 rows
    `;`  1 row
  Together 64 rows in 2.77M, so the conservative choice is nearly free.

Everything else that is not Hangul or a space - digits, Latin, and the
parenthesised or bracketed segments the corpus sometimes uses to note an
actual realisation ("읽었어요(일저더요)") and sometimes an insertion
("미끄럼을(틀)") - also makes the transcript unsupported. Those rows are
excluded and counted. Guessing which of the two a bracket means would
put invented text into the targets.
"""
from pathlib import Path
import hashlib
import re
import sys
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from python.build.g2p.ko.g2p import KoreanG2P
from python.build.g2p.ko.jamo_ipa import INITIALS, MEDIALS, FINALS

BLANK, BOUNDARY = 0, 1
TOKENS = ['<blank>', '|'] + list(INITIALS) + list(MEDIALS)
VOCAB = {s: i for i, s in enumerate(TOKENS)}
G2P = KoreanG2P()

# Removed before the Hangul-only check: none of these are spoken.
DROP = '.,?!~…"\'‘’“”'
SPACE = '　\xa0\t\n\r\v\f'

LABELS_VERSION = 'child-jamo-2026-09-15'

def normalize(text):
    """NFC, strip what is not spoken, collapse spacing."""
    out = unicodedata.normalize('NFC', text).replace('﻿', '')
    out = out.replace('．', '.').replace('，', ',')
    for ch in SPACE:
        out = out.replace(ch, ' ')
    for ch in DROP:
        out = out.replace(ch, ' ')
    return ' '.join(out.split())


def supported(text):
    return bool(text) and re.fullmatch(r'[가-힣 ]+', text) is not None


def jamo(surface):
    """Compatibility jamo for one surface-form string, or None."""
    out = []
    for ch in surface:
        if ch == ' ':
            if out and out[-1] != '|':
                out.append('|')
            continue
        if not ('가' <= ch <= '힣'):
            return None
        n = ord(ch) - 0xAC00
        out.append(INITIALS[n // 588])
        out.append(MEDIALS[n // 28 % 21])
        # The rules resolve every cluster coda, so the 11 that have no
        # single compatibility jamo never reach here; if one ever does,
        # `unmapped_jamo` rejects the row rather than inventing a coda.
        final = FINALS[n % 28]
        if final:
            out.append(final)
    while out and out[-1] == '|':
        out.pop()
    return out if all(c in VOCAB for c in out) else None


def targets(raw_text):
    """(normalised text, surface form, token ids) or (reason, None, None)."""
    text = normalize(raw_text)
    if not text:
        return 'blank', None, None
    if not supported(text):
        return 'unsupported_characters', None, None
    surface = G2P.apply_rules(text)
    ids = jamo(surface)
    if ids is None:
        return 'unmapped_jamo', None, None
    if len(ids) < 2:
        return 'too_few_targets', None, None
    return text, surface, [VOCAB[c] for c in ids]


def ctc_width(ids):
    """Frames a valid CTC alignment needs: repeats force a blank between."""
    return len(ids) + sum(a == b for a, b in zip(ids, ids[1:]))


def contract_sha256():
    """Hash of everything that decides a target, for the manifest.

    This module's own bytes are included: `normalize`, `jamo` and
    `targets` decide the labels just as much as the rule tables do, and
    editing one of them has to invalidate a manifest built with the old
    behaviour. Hashing only the version string and the constants would
    let a changed decomposition reuse a stale manifest silently.
    """
    h = hashlib.sha256()
    h.update(LABELS_VERSION.encode())
    h.update('\x00'.join(TOKENS).encode())
    h.update((DROP + '\x00' + SPACE).encode())
    h.update(Path(__file__).resolve().read_bytes())
    for path in ['python/build/g2p/ko/rules.py', 'python/build/g2p/ko/rules_data.py',
                 'python/build/g2p/ko/jamo_ipa.py']:
        h.update(Path(__file__).resolve().parents[3].joinpath(path).read_bytes())
    return h.hexdigest()
