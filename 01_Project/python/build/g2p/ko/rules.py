"""Korean phonological rules, without the morphological analyser.

g2pkk needs mecab, and mecab needs a 112 MB dictionary that cannot ship
inside a Unity build. Only one of its nine steps uses it: the annotate
pass that marks 조사/어미/어간, which five of the twelve special rules
then key off. Everything else - the 402-entry coda+onset table, the
linking rules, and seven of the special rules - is plain substitution on
a jamo string.

Dropping those five costs nothing here. They fire on 관형형 ㄹ 뒤 된소리
("만날 거야" -> "만날 꺼야") and 용언 어간 뒤 경음화 ("심자" -> "심짜"),
which need a word to carry an ending; the product asks for single words.
Measured: identical output on all 29 curriculum words and on 98.3% of
1,500 corpus sentences. End to end, detection went 29.9% -> 29.5% on
child speech and was unchanged on the adult curriculum.

The point of having it here is that C# can run the same thing. Answers
and user speech then pass through identical logic, which is what the
matching assumes and what the shipped app has not been doing - Unity
skips the rules entirely today, so a perfectly spoken 쳐 scores 0.700
against a threshold of 0.850 and can never be confirmed.

English, arabic numerals and the idiom table are deliberately absent:
the domain is Korean words.
"""

import re

from .rules_data import LINK, TABLE

_BASE = 0xAC00
_N_MEDIAL, _N_FINAL = 21, 28
_INITIAL_BASE, _MEDIAL_BASE, _FINAL_BASE = 0x1100, 0x1161, 0x11A7


def _decompose(text: str) -> str:
    """Hangul syllables to conjoining jamo; everything else untouched."""
    out = []
    for ch in text:
        code = ord(ch) - _BASE
        if 0 <= code < 11172:
            initial, rest = divmod(code, _N_MEDIAL * _N_FINAL)
            medial, final = divmod(rest, _N_FINAL)
            out.append(chr(_INITIAL_BASE + initial))
            out.append(chr(_MEDIAL_BASE + medial))
            if final:
                out.append(chr(_FINAL_BASE + final))
        else:
            out.append(ch)
    return "".join(out)


def _compose(letters: str) -> str:
    """Jamo back to syllables, supplying a silent onset where needed."""
    letters = re.sub("(^|[^\u1100-\u1112])([\u1161-\u1175])",
                     lambda m: m.group(1) + "\u110b" + m.group(2), letters)
    out = []
    i = 0
    while i < len(letters):
        ch = letters[i]
        if (0x1100 <= ord(ch) <= 0x1112 and i + 1 < len(letters)
                and 0x1161 <= ord(letters[i + 1]) <= 0x1175):
            initial = ord(ch) - _INITIAL_BASE
            medial = ord(letters[i + 1]) - _MEDIAL_BASE
            final, step = 0, 2
            if (i + 2 < len(letters)
                    and 0x11A8 <= ord(letters[i + 2]) <= 0x11C2):
                final = ord(letters[i + 2]) - _FINAL_BASE
                step = 3
            out.append(chr(_BASE + (initial * _N_MEDIAL + medial) * _N_FINAL
                           + final))
            i += step
        else:
            out.append(ch)
            i += 1
    return "".join(out)


# --- special rules that do not need the analyser ----------------------

def _sub(pattern: str, before: str, after: str, s: str) -> str:
    """Substitute, wrapping group 1 in literal text.

    A callable replacement rather than a backslash template: the same
    rules have to be written again in C#, and a template that reads
    "\\1ᅥ" in one language and "$1ᅥ" in the other is a place for the two
    to drift apart without the tests noticing.
    """
    return re.sub(pattern, lambda m: before + m.group(1) + after, s)


def _jyeo(s: str) -> str:
    """5.1  져/쪄/쳐 -> 저/쩌/처."""
    return _sub("([ᄌᄍᄎ])ᅧ", "", "ᅥ", s)


def _consonant_ui(s: str) -> str:
    """5.3  자음 뒤 ㅢ -> ㅣ."""
    return _sub("([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ", "", "ᅵ", s)


def _jamo_names(s: str) -> str:
    """16  letter names: 디귿이 -> 디그시."""
    s = _sub("([그])ᆮᄋ", "", "ᄉ", s)
    s = _sub("([으])[ᆽᆾᇀᇂ]ᄋ", "", "ᄉ", s)
    s = _sub("([으])[ᆿ]ᄋ", "", "ᄀ", s)
    s = _sub("([으])[ᇁ]ᄋ", "", "ᄇ", s)
    return s


def _balb(s: str) -> str:
    """10.1  밟- and 넓죽/넓둥 keep ㅂ."""
    s = re.sub("(바)ᆲ($|[^ᄋᄒ])",
               lambda m: m.group(1) + "ᆸ" + m.group(2), s)
    s = re.sub("(너)ᆲ([ᄌᄍ]ᅮ|[ᄃᄄ]ᅮ)",
               lambda m: m.group(1) + "ᆸ" + m.group(2), s)
    return s


def _palatalize(s: str) -> str:
    """17  구개음화."""
    s = _sub("ᆮᄋ([ᅵᅧ])", "ᄌ", "", s)
    s = _sub("ᇀᄋ([ᅵᅧ])", "ᄎ", "", s)
    s = _sub("ᆴᄋ([ᅵᅧ])", "ᆯᄎ", "", s)
    s = _sub("ᆮᄒ([ᅵ])", "ᄎ", "", s)
    return s


_SPECIAL = (_jyeo, _consonant_ui, _jamo_names, _balb, _palatalize)


def apply_rules(text: str) -> str:
    """Surface-form Hangul: what the written word actually sounds like."""
    if not text or not text.strip():
        return text
    out = _decompose(text)
    for rule in _SPECIAL:
        out = rule(out)
    for pattern, repl in TABLE:
        out = re.sub(pattern, repl, out)
    for a, b in LINK:
        out = out.replace(a, b)
    return _compose(out)


__all__ = ["apply_rules"]
