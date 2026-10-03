"""The parts of the viewer that could quietly misrepresent the model.

Composition and the error metric are the two places where a bug would
look like a better - or worse - model rather than like a bug, so they are
pinned here. Styling and layout are not tested; the browser pass covers
what those need.

    .venv-gpu/Scripts/python.exe -m pytest \
        python/tools/child_jamo_web/test_compose.py -q
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_jamo_finetune'))
import compose
import labels

V = {t: i for i, t in enumerate(labels.TOKENS)}


def ids(*jamo):
    return [V[j] for j in jamo]


def text(*jamo):
    return compose.compose(ids(*jamo))['text']


# --- CTC collapse ----------------------------------------------------

def test_collapse_merges_runs_then_drops_blanks():
    assert compose.collapse([2, 2, 2, 0, 0, 21, 21]) == [2, 21]


def test_a_blank_between_equal_frames_keeps_both_jamo():
    """This is how the decoder spells a genuine double jamo."""
    assert compose.collapse([11, 11, 0, 11, 11]) == [11, 11]
    assert compose.collapse([11, 11, 11]) == [11]


def test_collapse_keeps_boundaries():
    assert labels.BOUNDARY in compose.collapse([2, 21, 1, 1, 4, 21])


# --- composition -----------------------------------------------------

def test_silent_onset_composes_back_to_its_syllable():
    """labels.jamo writes vowel-initial syllables as 'ㅇ' + vowel."""
    assert text('ㄴ', 'ㅗ', 'ㅇ', 'ㅏ') == '노아'
    assert text('ㅇ', 'ㅣ', 'ㄱ', 'ㅡ', 'ㄴ') == '이근'


def test_a_bare_vowel_is_an_orphan_not_a_silent_onset_syllable():
    out = compose.compose(ids('ㅏ'))
    assert out['text'] == 'ㅏ' and out['orphan_jamo'] == 1
    assert not out['fully_composed']


def test_coda_versus_next_onset_is_decided_by_one_lookahead():
    assert text('ㄱ', 'ㅏ', 'ㄴ') == '간'
    assert text('ㄱ', 'ㅏ', 'ㄴ', 'ㅏ') == '가나'


def test_word_boundary_becomes_a_space():
    out = compose.compose(ids('ㅇ', 'ㅣ', 'ㄱ', 'ㅡ', 'ㄴ', '|',
                              'ㅅ', 'ㅏ', 'ㄱ', 'ㅘ'))
    assert out['text'] == '이근 사과'
    assert out['jamo'] == 'ㅇㅣㄱㅡㄴ 사과'.replace('사과', 'ㅅㅏㄱㅘ')
    assert out['fully_composed']


def test_consonants_that_cannot_be_codas_stay_visible():
    """ㄸ, ㅃ and ㅉ are onsets only; as a coda they must not be invented."""
    for onset_only in ['ㄸ', 'ㅃ', 'ㅉ']:
        out = compose.compose(ids('ㄱ', 'ㅏ', onset_only))
        assert out['text'] == '가' + onset_only
        assert out['orphan_jamo'] == 1


def test_a_repeated_jamo_run_is_shown_not_smoothed():
    out = compose.compose(ids('ㅅ', 'ㅅ'))
    assert out['text'] == 'ㅅㅅ' and out['orphan_jamo'] == 2


def test_trailing_consonant_without_a_vowel_is_an_orphan():
    out = compose.compose(ids('ㄱ', 'ㅏ', 'ㄴ', 'ㅁ'))
    assert out['text'] == '간ㅁ' and out['orphan_jamo'] == 1


def test_every_predicted_jamo_survives_into_the_rendering():
    """Nothing is dropped: the segments account for all of the input."""
    sequence = ids('ㄱ', 'ㅏ', 'ㄴ', 'ㅁ', '|', 'ㅏ', 'ㅅ', 'ㅅ', 'ㅗ')
    out = compose.compose(sequence)
    used = [j for s in out['segments'] if s['kind'] != 'space'
            for j in s['jamo']]
    assert used == [labels.TOKENS[i] for i in sequence
                    if i != labels.BOUNDARY]


def test_double_coda_jamo_are_valid_codas():
    assert text('ㄱ', 'ㅏ', 'ㄲ') == '갂'
    assert text('ㄱ', 'ㅏ', 'ㅆ') == '갔'


def test_empty_prediction_renders_as_empty_not_as_an_error():
    out = compose.compose([])
    assert out['text'] == '' and out['syllables'] == 0


# --- metrics ---------------------------------------------------------

def test_jer_uses_the_training_convention():
    reference = ids('ㄱ', 'ㅏ', '|', 'ㄴ', 'ㅏ')
    assert compose.compare(reference, reference)['jer'] == 0.0
    assert compose.compare(reference, reference)['exact']
    # Boundaries are excluded from both sides.
    assert compose.compare(reference, reference)['reference_jamo'] == 4


def test_jer_is_not_clamped_when_the_model_over_generates():
    reference = ids('ㄱ', 'ㅏ')
    hypothesis = ids('ㄱ', 'ㅏ', 'ㄴ', 'ㅏ', 'ㄷ', 'ㅏ')
    out = compose.compare(reference, hypothesis)
    assert out['errors'] == 4 and out['jer'] == 2.0


def test_alignment_reports_the_actual_differences():
    ops = compose.alignment(ids('ㄱ', 'ㅏ', 'ㄴ'), ids('ㄱ', 'ㅏ', 'ㅁ'))
    assert any(o['op'] == 'replace' and o['reference'] == ['ㄴ']
               and o['hypothesis'] == ['ㅁ'] for o in ops)


def test_composition_never_reads_the_reference():
    """The renderer takes prediction ids only - there is no way to leak gold."""
    import inspect
    source = inspect.getsource(compose.compose)
    assert 'reference' not in source and 'target' not in source
    assert list(inspect.signature(compose.compose).parameters) == ['ids']
