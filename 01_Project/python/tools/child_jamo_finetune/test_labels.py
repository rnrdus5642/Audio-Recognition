"""The target contract, pinned. Run with the repository's pytest.

    .venv-gpu/Scripts/python.exe -m pytest python/tools/child_jamo_finetune/test_labels.py
"""
from pathlib import Path
import hashlib
import importlib
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
import labels


def ids(text):
    _norm, _surface, out = labels.targets(text)
    return out


def tokens(text):
    return [labels.TOKENS[i] for i in ids(text)]


def test_vocabulary_is_the_comparison_vocabulary():
    assert len(labels.TOKENS) == 42
    assert labels.TOKENS[0] == '<blank>' and labels.TOKENS[1] == '|'
    assert len(set(labels.TOKENS)) == 42


def test_sentence_punctuation_is_not_spoken():
    assert labels.normalize('문을 두드려.') == '문을 두드려'
    assert labels.normalize('어휴, 무거워!') == '어휴 무거워'
    assert labels.normalize('﻿사랑하는　엄마가　생각났어요．') == \
        '사랑하는 엄마가 생각났어요'


def test_digits_latin_and_brackets_are_unsupported_not_deleted():
    # "(떠랐습니다)" marks an actual realisation, "(틀)" marks an insertion;
    # neither can be resolved from the transcript, so the row is excluded.
    for raw in ['미끄럼을(틀) 타요.', '왕자는 떠올랐습니다(떠랐습니다)',
                '어항이[어하니] 흔들려[흐뜨여].', '나는 tv를 보는데',
                '엘리자베스 2세의 얼굴', 'My dad likes to draw']:
        reason, surface, out = labels.targets(raw)
        assert out is None and reason == 'unsupported_characters', raw


def test_blank_and_single_jamo_transcripts_are_excluded():
    assert labels.targets('   ')[2] is None
    assert labels.targets('...')[0] == 'blank'


def test_rules_reach_the_targets():
    # 쳐 -> 처 (5.1), 놓아 -> 노아 (12), 두드려 stays put.
    assert tokens('쳐') == ['ㅊ', 'ㅓ']
    assert tokens('놓아') == ['ㄴ', 'ㅗ', 'ㅇ', 'ㅏ']
    assert tokens('두드려') == ['ㄷ', 'ㅜ', 'ㄷ', 'ㅡ', 'ㄹ', 'ㅕ']


def test_word_boundary_between_words_only():
    out = tokens('북채 들어')
    assert out.count('|') == 1 and out[0] != '|' and out[-1] != '|'
    assert tokens(' 잡아 ') == tokens('잡아')


def test_final_consonants_use_the_compatibility_jamo():
    assert tokens('실로폰채')[:3] == ['ㅅ', 'ㅣ', 'ㄹ']
    assert '|' not in tokens('실로폰채')


def test_ctc_width_counts_repeated_targets():
    assert labels.ctc_width([2, 3, 4]) == 3
    assert labels.ctc_width([2, 2, 3]) == 4
    assert labels.ctc_width([2, 2, 2]) == 5


def test_crop_policy_is_shared_and_records_malformed_metadata():
    """Selection, the audit and training must crop identically.

    A recording whose SpeechStart/SpeechEnd cannot be read falls back to
    the whole file and says so, rather than raising in one caller and
    silently changing the crop in another.
    """
    assert audio.span('1.0', '3.0', 4.0) == (0.75, 3.25, 'annotated_span')
    assert audio.span('0.1', '9.9', 4.0)[:2] == (0.0, 4.0)      # clipped
    assert audio.span('', '3.0', 4.0) == (0.0, 4.0, 'whole_file_unparsable_span')
    assert audio.span('N/A', 'N/A', 4.0)[2] == 'whole_file_unparsable_span'
    assert audio.span('3.0', '1.0', 4.0)[2] == 'whole_file_implausible_span'
    assert audio.span('2.0', '2.0', 4.0)[2] == 'whole_file_implausible_span'

    x = np.arange(64000, dtype=np.float32)
    cropped, policy = audio.crop(x, '1.0', '3.0')
    assert policy == 'annotated_span' and len(cropped) == 40000
    whole, policy = audio.crop(x, 'bad', 'bad')
    assert policy == 'whole_file_unparsable_span' and len(whole) == len(x)


def test_encoder_frame_arithmetic_matches_the_conv_stack():
    assert audio.frames(400) == 1
    assert audio.frames(16000) == 49          # one second
    assert audio.frames(8000) == 24           # the shortest utterances we keep
    assert audio.frames(240000) == 749        # the longest
    assert audio.frames(200) == 0


def test_contract_hash_is_stable_and_version_tagged():
    assert labels.contract_sha256() == labels.contract_sha256()
    assert labels.LABELS_VERSION == 'child-jamo-2026-09-15'


def test_contract_hash_covers_this_module_not_just_its_constants():
    """Editing normalize()/jamo()/targets() must invalidate a manifest."""
    source = Path(labels.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() != labels.contract_sha256()
    before = labels.contract_sha256()
    path = Path(labels.__file__)
    path.write_bytes(source + b'\n# probe\n')
    try:
        importlib.reload(labels)
        assert labels.contract_sha256() != before
    finally:
        path.write_bytes(source)
        importlib.reload(labels)
    assert labels.contract_sha256() == before


def test_delimiters_are_rejected_rather_than_stripped():
    """Checked against the corpus: these join or delimit, they do not decorate.

    Stripping '#' would turn the trailing marker in "좋아 #" into a clean
    sentence, and stripping '·' would fuse "아빠·엄마" into one word.
    """
    for raw in ['내가 좋아하는 사촌들과 놀 수 있어서 좋아 #',
                '솔: 솔직히 울고 싶을 때가 있지',
                '두더지는 아빠·엄마와 함께 살고 있어요',
                '학습자간, 학습자-교수자간 상호작용',
                '갈비가 달고 맛있었다.;']:
        reason, _surface, out = labels.targets(raw)
        assert out is None and reason == 'unsupported_characters', raw


def test_decoration_is_still_removed():
    """Prolongation and ellipsis are not spoken words."""
    assert labels.normalize('아~ 버튼~ 금방 생각났다.') == '아 버튼 금방 생각났다'
    assert labels.normalize('나는 혼자는 싫은데…') == '나는 혼자는 싫은데'
    assert tokens('"쳐"') == tokens('쳐')
