"""What the acoustic head can emit, and what the curriculum asks for.

The width of a CTC head is `config.vocab_size`, never `len(tokenizer)`.
Most of these tests exist to keep those two numbers from being confused
again, because the difference is invisible until a checkpoint with extra
sentence markers ships and every token id after them is off by two.
"""

import json
import subprocess
import sys
import unicodedata

import pytest

from python.tools import vocab_coverage as vc

NWA = "놔"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8")
    return path


def checkpoint(tmp_path, vocab, size, added=None, name="ckpt"):
    """A checkpoint directory with only the files the loader reads."""
    directory = tmp_path / name
    write(directory / "vocab.json", vocab)
    write(directory / "config.json", {"vocab_size": size})
    write(directory / "tokenizer_config.json",
          {"pad_token": "[PAD]", "unk_token": "[UNK]",
           "word_delimiter_token": "|"})
    if added:
        write(directory / "added_tokens.json", added)
    return directory


BASE = {"가": 0, "나": 1, "|": 2, "[UNK]": 3, "[PAD]": 4}


def test_the_shipped_export_is_bounded_by_its_own_declared_width():
    vocab = vc.load_exported(vc.DEFAULT_VOCAB)

    assert vocab.size == 1205
    assert len(vocab.ids) == 1205
    assert vocab.blank_id == 1204
    assert vocab.unk_id == 1203
    assert vocab.delimiter_id == 859
    assert vocab.emits("사") and not vocab.emits(NWA)


def test_an_export_that_miscounts_its_tokens_is_refused(tmp_path):
    path = write(tmp_path / "v.json", {"vocab_size": 9, "tokens": ["가"]})
    with pytest.raises(vc.VocabularyError, match="vocab_size"):
        vc.load_exported(path)


def test_a_checkpoints_width_comes_from_the_config_not_the_tokenizer(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5, added={"<s>": 5, "</s>": 6})

    vocab = vc.load_checkpoint(directory)

    assert vocab.size == 5
    assert vocab.beyond == {"<s>": 5, "</s>": 6}
    assert not vocab.emits("<s>")
    assert (vocab.blank_id, vocab.unk_id, vocab.delimiter_id) == (4, 3, 2)


def test_added_tokens_are_read_from_the_tokenizer_config_when_alone(tmp_path):
    # transformers writes the ids into tokenizer_config.json and skips
    # added_tokens.json depending on version; both have to work.
    directory = checkpoint(tmp_path, BASE, 5)
    write(directory / "tokenizer_config.json", {
        "pad_token": "[PAD]", "unk_token": "[UNK]",
        "word_delimiter_token": "|",
        "added_tokens_decoder": {
            "4": {"content": "[PAD]"}, "5": {"content": "<s>"}}})

    vocab = vc.load_checkpoint(directory)

    assert vocab.beyond == {"<s>": 5}
    assert vocab.size == 5


def test_a_marker_inside_the_acoustic_range_is_refused(tmp_path):
    # The failure this guards: add_tokens + resize leaves a sentence
    # marker with a logit column, so the head can emit it into a
    # transcript.
    directory = checkpoint(tmp_path, BASE, 6, added={"<s>": 5})
    with pytest.raises(vc.VocabularyError, match="inside the acoustic"):
        vc.load_checkpoint(directory)


def test_two_tokens_sharing_an_id_are_refused(tmp_path):
    directory = checkpoint(tmp_path, dict(BASE, 다=1), 5)
    with pytest.raises(vc.VocabularyError, match="used by both"):
        vc.load_checkpoint(directory)


def test_a_hole_below_the_width_is_refused(tmp_path):
    directory = checkpoint(tmp_path, BASE, 7)
    with pytest.raises(vc.VocabularyError, match="no token for id"):
        vc.load_checkpoint(directory)


def test_a_token_added_at_the_id_it_already_has_is_redundant_not_wrong(
        tmp_path):
    # Real checkpoints mirror [PAD], [UNK] and the delimiter in both
    # tables. Saying the same thing twice is not a fault.
    directory = checkpoint(tmp_path, BASE, 5, added={"[PAD]": 4})

    vocab = vc.load_checkpoint(directory)

    assert vocab.blank_id == 4
    assert vocab.beyond == {}


def test_a_token_added_at_a_different_id_is_refused(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5, added={"[PAD]": 9})
    with pytest.raises(vc.VocabularyError, match="추가 토큰"):
        vc.load_checkpoint(directory)


def test_two_added_tokens_claiming_one_id_are_refused(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5,
                           added={"<s>": 5, "</s>": 5})
    with pytest.raises(vc.VocabularyError, match="함께"):
        vc.load_checkpoint(directory)


def test_the_two_added_token_tables_must_agree(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5, added={"<s>": 5})
    write(directory / "tokenizer_config.json", {
        "pad_token": "[PAD]", "unk_token": "[UNK]",
        "word_delimiter_token": "|",
        "added_tokens_decoder": {"6": {"content": "<s>"}}})

    with pytest.raises(vc.VocabularyError, match="tokenizer_config"):
        vc.load_checkpoint(directory)


def test_a_bare_vocab_file_says_its_width_was_never_verified(tmp_path):
    path = write(tmp_path / "vocab.json", BASE)

    vocab = vc.load_vocabulary(path)

    assert vocab.size == 5
    assert vocab.width_source == "assumed"
    assert vocab.width_verified is False
    assert vocab.describe()["width_verified"] is False


def test_a_checkpoint_without_a_config_will_not_guess_its_width(tmp_path):
    """The widest id counts markers the head may have no column for."""
    directory = tmp_path / "ckpt"
    write(directory / "vocab.json", BASE)

    with pytest.raises(vc.VocabularyError, match="config.json"):
        vc.load_vocabulary(directory)

    # A size the caller typed is a claim about the head, not proof of it.
    declared = vc.load_vocabulary(directory, size=5)
    assert declared.size == 5
    assert declared.width_source == "declared"
    assert declared.width_verified is False


def test_a_width_that_came_from_a_model_says_where_it_came_from(tmp_path):
    assert vc.load_vocabulary(vc.DEFAULT_VOCAB).width_source == "export"
    directory = checkpoint(tmp_path, BASE, 5, added={"<s>": 5, "</s>": 6})
    assert vc.load_checkpoint(directory).width_source == "config"


def test_an_allowed_syllable_may_be_written_decomposed(capsys):
    assert vc.main(["--strict", "--allow",
                    unicodedata.normalize("NFD", NWA)]) == 0


def test_load_vocabulary_takes_a_directory_an_export_or_a_vocab_file(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5, added={"<s>": 5})
    export = write(tmp_path / "export.json",
                   {"vocab_size": 2, "tokens": ["가", "나"]})

    assert vc.load_vocabulary(directory).size == 5
    assert vc.load_vocabulary(directory / "vocab.json").size == 5
    assert vc.load_vocabulary(export).size == 2


@pytest.mark.parametrize("payload", [[], {}, {"가": "0"}, {"tokens": 3}])
def test_something_that_is_not_a_vocabulary_is_refused(tmp_path, payload):
    path = write(tmp_path / "junk.json", payload)
    with pytest.raises((vc.VocabularyError, TypeError, KeyError)):
        vc.load_vocabulary(path)


@pytest.mark.parametrize("text,expected", [
    ("놔요", ["놔", "요"]),
    ("사과 먹어요", list("사과먹어요")),
    ("a1!", []),
])
def test_only_hangul_syllables_are_checked(text, expected):
    assert vc.syllables(text) == expected


def test_a_decomposed_target_is_still_the_syllable_it_spells(tmp_path):
    """The failure this guards is silent and total.

    A target written as jamo (U+1102 U+116A) holds no precomposed
    syllable, so the audit would find nothing to look up and call the
    word supported. It would then be shipped and never transcribed.
    """
    decomposed = unicodedata.normalize("NFD", NWA)
    assert decomposed != NWA

    assert vc.syllables(decomposed) == [NWA]

    vocab = vc.load_vocabulary(vc.DEFAULT_VOCAB)
    report = vc.audit([vc.Entry("t.json", "put", decomposed)], vocab)
    assert list(report["unsupported"]) == [NWA]


def test_a_vocabulary_stored_decomposed_is_refused_not_repaired(tmp_path):
    """Composing a token key would change what its id decodes to."""
    decomposed = unicodedata.normalize("NFD", "나")
    directory = checkpoint(
        tmp_path, {"가": 0, decomposed: 1, "|": 2, "[UNK]": 3, "[PAD]": 4}, 5)

    with pytest.raises(vc.VocabularyError, match="NFC"):
        vc.load_checkpoint(directory)


def test_a_vocabulary_holding_both_forms_says_so(tmp_path):
    both = dict(BASE)
    both[unicodedata.normalize("NFD", "나")] = 5
    directory = checkpoint(tmp_path, both, 6)

    with pytest.raises(vc.VocabularyError, match="이미 NFC"):
        vc.load_checkpoint(directory)


def test_special_tokens_may_be_written_as_added_token_mappings(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5)
    write(directory / "tokenizer_config.json", {
        "pad_token": {"content": "[PAD]", "lstrip": True},
        "unk_token": {"content": "[UNK]"},
        "word_delimiter_token": "|"})

    vocab = vc.load_checkpoint(directory)

    assert (vocab.blank_id, vocab.unk_id, vocab.delimiter_id) == (4, 3, 2)


def test_unreadable_special_metadata_is_a_vocabulary_error(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5)
    write(directory / "tokenizer_config.json", {"pad_token": {"id": 4}})

    with pytest.raises(vc.VocabularyError, match="pad_token"):
        vc.load_checkpoint(directory)


def test_a_special_token_nobody_declared_stays_unknown(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5)
    write(directory / "tokenizer_config.json", {"pad_token": "[PAD]"})

    vocab = vc.load_checkpoint(directory)

    assert vocab.blank_id == 4
    assert vocab.unk_id is None and vocab.delimiter_id is None


def test_a_blank_the_head_cannot_emit_is_refused(tmp_path):
    directory = checkpoint(tmp_path, BASE, 4, added={"[PAD]": 4})
    write(directory / "tokenizer_config.json", {"pad_token": "[PAD]"})

    with pytest.raises(vc.VocabularyError, match="음향 범위 밖"):
        vc.load_checkpoint(directory)


def test_a_declared_special_token_must_exist(tmp_path):
    directory = checkpoint(tmp_path, BASE, 5)
    write(directory / "tokenizer_config.json", {"pad_token": "<pad>"})

    with pytest.raises(vc.VocabularyError, match="어휘에 없습니다"):
        vc.load_checkpoint(directory)


def test_an_answer_is_reported_once_per_missing_syllable(tmp_path):
    vocab = vc.load_vocabulary(checkpoint(tmp_path, BASE, 5))
    entries = [vc.Entry("t.json", "a", "가나"),
               vc.Entry("t.json", "b", "다다라")]

    report = vc.audit(entries, vocab)

    assert sorted(report["unsupported"]) == ["다", "라"]
    assert report["unsupported"]["다"]["answers"] == ["t.json:b"]
    assert report["unsupported"]["다"]["texts"] == ["다다라"]
    assert report["sources"][0]["unsupported_answers"] == 1
    assert report["checked"] == {"answers": 2, "unique_syllables": 4,
                                 "sources": 1}


def test_a_covered_curriculum_reports_nothing(tmp_path):
    vocab = vc.load_vocabulary(checkpoint(tmp_path, BASE, 5))

    report = vc.audit([vc.Entry("t.json", "a", "가 나")], vocab)

    assert report["unsupported"] == {}
    assert "매칭은 IPA" in report["_note"]
    assert "모든 음절" in vc.format_report(report)


def test_targets_and_words_read_the_same_answers(tmp_path):
    targets = write(tmp_path / "targets.json",
                    {"answers": [{"id": "put_plain", "text": NWA}]})
    words = tmp_path / "words.csv"
    words.write_text("answer_id,text,language\nput_plain,놔,ko\n",
                     encoding="utf-8-sig")

    assert (vc.read_source(targets)[0].word_id
            == vc.read_source(words)[0].word_id == "put_plain")
    assert vc.read_source(words)[0].text == NWA


def test_the_older_grouped_targets_format_is_still_read(tmp_path):
    """`targets_norules.json` is still the 1.0.0 segments shape.

    Reading it as an empty list would have reported perfect coverage for
    a file nothing had actually looked at.
    """
    path = write(tmp_path / "targets.json", {
        "version": "1.0.0",
        "segments": [{"id": "lesson_01", "answers": [
            {"id": "put_plain", "text": NWA}]},
            {"id": "lesson_02", "answers": [{"id": "mom", "text": "엄마"}]}]})

    entries = vc.read_source(path)

    assert [(e.word_id, e.text) for e in entries] == [
        ("put_plain", NWA), ("mom", "엄마")]


@pytest.mark.parametrize("payload", [
    {"version": "9", "lessons": []},
    {"answers": []},
    {"segments": [{"id": "x"}]},
    {"answers": [{"id": "a"}]},
])
def test_a_targets_file_this_cannot_read_is_never_silently_empty(
        tmp_path, payload):
    path = write(tmp_path / "targets.json", payload)
    with pytest.raises(vc.VocabularyError):
        vc.read_source(path)


def test_the_generated_catalogues_can_be_audited_too():
    present = [p for p in vc.GENERATED_SOURCES if p.exists()]
    if not present:
        pytest.skip("빌드 산출물 없음 (gitignored)")

    assert vc.main(["--include-generated"]) == 0
    for path in present:
        assert vc.read_source(path), f"{path} 에서 답을 하나도 못 읽었습니다"


def test_the_only_unsupported_curriculum_syllable_is_the_known_one():
    """Characterisation, not aspiration.

    Adding a word the model cannot spell turns this red before anyone
    records a child saying it. It also turns red the day 놔 becomes
    emittable, which is the moment the READMEs and the shipped
    vocabulary have to be revisited together.
    """
    vocab = vc.load_vocabulary(vc.DEFAULT_VOCAB)
    entries = [e for path in vc.DEFAULT_SOURCES
               for e in vc.read_source(path)]

    report = vc.audit(entries, vocab)

    assert list(report["unsupported"]) == [NWA]
    assert sorted(report["unsupported"][NWA]["texts"]) == ["놔", "놔요"]
    assert report["checked"]["sources"] == len(vc.DEFAULT_SOURCES)
    assert all(s["unsupported_answers"] == 2 for s in report["sources"]), \
        "every shipped copy of the curriculum should agree"


def test_the_cli_only_fails_on_an_unexpected_syllable(capsys):
    assert vc.main([]) == 0
    assert vc.main(["--strict"]) == 1
    assert vc.main(["--strict", "--allow", NWA]) == 0
    assert NWA in capsys.readouterr().out


def test_the_cli_writes_the_report_it_printed(tmp_path):
    out = tmp_path / "nested" / "report.json"

    assert vc.main(["--json", str(out)]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["vocabulary"]["size"] == 1205
    assert list(report["unsupported"]) == [NWA]


def test_the_cli_says_which_file_is_missing(tmp_path, capsys):
    assert vc.main(["--source", str(tmp_path / "nope.json")]) == 1
    assert "nope.json" in capsys.readouterr().err


def test_importing_the_module_leaves_the_console_alone():
    """Encoding is the caller's business until the CLI is the caller.

    Reconfiguring stdout at import time reaches into whichever program
    imported the module - a test runner, a notebook, another tool - and
    changes its output stream as a side effect of a lookup.
    """
    code = ("import sys; before = sys.stdout.encoding;"
            "import python.tools.vocab_coverage;"
            "print(before == sys.stdout.encoding)")

    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(vc.PROJECT_ROOT))

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "True"
