"""What the trainer refuses to start on.

Every check here runs before the GPU is touched, because the failures it
catches are ones you would otherwise discover after a night of training:
a manifest full of syllables the tokenizer spells `[UNK]`, or a
checkpoint whose head and tokenizer disagree about how wide it is.

No test here starts training. The paths exercised all return before
`torch.cuda.is_available()`.
"""

import csv

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from python.tools.child_finetune import train as tr

NWA = "놔"
VOCAB = {"가": 0, "요": 1, "|": 2, "사": 3, "과": 4, "[UNK]": 5, "[PAD]": 6}


def tokenizer_for(tmp_path, vocab=None):
    from transformers import Wav2Vec2CTCTokenizer

    path = tmp_path / "vocab.json"
    path.write_text(
        __import__("json").dumps(vocab or VOCAB, ensure_ascii=False),
        encoding="utf-8")
    return Wav2Vec2CTCTokenizer(
        str(path), unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|", do_lower_case=False)


def manifest(directory, *texts, name="train.csv"):
    directory.mkdir(parents=True, exist_ok=True)
    fields = ["wav", "text", "seconds", "start", "end", "speaker", "age",
              "style"]
    with open(directory / name, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, text in enumerate(texts):
            writer.writerow({"wav": f"{i}.wav", "text": text,
                             "seconds": "3.0", "start": "", "end": "",
                             "speaker": "s", "age": "6", "style": "read"})
    return directory / name


def test_characters_reads_every_manifest(tmp_path):
    manifest(tmp_path, "사과", name="train.csv")
    manifest(tmp_path, "가요", name="dev.csv")

    found = tr.characters([tmp_path / "train.csv", tmp_path / "dev.csv"])

    assert found == set("사과가요")


def test_a_syllable_with_no_id_is_named(tmp_path):
    tokenizer = tokenizer_for(tmp_path)

    assert tr.unknown(set("사과"), tokenizer, 7) == []
    assert tr.unknown(set("사과" + NWA), tokenizer, 7) == [NWA]


def test_a_space_is_the_delimiter_not_an_unknown(tmp_path):
    tokenizer = tokenizer_for(tmp_path)

    assert tr.unknown({" ", "사"}, tokenizer, 7) == []


def test_a_token_past_the_head_counts_as_unknown(tmp_path):
    """`len(tokenizer)` is not the acoustic width, and this is why.

    The sentence markers have ids the head has no column for. A label
    holding one would index past the logits.
    """
    tokenizer = tokenizer_for(tmp_path)
    assert tokenizer.get_vocab()["<s>"] == 7

    assert tr.unknown({"<s>"}, tokenizer, 7) == ["<s>"]
    # 사 is at id 3: inside a width of 4, outside a width of 3.
    assert tr.unknown(set("사"), tokenizer, 4) == []
    assert tr.unknown(set("사"), tokenizer, 3) == ["사"]


def weightless_checkpoint(tmp_path, vocab=None, width=None):
    """Config and processor only - enough for everything before the GPU."""
    from transformers import (Wav2Vec2Config, Wav2Vec2FeatureExtractor,
                              Wav2Vec2Processor)

    vocab = vocab or VOCAB
    directory = tmp_path / "ckpt"
    directory.mkdir(parents=True, exist_ok=True)
    Wav2Vec2Processor(
        feature_extractor=Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True),
        tokenizer=tokenizer_for(tmp_path, vocab)).save_pretrained(directory)
    Wav2Vec2Config(vocab_size=width or len(vocab),
                   pad_token_id=vocab["[PAD]"]).save_pretrained(directory)
    return directory


def test_a_run_cannot_both_resume_and_start_over(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "train", "--resume", str(tmp_path), "--init-model", str(tmp_path)])

    with pytest.raises(SystemExit) as exit_code:
        tr.main()

    assert exit_code.value.code == 2


def run(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["train", *argv])
    return tr.main()


def test_a_warm_start_has_to_say_which_manifest(tmp_path, monkeypatch,
                                                capsys):
    """The default manifest was built against the shipped vocabulary.

    Warm-starting a wider head on it would train every new row on
    nothing, because the utterances that hold the new syllable were
    filtered out before the manifest was written.
    """
    checkpoint = weightless_checkpoint(tmp_path)

    assert run(monkeypatch, "--init-model", str(checkpoint)) == 1
    assert "--data" in capsys.readouterr().err


def test_a_warm_start_will_not_write_into_a_finished_run(tmp_path,
                                                         monkeypatch, capsys):
    checkpoint = weightless_checkpoint(tmp_path)
    data = tmp_path / "data"
    manifest(data, "사과")
    manifest(data, "사과", name="dev.csv")
    out = tmp_path / "run"
    out.mkdir()
    (out / "history.json").write_text("[]", encoding="utf-8")

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", str(data), "--out", str(out)) == 1
    assert "history.json" in capsys.readouterr().err


def test_a_manifest_the_tokenizer_cannot_spell_stops_the_run(
        tmp_path, monkeypatch, capsys):
    checkpoint = weightless_checkpoint(tmp_path)
    data = tmp_path / "data"
    manifest(data, "사과", NWA + "요")
    manifest(data, "사과", name="dev.csv")

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", str(data), "--out", str(tmp_path / "run")) == 1
    error = capsys.readouterr().err
    assert NWA in error and "UNK" in error


def test_a_head_and_tokenizer_that_disagree_stop_the_run(
        tmp_path, monkeypatch, capsys):
    checkpoint = weightless_checkpoint(tmp_path, width=99)
    data = tmp_path / "data"
    manifest(data, "사과")
    manifest(data, "사과", name="dev.csv")

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", str(data), "--out", str(tmp_path / "run")) == 1
    assert "vocab_size" in capsys.readouterr().err


def test_a_missing_manifest_is_named(tmp_path, monkeypatch, capsys):
    checkpoint = weightless_checkpoint(tmp_path)

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", str(tmp_path / "nope"),
               "--out", str(tmp_path / "run")) == 1
    assert "train.csv" in capsys.readouterr().err


def test_the_output_may_not_live_inside_the_checkpoint(tmp_path):
    checkpoint = weightless_checkpoint(tmp_path)

    assert "안에 있습니다" in tr.check_out(
        checkpoint / "run", checkpoint, force=False)
    assert tr.check_out(tmp_path / "elsewhere", checkpoint,
                        force=False) is None


def test_a_fresh_run_may_not_contain_the_checkpoint_it_started_from(
        tmp_path):
    """`--init-model runs/child/epoch-08 --out runs/child`.

    A new run numbers epochs from one again, so it would write over the
    very checkpoint it started from. `--force` is about an existing log,
    not about permission to destroy the source.
    """
    run = tmp_path / "run"
    epoch = run / "epoch-08"
    epoch.mkdir(parents=True)

    for force in (False, True):
        assert "안에 있습니다" in tr.check_out(run, epoch, force=force)
    assert tr.check_out(run, run, force=True) is not None


def test_resuming_into_the_run_directory_stays_allowed(tmp_path):
    """The documented way to resume, and it must keep working."""
    run = tmp_path / "run"
    epoch = run / "epoch-03"
    epoch.mkdir(parents=True)
    (run / "history.json").write_text("[]", encoding="utf-8")

    assert tr.check_out(run, epoch, force=False, fresh=False) is None
    # The reverse is still refused, even for a resume.
    assert tr.check_out(epoch / "inner", epoch, force=False,
                        fresh=False) is not None
    assert tr.check_out(epoch, epoch, force=False, fresh=False) is not None


def test_a_relative_data_path_cannot_slip_past_the_fresh_data_guard(
        tmp_path, monkeypatch, capsys):
    checkpoint = weightless_checkpoint(tmp_path)
    monkeypatch.chdir(tr.DATA.parent)

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", "data", "--out", str(tmp_path / "run")) == 1
    assert "--data" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["train.csv", "dev.csv"])
def test_a_manifest_with_no_utterances_stops_the_run(
        tmp_path, monkeypatch, capsys, name):
    checkpoint = weightless_checkpoint(tmp_path)
    data = tmp_path / "data"
    manifest(data, "사과")
    manifest(data, "사과", name="dev.csv")
    manifest(data, name=name)          # header only

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", str(data), "--out", str(tmp_path / "run")) == 1
    assert "한 줄도 없습니다" in capsys.readouterr().err


def test_a_blank_transcript_stops_the_run(tmp_path, monkeypatch, capsys):
    checkpoint = weightless_checkpoint(tmp_path)
    data = tmp_path / "data"
    manifest(data, "사과", "   ")
    manifest(data, "사과", name="dev.csv")

    assert run(monkeypatch, "--init-model", str(checkpoint),
               "--data", str(data), "--out", str(tmp_path / "run")) == 1
    assert "빈 줄" in capsys.readouterr().err


@pytest.mark.parametrize("space", ["\t", "\u00a0"])
def test_only_a_plain_space_is_the_delimiter(tmp_path, space):
    """`_tokenize` replaces " " and nothing else.

    Treating every Unicode space as the delimiter would wave through a
    tab that the tokenizer then looks up as itself and cannot find.
    """
    tokenizer = tokenizer_for(tmp_path)

    assert tokenizer(f"사{space}과").input_ids.count(VOCAB["|"]) == 0
    assert tr.unknown({space}, tokenizer, 7) == [space]


class Idle:
    """Stands in for the model: scoring nothing never reaches it."""

    def eval(self):
        pass

    def train(self):
        pass


def test_scoring_nothing_is_an_error_not_a_perfect_result(tmp_path):
    """CER over zero reference characters used to come back 0.00%.

    Which reads as flawless, in the one case where the run is empty.
    """
    from transformers import Wav2Vec2Processor

    checkpoint = weightless_checkpoint(tmp_path)
    empty = manifest(tmp_path / "data", name="dev.csv")
    data = tr.Utterances(empty, Wav2Vec2Processor.from_pretrained(checkpoint))
    assert len(data) == 0

    with pytest.raises(ValueError, match="채점할 글자"):
        tr.character_error(Idle(), None, data, "cpu")
