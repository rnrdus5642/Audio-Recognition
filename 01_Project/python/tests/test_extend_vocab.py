"""Widening a CTC head without moving anything already in it.

Everything here runs on a deliberately tiny wav2vec2 - two layers, 32
hidden - because the properties under test are about ids and rows, not
about accuracy, and a real checkpoint is 1.27GB. The same code path
produces the real candidate.
"""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from python.tools import vocab_coverage as vc
from python.tools.child_finetune import extend_vocab as ev

NWA = "놔"
YO = "요"

# Ids deliberately unsorted and with the placeholders last, the way a
# real wav2vec2 vocabulary arrives.
VOCAB = {"가": 0, YO: 1, "|": 2, "나": 3, "[UNK]": 4, "[PAD]": 5}


def tiny_checkpoint(tmp_path, vocab=None, name="source"):
    """A complete, loadable checkpoint small enough to build per test."""
    from transformers import (Wav2Vec2Config, Wav2Vec2CTCTokenizer,
                              Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                              Wav2Vec2Processor)

    vocab = VOCAB if vocab is None else vocab
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    tokenizer = Wav2Vec2CTCTokenizer(
        str(directory / "vocab.json"), unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|", do_lower_case=False)
    extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True)
    Wav2Vec2Processor(feature_extractor=extractor,
                      tokenizer=tokenizer).save_pretrained(directory)

    config = Wav2Vec2Config(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64,
        conv_dim=(32, 32, 32), conv_stride=(4, 4, 2), conv_kernel=(8, 4, 4),
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
        pad_token_id=vocab["[PAD]"], bos_token_id=1, eos_token_id=2,
        ctc_loss_reduction="mean", ctc_zero_infinity=True)
    torch.manual_seed(0)
    Wav2Vec2ForCTC(config).save_pretrained(directory)
    return directory


def reload(directory):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    return (Wav2Vec2ForCTC.from_pretrained(directory),
            Wav2Vec2Processor.from_pretrained(directory))


def test_the_new_syllable_takes_the_old_width_and_markers_move_above_it(
        tmp_path):
    source = tiny_checkpoint(tmp_path)

    report = ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)

    _, processor = reload(tmp_path / "out")
    live = processor.tokenizer.get_vocab()
    assert report["added"] == {NWA: 6}
    assert {t: live[t] for t in VOCAB} == VOCAB, "an old id moved"
    assert live[NWA] == 6
    assert processor.tokenizer.vocab_size == 7
    assert live["<s>"] == 7 and live["</s>"] == 8
    assert len(live) == 9, "markers must sit above the acoustic width"


def test_the_placeholders_the_decoder_relies_on_do_not_move(tmp_path):
    source = tiny_checkpoint(tmp_path)

    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)

    _, processor = reload(tmp_path / "out")
    tokenizer = processor.tokenizer
    assert tokenizer.pad_token_id == VOCAB["[PAD]"]
    assert tokenizer.unk_token_id == VOCAB["[UNK]"]
    assert tokenizer.word_delimiter_token_id == VOCAB["|"]


def test_add_tokens_is_the_thing_this_exists_to_avoid(tmp_path):
    """Kept as a test so the rationale cannot quietly stop being true.

    `add_tokens` numbers from `len(tokenizer)`, which counts the two
    markers, so the syllable lands two past the head and resizing to the
    tokenizer length would hand the model two random marker rows.
    """
    from transformers import Wav2Vec2Processor

    source = tiny_checkpoint(tmp_path)
    tokenizer = Wav2Vec2Processor.from_pretrained(source).tokenizer

    tokenizer.add_tokens(NWA)

    assert len(VOCAB) == 6 and tokenizer.vocab_size == 6
    assert tokenizer.get_vocab()[NWA] == 8, "not the next acoustic column"
    assert len(tokenizer) == 9


def test_every_other_weight_survives_bit_for_bit(tmp_path):
    source = tiny_checkpoint(tmp_path)
    before = {k: v.clone() for k, v in reload(source)[0].state_dict().items()}

    report = ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)

    after = reload(tmp_path / "out")[0].state_dict()
    assert set(before) == set(after)
    for key, tensor in before.items():
        other = after[key][:len(VOCAB)] if key.startswith("lm_head.") \
            else after[key]
        assert torch.equal(tensor, other), key
    assert report["weights"]["rows_carried_over"] == 6
    assert report["weights"]["head_shape"] == [7, 32]


def test_the_new_row_starts_inert_and_says_so(tmp_path):
    source = tiny_checkpoint(tmp_path)

    report = ev.extend(source, tmp_path / "out", [NWA], verify_clips=0,
                       bias=-12.5)

    model = reload(tmp_path / "out")[0]
    assert torch.count_nonzero(model.lm_head.weight[6]) == 0
    assert float(model.lm_head.bias[6].detach()) == pytest.approx(-12.5)
    assert report["new_row_init"] == {"mode": "inert", "bias": -12.5,
                                      "seed": 0}


def test_a_seeded_head_init_repeats_exactly(tmp_path):
    source = tiny_checkpoint(tmp_path)

    ev.extend(source, tmp_path / "a", [NWA], init="hf", seed=7,
              verify_clips=0)
    ev.extend(source, tmp_path / "b", [NWA], init="hf", seed=7,
              verify_clips=0)
    ev.extend(source, tmp_path / "c", [NWA], init="hf", seed=8,
              verify_clips=0)

    rows = [reload(tmp_path / n)[0].lm_head.weight[6] for n in "abc"]
    assert torch.equal(rows[0], rows[1])
    assert not torch.equal(rows[0], rows[2])
    assert torch.count_nonzero(rows[0]) > 0


def test_the_new_syllable_survives_a_round_trip(tmp_path):
    source = tiny_checkpoint(tmp_path)
    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)
    tokenizer = reload(tmp_path / "out")[1].tokenizer

    assert tokenizer(NWA + YO).input_ids == [6, 1]
    assert tokenizer.decode([6, 1]) == NWA + YO


def test_the_new_token_collapses_like_every_other_one(tmp_path):
    """CTC rules do not get a special case for a token added later.

    A repeat collapses, and a blank between two of them does not - which
    is the only way the model can ever say the syllable twice.
    """
    source = tiny_checkpoint(tmp_path)
    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)
    tokenizer = reload(tmp_path / "out")[1].tokenizer
    blank = tokenizer.pad_token_id

    assert tokenizer.decode([6, 6, 6]) == NWA
    assert tokenizer.decode([6, blank, 6]) == NWA + NWA
    assert tokenizer.decode([blank, 6, 1, blank]) == NWA + YO


def test_the_new_row_can_actually_be_trained(tmp_path):
    """An inert row is useless if it is also a dead one.

    Zero weights and a low bias make the new class improbable, which is
    the point, but improbable enough and the softmax gradient underflows
    and the row never moves. This is the check that it still moves.
    """
    source = tiny_checkpoint(tmp_path)
    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)
    model = reload(tmp_path / "out")[0]
    model.train()

    torch.manual_seed(0)
    result = model(torch.randn(1, 4000),
                   labels=torch.tensor([[6, 1]]))

    assert torch.isfinite(result.loss), "CTC loss on the new token diverged"
    result.loss.backward()
    assert torch.count_nonzero(model.lm_head.weight.grad[6]) > 0
    assert model.lm_head.bias.grad[6] != 0


def test_asking_again_for_a_syllable_it_already_has_does_nothing(tmp_path):
    source = tiny_checkpoint(tmp_path)
    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)

    again = ev.extend(tmp_path / "out", tmp_path / "twice", [NWA],
                      verify_clips=0)

    assert again["status"] == "NOTHING_TO_ADD"
    assert again["added"] == {}
    assert not (tmp_path / "twice").exists(), "wrote a checkpoint anyway"


def test_two_syllables_are_numbered_in_a_fixed_order(tmp_path):
    source = tiny_checkpoint(tmp_path)

    one = ev.extend(source, tmp_path / "one", [NWA, "돠"], verify_clips=0)
    other = ev.extend(source, tmp_path / "other", ["돠", NWA, "돠"],
                      verify_clips=0)

    # Code point order, so the ids do not depend on how they were typed.
    assert one["added"] == other["added"] == {NWA: 6, "돠": 7}


@pytest.mark.parametrize("out", ["source", "source/inner", ""])
def test_the_source_checkpoint_is_never_written_into(tmp_path, out):
    source = tiny_checkpoint(tmp_path)
    destination = (source / "inner") if out == "source/inner" else (
        source if out == "source" else tmp_path)

    with pytest.raises(ev.ExtensionError):
        ev.extend(source, destination, [NWA], verify_clips=0)


def test_a_destination_with_anything_in_it_is_refused(tmp_path):
    source = tiny_checkpoint(tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "note.txt").write_text("x", encoding="utf-8")

    with pytest.raises(ev.ExtensionError, match="비어 있지 않"):
        ev.extend(source, busy, [NWA], verify_clips=0)


@pytest.mark.parametrize("bad", ["놔요", "a", "", "ㄱ"])
def test_only_single_hangul_syllables_may_be_added(tmp_path, bad):
    source = tiny_checkpoint(tmp_path)

    with pytest.raises(ev.ExtensionError, match="한글 음절"):
        ev.extend(source, tmp_path / "out", [bad], verify_clips=0)
    assert not (tmp_path / "out").exists()


def test_the_config_stops_naming_ids_that_were_never_the_markers(tmp_path):
    """The source names 1 and 2, which are Hangul syllables.

    Vestigial values that predate the checkpoint and never matched the
    tokenizer. The replacement is None rather than the tokenizer's marker
    ids: the field describes the model's own vocabulary, and a CTC head
    has no column for a sentence marker - transformers itself warns when
    the id sits past the width.
    """
    source = tiny_checkpoint(tmp_path)

    report = ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)

    model, processor = reload(tmp_path / "out")
    assert (report["config_token_ids"]["bos_token_id_before"],
            report["config_token_ids"]["eos_token_id_before"]) == (1, 2)
    assert model.config.bos_token_id is None
    assert model.config.eos_token_id is None
    # Not lost, just recorded where it is true.
    assert processor.tokenizer.bos_token_id == 7
    assert report["config_token_ids"]["tokenizer_bos_id"] == 7
    assert report["config_token_ids"]["tokenizer_eos_id"] == 8
    assert model.config.pad_token_id == VOCAB["[PAD]"], "blank moved"


def test_the_old_config_ids_can_be_left_alone(tmp_path):
    source = tiny_checkpoint(tmp_path)

    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0,
              keep_config_token_ids=True)

    assert reload(tmp_path / "out")[0].config.bos_token_id == 1


@pytest.mark.parametrize("bias", [float("nan"), float("inf"),
                                  float("-inf")])
def test_a_bias_that_is_not_a_number_is_refused_before_anything_is_written(
        tmp_path, bias):
    """NaN would reach every logit through the softmax.

    It also cannot be written as JSON anything else will read back, so
    the provenance beside it would be unparseable. Caught before the
    output directory exists.
    """
    source = tiny_checkpoint(tmp_path)

    with pytest.raises(ev.ExtensionError, match="유한"):
        ev.extend(source, tmp_path / "out", [NWA], verify_clips=0,
                  bias=bias)

    assert not (tmp_path / "out").exists()
    assert list(tmp_path.glob("*staging*")) == []


def test_the_provenance_is_json_anything_can_read(tmp_path):
    source = tiny_checkpoint(tmp_path)

    ev.extend(source, tmp_path / "out", [NWA], verify_clips=0)

    written = (tmp_path / "out" / "vocab_extension.json").read_text(
        encoding="utf-8")
    assert "NaN" not in written and "Infinity" not in written
    assert json.loads(written)["status"] == "UNTRAINED"


def test_a_failure_after_saving_leaves_no_checkpoint_behind(
        tmp_path, monkeypatch):
    """The output path is either a checked candidate or nothing at all.

    Saving straight to it would leave something that loads, has no
    provenance saying it was never validated, and blocks the retry by
    being in the way.
    """
    source = tiny_checkpoint(tmp_path)
    before = {k: v.clone() for k, v in reload(source)[0].state_dict().items()}
    out = tmp_path / "out"

    def explode(*_args, **_kwargs):
        raise ev.ExtensionError("검증 실패 흉내")

    monkeypatch.setattr(ev, "check_weights", explode)
    with pytest.raises(ev.ExtensionError, match="검증 실패"):
        ev.extend(source, out, [NWA], verify_clips=0)

    assert not out.exists()
    assert list(tmp_path.glob("*staging*")) == []
    after = reload(source)[0].state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), \
        "the source checkpoint was modified"
    assert vc.load_checkpoint(source).size == len(VOCAB)

    monkeypatch.undo()
    assert ev.extend(source, out, [NWA], verify_clips=0)["added"] == {NWA: 6}
    assert (out / "vocab_extension.json").exists()


def test_a_rename_that_fails_does_not_strand_the_staging_copy(
        tmp_path, monkeypatch):
    """Publishing is inside the cleanup boundary, not after it.

    A checkpoint is 1.27GB. A locked destination that left one staging
    copy behind per attempt would fill the disk in a handful of retries.
    """
    source = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"

    def refuse(*_args, **_kwargs):
        raise PermissionError("이름 바꾸기 실패 흉내")

    monkeypatch.setattr(ev.os, "replace", refuse)
    with pytest.raises(PermissionError):
        ev.extend(source, out, [NWA], verify_clips=0)

    assert not out.exists()
    assert list(tmp_path.glob("*staging*")) == []


def test_a_destination_someone_else_filled_is_left_alone(
        tmp_path, monkeypatch):
    """Cleanup deletes the staging directory, never the final path."""
    source = tiny_checkpoint(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    def occupy(*_args, **_kwargs):
        (out / "someone-elses.txt").write_text("x", encoding="utf-8")
        return {"status": "ok"}

    monkeypatch.setattr(ev, "_build_candidate", occupy)
    with pytest.raises(OSError):
        ev.extend(source, out, [NWA], verify_clips=0)

    assert (out / "someone-elses.txt").exists()
    assert list(tmp_path.glob("*staging*")) == []


def test_what_the_low_bias_buys_is_measured_not_assumed(tmp_path):
    """The inert row keeps out of the argmax here. That is an observation.

    A constant bias cannot guarantee it: any frame whose trained logits
    all sit below that constant hands the argmax straight to the new row.
    Raising the bias shows exactly that happening, which is why the tool
    reports a count from real clips instead of asserting a property.
    """
    if not ev.reference_clips([NWA], 2):
        pytest.skip("골든 오디오 없음 (gitignored)")

    source = tiny_checkpoint(tmp_path)

    quiet = ev.extend(source, tmp_path / "quiet", [NWA], verify_clips=2)
    loud = ev.extend(source, tmp_path / "loud", [NWA], verify_clips=2,
                     bias=50.0)

    assert quiet["reference_audio"]["clips"] == 2
    assert quiet["reference_audio"]["new_token_frames"] == 0
    assert quiet["reference_audio"]["transcripts_changed"] == 0
    # Not zero, and not expected to be. The weights are bit-identical
    # (see above) but a wider output means a differently blocked GEMM,
    # and the old columns come back a few units in the last place apart.
    # Measured at 6e-08 on these clips; the transcript is what has to
    # match exactly, not the float.
    assert quiet["reference_audio"]["max_old_column_diff"] < 1e-5
    assert loud["reference_audio"]["new_token_frames"] > 0, \
        "a high enough bias must win frames, or this proves nothing"
