"""Which utterances survive selection, and what the counts mean.

All fixtures here are synthetic. The corpus itself is private and none of
it is read: what is under test is the arithmetic that decides what gets
trained on, which is exactly the part that quietly lost a rare syllable.
"""

import csv
import json
import unicodedata

import pytest

from python.tools.aihub import config
from python.tools.child_finetune import manifest as mf

NWA = "놔"


def row(speaker, text, seconds=3.0, index=0):
    return {"wav": f"{speaker}-{index}.wav", "text": text,
            "seconds": f"{seconds:.3f}", "start": "", "end": "",
            "speaker": speaker, "age": "6", "style": "read"}


def speaker_rows(speaker, count, rare_at=None, seconds=3.0):
    """`count` utterances, optionally with the rare syllable at one index."""
    return [row(speaker, NWA + "요" if i == rare_at else "사과",
                seconds, i)
            for i in range(count)]


def select(pool, hours, keep=None, repeat=1, drop=()):
    """choose, drop unreadable files, then count - the order main uses."""
    rows, cap, kept = mf.choose(pool, hours, keep, repeat)
    rows = [r for r in rows if r["wav"] not in drop]
    return rows, mf.tally(rows, keep, cap, kept)


def test_the_cap_is_the_largest_one_that_fits_the_budget():
    pool = {"a": speaker_rows("a", 10), "b": speaker_rows("b", 10)}

    # 20 utterances of 3s is 60s; a 30s budget buys 5 each.
    assert mf.cap_for(pool, 30 / 3600) == 5
    # Once everything fits there is nothing to cap, and the sentinel says
    # so rather than naming the longest speaker.
    assert mf.cap_for(pool, 60 / 3600) == 10_000
    assert mf.cap_for(pool, 1) == 10_000


def test_the_cap_loses_a_rare_syllable_that_retention_gets_back():
    """The regression this whole option exists for.

    The cap keeps a speaker's first utterances. A syllable that turns up
    a few hundred times in four hundred thousand rows is almost never in
    that slice, so widening the vocabulary on its own would change the
    manifest by nothing at all.
    """
    pool = {"a": speaker_rows("a", 10, rare_at=9)}

    lost, _ = select(pool, 15 / 3600)
    kept, stats = select(pool, 15 / 3600, keep={NWA})

    assert len(lost) == 5 and not any(NWA in r["text"] for r in lost)
    assert len(kept) == 6 and sum(NWA in r["text"] for r in kept) == 1
    assert stats["retained_past_cap"] == 1
    assert stats["carrying"] == 1
    assert stats["speakers_carrying"] == 1


def test_a_rare_row_inside_the_cap_is_not_counted_twice():
    pool = {"a": speaker_rows("a", 10, rare_at=0)}

    kept, stats = select(pool, 15 / 3600, keep={NWA})

    assert len(kept) == 5
    assert stats["retained_past_cap"] == 0
    assert stats["carrying"] == 1


def test_repeating_multiplies_only_the_wanted_rows():
    pool = {"a": speaker_rows("a", 4, rare_at=3)}

    rows, stats = select(pool, 1, keep={NWA}, repeat=4)

    assert len(rows) == 7, "3 ordinary + 1 rare seen four times"
    assert sum(NWA in r["text"] for r in rows) == 4
    assert stats["unique"] == 4 and stats["extra_copies"] == 3


def test_repeated_audio_is_never_reported_as_more_audio():
    """An oversampled hour is not an hour of speech."""
    pool = {"a": speaker_rows("a", 4, rare_at=3)}

    _, stats = select(pool, 1, keep={NWA}, repeat=4)

    assert stats["unique_seconds"] == pytest.approx(12.0)
    assert stats["total_seconds"] == pytest.approx(21.0)


@pytest.mark.parametrize("repeat", [0, 1])
def test_not_repeating_changes_nothing(repeat):
    pool = {"a": speaker_rows("a", 4, rare_at=3)}

    rows, stats = select(pool, 1, keep={NWA}, repeat=repeat)

    assert len(rows) == 4 and stats["extra_copies"] == 0
    assert stats["unique_seconds"] == stats["total_seconds"]


def test_selection_repeats_exactly():
    pool = {"a": speaker_rows("a", 10, rare_at=9),
            "b": speaker_rows("b", 10, rare_at=8)}

    first, one = select(pool, 15 / 3600, keep={NWA}, repeat=3)
    again, two = select(pool, 15 / 3600, keep={NWA}, repeat=3)

    assert [r["wav"] for r in first] == [r["wav"] for r in again]
    assert one == two


def test_a_speaker_is_only_ever_in_one_pool():
    rows = [r for s in (f"spk{i:03}" for i in range(60))
            for r in speaker_rows(s, 3)]

    train_pool, dev_pool = mf.pools(rows, 0.055)

    assert set(train_pool) & set(dev_pool) == set(), "a speaker leaked"
    assert sum(len(v) for v in train_pool.values()) \
        + sum(len(v) for v in dev_pool.values()) == len(rows)
    assert dev_pool, "0.055 of 60 speakers should hold some out"
    assert mf.pools(rows, 0.055)[1].keys() == dev_pool.keys()


def test_retention_never_moves_a_held_out_speaker_into_training():
    rows = [r for s in (f"spk{i:03}" for i in range(60))
            for r in speaker_rows(s, 10, rare_at=9)]
    train_pool, dev_pool = mf.pools(rows, 0.055)

    train_rows, _ = select(train_pool, 15 / 3600, keep={NWA}, repeat=3)
    dev_rows, dev_stats = select(dev_pool, 15 / 3600, keep={NWA})

    held_out = set(dev_pool)
    assert not {r["speaker"] for r in train_rows} & held_out
    assert {r["speaker"] for r in dev_rows} <= held_out
    assert dev_stats["extra_copies"] == 0, "dev must never be oversampled"


@pytest.mark.parametrize("text,expected", [
    ("사과 먹어요", "사과 먹어요"),
    ("사과, 배!", "사과 배"),
    ("사과(2)", "사과"),
    ("a1?!", ""),
    ("  사과   먹어요 ", "사과 먹어요"),
])
def test_normalise_keeps_hangul_and_spacing_only(text, expected):
    assert mf.normalise(text) == expected


def test_a_decomposed_transcript_is_not_deleted_by_the_filter():
    """Jamo are not in the 가-힣 range, so order matters.

    Filtering first turns a fully decomposed transcript into nothing, and
    a partly decomposed one into a shorter word than was said - which
    then becomes the CTC target.
    """
    fully = unicodedata.normalize("NFD", "놔요")
    partly = "엄마 " + unicodedata.normalize("NFD", NWA)

    assert mf.normalise(fully) == "놔요"
    assert mf.normalise(partly) == "엄마 놔"


def test_a_decomposed_transcript_survives_the_vocabulary_filter(
        tmp_path, monkeypatch):
    index_fixture(tmp_path, monkeypatch,
                  [entry("nfd", unicodedata.normalize("NFD", "사과"))])

    rows, dropped = mf.load(mf.vocabulary())

    assert dropped["empty"] == 0 and dropped["vocab"] == 0
    assert rows[0]["text"] == "사과"


def test_the_report_counts_only_what_reaches_the_csv():
    """A rare utterance whose wav will not open is not evidence.

    Counting at selection time claimed it anyway, because the readability
    filter runs afterwards.
    """
    pool = {"a": speaker_rows("a", 10, rare_at=9),
            "b": speaker_rows("b", 10, rare_at=9)}

    rows, stats = select(pool, 15 / 3600, keep={NWA}, repeat=3,
                         drop={"a-9.wav"})

    assert sum(NWA in r["text"] for r in rows) == 3, "only b's, three times"
    assert stats["carrying"] == 1
    assert stats["retained_past_cap"] == 1
    assert stats["speakers_carrying"] == 1
    assert stats["unique"] == len({id(r) for r in rows})
    assert stats["extra_copies"] == 2
    assert stats["total_seconds"] == pytest.approx(
        sum(float(r["seconds"]) for r in rows))


def test_dropping_every_rare_row_reports_no_evidence_at_all():
    pool = {"a": speaker_rows("a", 10, rare_at=9)}

    rows, stats = select(pool, 15 / 3600, keep={NWA}, repeat=4,
                         drop={"a-9.wav"})

    assert not any(NWA in r["text"] for r in rows)
    assert stats["carrying"] == 0 and stats["extra_copies"] == 0
    assert stats["retained_past_cap"] == 0
    assert stats["unique_seconds"] == stats["total_seconds"]


def index_fixture(tmp_path, monkeypatch, rows):
    """A private-corpus-shaped index with nothing private in it."""
    audio = tmp_path / "audio"
    splits = {e["speaker"]: {"split": e.get("_split", "train")}
              for e in rows}
    fields = ["speaker", "text", "seconds", "speech_start", "speech_end",
              "age", "style", "wav"]
    with open(tmp_path / "index.csv", "w", encoding="utf-8",
              newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in rows:
            exists = entry.pop("_exists", True)
            entry.pop("_split", None)
            writer.writerow(entry)
            if exists:
                folder = audio / f"{int(entry['age']):02d}세" / entry["style"]
                folder.mkdir(parents=True, exist_ok=True)
                (folder / entry["wav"]).write_bytes(b"")

    (tmp_path / "splits.json").write_text(
        json.dumps({"speakers": splits}), encoding="utf-8")

    monkeypatch.setattr(config, "INDEX", str(tmp_path / "index.csv"))
    monkeypatch.setattr(config, "SPLITS", str(tmp_path / "splits.json"))
    monkeypatch.setattr(config, "AUDIO", str(audio))


def entry(speaker, text, seconds=3.0, exists=True, split="train"):
    return {"speaker": speaker, "text": text, "seconds": f"{seconds}",
            "speech_start": "", "speech_end": "", "age": "6",
            "style": "read", "wav": f"{speaker}.wav",
            "_exists": exists, "_split": split}


def test_load_says_why_each_utterance_was_left_out(tmp_path, monkeypatch):
    index_fixture(tmp_path, monkeypatch, [
        entry("keep", "사과"),
        entry("oov", NWA + "요"),
        entry("short", "사과", seconds=0.5),
        entry("long", "사과", seconds=20.0),
        entry("empty", "123"),
        entry("gone", "사과", exists=False),
        entry("other", "사과", split="valid"),
    ])

    rows, dropped = mf.load(mf.vocabulary())

    assert [r["speaker"] for r in rows] == ["keep"]
    assert dropped == {"vocab": 1, "short": 1, "long": 1, "empty": 1,
                       "missing": 1}


def test_a_wider_vocabulary_is_what_stops_the_row_being_dropped(
        tmp_path, monkeypatch):
    index_fixture(tmp_path, monkeypatch, [entry("oov", NWA + "요")])

    assert mf.load(mf.vocabulary())[1]["vocab"] == 1

    widened = mf.vocabulary() | {NWA}
    rows, dropped = mf.load(widened)
    assert dropped["vocab"] == 0 and len(rows) == 1


def test_an_existing_manifest_is_not_overwritten_by_accident(tmp_path):
    (tmp_path / "train.csv").write_text("x", encoding="utf-8")

    assert "train.csv" in mf.check_out(tmp_path, force=False)
    assert mf.check_out(tmp_path, force=True) is None
    assert mf.check_out(tmp_path / "fresh", force=False) is None
