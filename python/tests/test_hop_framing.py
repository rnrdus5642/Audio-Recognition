"""A frame is a hop of new samples, never a callback.

The guarantee is about identical normalised 16 kHz mono PCM. Input that
needs resampling first goes through a separate, unchanged stage.
"""

import numpy as np
import pytest

from python.runtime.audio import HopFramer

WINDOW = 40000
HOP = 8000


def ramp(count, offset=0):
    """Samples that differ by absolute position, so a window taken at the
    wrong moment cannot compare equal to the right one."""
    return ((np.arange(count, dtype=np.float32) + offset + 1)
            / 100000.0).astype(np.float32)


def feed_in_chunks(audio, sizes, window=WINDOW, hop=HOP):
    framer = HopFramer(window, hop)
    frames, offset, i = [], 0, 0
    while offset < len(audio):
        size = min(sizes[i % len(sizes)], len(audio) - offset)
        frames.extend(framer.feed(audio[offset:offset + size]))
        offset += size
        i += 1
    return frames


def assert_same_frames(expected, actual):
    assert [f.end_sample for f in actual] == [f.end_sample for f in expected]
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a.samples, b.samples)


@pytest.mark.parametrize("sizes", [
    [16000], [8000], [4000], [1], [1, 7999, 3, 5000, 4001], [12000, 3999, 1],
])
def test_split_and_merged_input_produce_the_same_frames(sizes):
    audio = ramp(16000)
    reference = feed_in_chunks(audio, [16000])
    assert [f.end_sample for f in reference] == [8000, 16000]
    assert_same_frames(reference, feed_in_chunks(audio, sizes))


def test_a_chunk_longer_than_the_window_keeps_every_frame():
    audio = ramp(3 * 16000)
    one_chunk = feed_in_chunks(audio, [len(audio)])
    assert [f.end_sample for f in one_chunk] == [
        8000, 16000, 24000, 32000, 40000, 48000]
    assert_same_frames(one_chunk, feed_in_chunks(audio, [HOP]))


def test_a_partial_hop_is_not_a_frame():
    framer = HopFramer(WINDOW, HOP)
    assert framer.feed(ramp(7999)) == []
    assert framer.total_samples == 7999

    frames = framer.feed(ramp(1, 7999))

    assert len(frames) == 1
    assert frames[0].end_sample == 8000
    assert frames[0].end_seconds() == 0.5
    np.testing.assert_array_equal(frames[0].samples[:32000], 0)
    np.testing.assert_array_equal(frames[0].samples[32000:], ramp(8000))


def test_a_window_is_the_trailing_audio_at_its_own_boundary():
    audio = ramp(30000)
    for frame in feed_in_chunks(audio, [1000]):
        expected = audio[max(0, frame.end_sample - WINDOW):frame.end_sample]
        assert len(frame.samples) == WINDOW
        np.testing.assert_array_equal(frame.samples[-len(expected):], expected)
        np.testing.assert_array_equal(frame.samples[:WINDOW - len(expected)], 0)


def test_one_big_chunk_yields_every_window_in_order():
    """8000 then 24000 must still be four windows, oldest first.

    Each one holds the audio as it stood at its own boundary. Handing
    back the newest window four times would let a merged chunk confirm on
    material a split one never scored.
    """
    audio = ramp(4 * HOP)
    framer = HopFramer(WINDOW, HOP)
    frames = framer.feed(audio[:HOP]) + framer.feed(audio[HOP:])

    assert [f.end_sample for f in frames] == [8000, 16000, 24000, 32000]
    for frame in frames:
        expected = audio[max(0, frame.end_sample - WINDOW):frame.end_sample]
        np.testing.assert_array_equal(frame.samples[-len(expected):], expected)
        np.testing.assert_array_equal(frame.samples[:WINDOW - len(expected)], 0)


def test_chunks_that_straddle_quarter_hops_frame_the_same_way():
    # A hop no chunk size lines up with, which is the normal case once a
    # device picks its own buffer length.
    audio = ramp(12000)
    reference = feed_in_chunks(audio, [12000], window=16000, hop=4000)
    assert [f.end_sample for f in reference] == [4000, 8000, 12000]
    assert_same_frames(
        reference, feed_in_chunks(audio, [1500], window=16000, hop=4000))
    assert_same_frames(
        reference,
        feed_in_chunks(audio, [1500, 2500, 500, 7500], window=16000, hop=4000))


@pytest.mark.parametrize("empty", [np.zeros(0, dtype=np.float32), []])
def test_empty_input_makes_no_frames(empty):
    framer = HopFramer(WINDOW, HOP)
    for _ in range(5):
        assert framer.feed(empty) == []
    assert framer.total_samples == 0


def test_identical_samples_are_still_new_audio():
    # Silence is audio. Nothing may deduplicate by value.
    framer = HopFramer(WINDOW, HOP)
    silence = np.zeros(HOP, dtype=np.float32)
    ends = [f.end_sample for _ in range(4) for f in framer.feed(silence)]
    assert ends == [8000, 16000, 24000, 32000]


def test_reset_forgets_the_partial_hop_as_well_as_the_audio():
    framer = HopFramer(WINDOW, HOP)
    framer.feed(ramp(4000))
    framer.reset()

    assert framer.feed(ramp(4000)) == [], "boundary phase survived a reset"
    frames = framer.feed(ramp(4000, 4000))

    assert [f.end_sample for f in frames] == [8000]
    np.testing.assert_array_equal(framer.snapshot()[-8000:], ramp(8000))


def test_a_non_default_hop_frames_on_its_own_boundaries():
    audio = ramp(16000)
    reference = feed_in_chunks(audio, [16000], window=16000, hop=4000)
    assert [f.end_sample for f in reference] == [4000, 8000, 12000, 16000]
    assert len(reference[0].samples) == 16000
    assert_same_frames(
        reference, feed_in_chunks(audio, [999, 1], window=16000, hop=4000))


def test_the_input_array_is_left_alone():
    framer = HopFramer(WINDOW, HOP)
    chunk = ramp(HOP)
    original = chunk.copy()
    framer.feed(chunk)
    np.testing.assert_array_equal(chunk, original)
    assert framer.snapshot().dtype == np.float32


@pytest.mark.parametrize("window,hop", [(0, HOP), (WINDOW, 0), (-1, HOP)])
def test_impossible_configurations_are_rejected(window, hop):
    with pytest.raises(ValueError):
        HopFramer(window, hop)
