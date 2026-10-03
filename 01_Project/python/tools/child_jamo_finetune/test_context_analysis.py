import pytest

from context_analysis import cluster_interval


def row(key, speaker, errors, reference):
    return {'id': key, 'speaker': speaker, 'variants': {
        name: {'errors': errors, 'reference_jamo': reference} for name in ('a', 'b')}}


def test_effect_size_and_interval_keep_repeated_variants_with_their_speaker():
    parent = [row('a1', 'A', 4, 10), row('a2', 'A', 4, 10), row('b1', 'B', 0, 40)]
    candidate = [row('a1', 'A', 2, 10), row('a2', 'A', 2, 10), row('b1', 'B', 0, 40)]
    result = cluster_interval(parent, candidate, ['a', 'b'])
    assert result['speakers'] == 2
    assert result['candidate_minus_comparator_jer'] == pytest.approx(-4 / 60)
    assert result['speaker_cluster_bootstrap_95ci'] == pytest.approx([-.2, 0.])


def test_identical_models_have_zero_difference_and_zero_interval():
    rows = [row('a', 'A', 3, 9), row('b', 'B', 0, 21)]
    result = cluster_interval(rows, rows, ['a', 'b'])
    assert result['candidate_minus_comparator_jer'] == 0.
    assert result['speaker_cluster_bootstrap_95ci'] == [0., 0.]
