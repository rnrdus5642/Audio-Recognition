"""Unit tests for the matching engine (confusion matrix + edit distance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.matcher import (
    Matcher,
    similarity_score,
    substring_edit_distance,
    weighted_edit_distance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v1.json"
)


@pytest.fixture(scope="module")
def matrix() -> ConfusionMatrix:
    return ConfusionMatrix.from_json(MATRIX_PATH)


class TestConfusionMatrixLoading:
    def test_loads_real_matrix(self, matrix: ConfusionMatrix) -> None:
        assert matrix.language == "ko"
        assert matrix.matrix_id  # has some id
        # Probe a few entries from the v1 matrix
        # s|t = 0.3 (혀짧은소리)
        assert matrix.sub_cost("s", "t") == pytest.approx(0.3)
        assert matrix.sub_cost("t", "s") == pytest.approx(0.3)  # commutative
        # Lax/tense/aspirated stops are cheap to swap
        assert matrix.sub_cost("k", "kʰ") == pytest.approx(0.2)
        # ㅔ/ㅐ near-merger
        assert matrix.sub_cost("e", "ɛ") == pytest.approx(0.1)

    def test_identity_is_free(self, matrix: ConfusionMatrix) -> None:
        assert matrix.sub_cost("s", "s") == 0.0
        assert matrix.sub_cost("a", "a") == 0.0

    def test_unknown_pair_uses_default(
        self, matrix: ConfusionMatrix
    ) -> None:
        # 'zz' is not in any rule -> default substitution
        assert matrix.sub_cost("zz", "qq") == matrix.default_substitution

    def test_coda_drop_is_cheap(self, matrix: ConfusionMatrix) -> None:
        # 종성 ㅇ 탈락 is very common in child speech
        assert matrix.del_cost("ŋ") == pytest.approx(0.3)
        # Unknown phoneme falls back to default
        assert matrix.del_cost("qq") == matrix.default_deletion


class TestWeightedEditDistance:
    def test_identical_zero_distance(self, matrix: ConfusionMatrix) -> None:
        assert weighted_edit_distance(["s", "a"], ["s", "a"], matrix) == 0.0

    def test_single_substitution(self, matrix: ConfusionMatrix) -> None:
        # /ㅅ/ -> /ㅌ/ has cost 0.3
        d = weighted_edit_distance(["t", "a"], ["s", "a"], matrix)
        assert d == pytest.approx(0.3)

    def test_unrelated_substitution_uses_default(
        self, matrix: ConfusionMatrix
    ) -> None:
        d = weighted_edit_distance(["zz"], ["qq"], matrix)
        assert d == pytest.approx(matrix.default_substitution)

    def test_coda_deletion_is_cheap(self, matrix: ConfusionMatrix) -> None:
        # User dropped final ŋ (very common)
        # target: [k, a, ŋ], user: [k, a]
        d = weighted_edit_distance(["k", "a"], ["k", "a", "ŋ"], matrix)
        assert d == pytest.approx(0.3)  # del_cost(ŋ) = 0.3

    def test_empty_user_full_target_cost(
        self, matrix: ConfusionMatrix
    ) -> None:
        # No speech detected: cost is sum of deletions
        target = ["s", "a"]
        d = weighted_edit_distance([], target, matrix)
        assert d == pytest.approx(
            matrix.del_cost("s") + matrix.del_cost("a")
        )


class TestSimilarityScore:
    def test_perfect_match(self) -> None:
        assert similarity_score(0.0, 5) == 1.0

    def test_full_distance_zero_score(self) -> None:
        assert similarity_score(5.0, 5) == 0.0

    def test_partial(self) -> None:
        # Half-distance -> half-score
        assert similarity_score(2.5, 5) == 0.5

    def test_clamps_below_zero(self) -> None:
        # If distance > target length, score is clamped to 0
        assert similarity_score(100.0, 3) == 0.0

    def test_zero_length_target_edge(self) -> None:
        # Both empty -> perfect, otherwise zero
        assert similarity_score(0.0, 0) == 1.0
        assert similarity_score(0.5, 0) == 0.0


class TestSubstringEditDistance:
    """Substring matching: target must appear inside user; surrounding
    noise is free."""

    def test_exact_match_zero_distance(
        self, matrix: ConfusionMatrix
    ) -> None:
        d, ws, we, ops = substring_edit_distance(
            ["s", "a", "k", "w", "a"], ["s", "a", "k", "w", "a"], matrix
        )
        assert d == 0.0
        assert (ws, we) == (0, 5)
        assert all(op == "match" for _, _, op in ops)

    def test_noise_prefix_ignored(
        self, matrix: ConfusionMatrix
    ) -> None:
        # User: noise [n, n] then 사과; target: 사과
        user = ["n", "n", "s", "a", "k", "w", "a"]
        target = ["s", "a", "k", "w", "a"]
        d, ws, we, _ = substring_edit_distance(user, target, matrix)
        assert d == 0.0
        assert (ws, we) == (2, 7)

    def test_noise_suffix_ignored(
        self, matrix: ConfusionMatrix
    ) -> None:
        # User: 사과 then noise; target: 사과
        user = ["s", "a", "k", "w", "a", "m", "m"]
        target = ["s", "a", "k", "w", "a"]
        d, ws, we, _ = substring_edit_distance(user, target, matrix)
        assert d == 0.0
        assert (ws, we) == (0, 5)

    def test_noise_both_sides(
        self, matrix: ConfusionMatrix
    ) -> None:
        user = ["n", "s", "a", "k", "w", "a", "m"]
        target = ["s", "a", "k", "w", "a"]
        d, ws, we, _ = substring_edit_distance(user, target, matrix)
        assert d == 0.0
        assert (ws, we) == (1, 6)

    def test_substitution_inside_window(
        self, matrix: ConfusionMatrix
    ) -> None:
        # Noise + 타과 (s->t) + noise
        user = ["n", "t", "a", "k", "w", "a", "n"]
        target = ["s", "a", "k", "w", "a"]
        d, ws, we, _ = substring_edit_distance(user, target, matrix)
        # Cost of s↔t = 0.3 from matrix; window should be 타과 portion
        assert d == pytest.approx(0.3)
        assert (ws, we) == (1, 6)

    def test_empty_user(self, matrix: ConfusionMatrix) -> None:
        user: list[str] = []
        target = ["s", "a"]
        d, ws, we, _ = substring_edit_distance(user, target, matrix)
        assert d == pytest.approx(
            matrix.del_cost("s") + matrix.del_cost("a")
        )
        assert (ws, we) == (0, 0)

    def test_empty_target(self, matrix: ConfusionMatrix) -> None:
        d, ws, we, ops = substring_edit_distance(
            ["a", "b"], [], matrix
        )
        assert d == 0.0
        assert ws == we
        assert ops == []


class TestMatcherSubstringMode:
    """Matcher in default substring mode should accept noisy ASR
    output (extra phonemes) without losing the match."""

    def test_noise_does_not_break_match(
        self, matrix: ConfusionMatrix
    ) -> None:
        candidates = [
            {
                "id": "apple",
                "text": "사과",
                "phonemes": ["s", "a", "k", "w", "a"],
                "threshold": 0.6,
            },
        ]
        matcher = Matcher(matrix)  # default mode = substring
        # User output: real word + breath noise
        user = ["s", "a", "k", "w", "a", "h", "n"]
        result = matcher.best_match(user, candidates)
        assert result.target_id == "apple"
        # Window should cover the real 사과 part
        assert (result.window_start, result.window_end) == (0, 5)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_exact_mode_still_available(
        self, matrix: ConfusionMatrix
    ) -> None:
        candidates = [
            {
                "id": "apple",
                "text": "사과",
                "phonemes": ["s", "a", "k", "w", "a"],
                "threshold": 0.6,
            },
        ]
        matcher = Matcher(matrix, mode="exact")
        # In exact mode, extra noise costs insertions
        user = ["s", "a", "k", "w", "a", "h", "n"]
        result = matcher.best_match(user, candidates)
        assert result.target_id == "apple"
        # Score should be lower than 1.0 because the noise counts
        assert result.score < 1.0


class TestMatcher:
    @pytest.fixture
    def candidates(self) -> list[dict]:
        # Mimics two answers in a single segment
        return [
            {
                "id": "apple",
                "text": "사과",
                "phonemes": ["s", "a", "k", "w", "a"],
                "threshold": 0.6,
            },
            {
                "id": "rabbit",
                "text": "토끼",
                "phonemes": ["tʰ", "o", "k͈", "i"],
                "threshold": 0.65,
            },
        ]

    def test_exact_match_passes(
        self,
        matrix: ConfusionMatrix,
        candidates: list[dict],
    ) -> None:
        matcher = Matcher(matrix)
        result = matcher.best_match(["s", "a", "k", "w", "a"], candidates)
        assert result.target_id == "apple"
        assert result.score == 1.0
        assert result.passed is True

    def test_child_pronunciation_still_passes(
        self,
        matrix: ConfusionMatrix,
        candidates: list[dict],
    ) -> None:
        # Child says 사과 as /타과/ (s -> t lisp)
        matcher = Matcher(matrix)
        result = matcher.best_match(["t", "a", "k", "w", "a"], candidates)
        assert result.target_id == "apple"
        # 1 substitution of cost 0.3 over length 5 -> score = 1 - 0.3/5 = 0.94
        assert result.score == pytest.approx(0.94, abs=0.01)
        assert result.passed is True

    def test_completely_different_word_fails(
        self,
        matrix: ConfusionMatrix,
        candidates: list[dict],
    ) -> None:
        # User says 나비, totally different from both apple and rabbit
        matcher = Matcher(matrix)
        result = matcher.best_match(["n", "a", "p", "i"], candidates)
        # Should fail both thresholds
        assert result.passed is False
        # Best match should be whichever is closer, but score should be low
        assert result.score < 0.6

    def test_empty_candidates(
        self, matrix: ConfusionMatrix
    ) -> None:
        matcher = Matcher(matrix)
        result = matcher.best_match(["a"], [])
        assert result.target_id is None
        assert result.passed is False
