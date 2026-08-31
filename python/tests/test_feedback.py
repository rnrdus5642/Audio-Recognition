"""Unit tests for per-syllable feedback on a passing utterance."""

from __future__ import annotations

from pathlib import Path

import pytest

from python.build.build_targets import syllable_spans
from python.runtime.matching.confusion_matrix import ConfusionMatrix
from python.runtime.matching.feedback import (
    ModelErrors,
    explain,
)
from python.runtime.matching.matcher import Matcher


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = (
    PROJECT_ROOT / "shared" / "confusion_matrices" / "ko_child_v2.json"
)
ERRORS_PATH = PROJECT_ROOT / "shared" / "reference" / "reference_errors.json"

# 사과 = [s,a,k,w,a], and ㅘ is one character worth two phonemes.
SAGWA = [
    {"text": "사", "spoken": "사", "span": [0, 2]},
    {"text": "과", "spoken": "과", "span": [2, 5]},
]
SAGWA_PHONEMES = ["s", "a", "k", "w", "a"]


@pytest.fixture(scope="module")
def matrix() -> ConfusionMatrix:
    return ConfusionMatrix.from_json(MATRIX_PATH)


@pytest.fixture(scope="module")
def errors() -> ModelErrors:
    return ModelErrors.from_json(ERRORS_PATH)


class TestAttribution:
    def test_clean_utterance_says_nothing(self, matrix: ConfusionMatrix) -> None:
        alignment = [(p, p, "match") for p in SAGWA_PHONEMES]

        reports = explain(alignment, SAGWA, matrix)

        assert [r.status for r in reports] == ["clean", "clean"]
        assert all(not r.slips for r in reports)

    def test_substitution_lands_in_its_own_syllable(
        self, matrix: ConfusionMatrix
    ) -> None:
        # 사과 -> 타과: /s/ heard as /t/, the classic 혀짧은소리
        alignment = [
            ("t", "s", "sub"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert reports[0].text == "사"
        assert len(reports[0].slips) == 1
        assert reports[0].slips[0].target == "s"
        assert reports[0].slips[0].heard == "t"
        assert reports[1].status == "clean"

    def test_slip_in_the_second_half_of_a_diphthong(
        self, matrix: ConfusionMatrix
    ) -> None:
        # The glide of ㅘ is phoneme 3 of 5 - a naive halfway split would
        # file it under 사.
        alignment = [
            ("s", "s", "match"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("", "w", "del"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert not reports[0].slips
        assert [s.target for s in reports[1].slips] == ["w"]

    def test_alignment_for_another_word_is_rejected(
        self, matrix: ConfusionMatrix
    ) -> None:
        # Three target phonemes against a five-phoneme word: silently
        # attributing these would point at the wrong syllables.
        alignment = [("p͈", "p͈", "match"), ("a", "a", "match"),
                     ("ŋ", "ŋ", "match")]

        with pytest.raises(ValueError, match="different words"):
            explain(alignment, SAGWA, matrix)


class TestSeverity:
    def test_cheap_substitution_is_near(self, matrix: ConfusionMatrix) -> None:
        # k/kʰ costs 0.2 - the same sound with more air behind it
        alignment = [
            ("s", "s", "match"),
            ("a", "a", "match"),
            ("kʰ", "k", "sub"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert reports[1].status == "near"

    def test_place_of_articulation_move_is_blurred(
        self, matrix: ConfusionMatrix
    ) -> None:
        # s/t costs 0.3
        alignment = [
            ("t", "s", "sub"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert reports[0].status == "blurred"

    def test_unlisted_pair_is_different(self, matrix: ConfusionMatrix) -> None:
        # Not in the matrix at all -> default 0.8
        alignment = [
            ("l", "s", "sub"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert reports[0].status == "different"

    def test_worst_slip_sets_the_status(self, matrix: ConfusionMatrix) -> None:
        alignment = [
            ("s", "s", "match"),
            ("a", "a", "match"),
            ("kʰ", "k", "sub"),   # near (0.2)
            ("t", "w", "sub"),    # unlisted -> 0.8
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert reports[1].status == "different"


class TestMuting:
    def test_without_a_table_nothing_is_muted(
        self, matrix: ConfusionMatrix
    ) -> None:
        # 아빠: the model writes /p͈/ as /p/ 12% of the time by itself
        appa = [
            {"text": "아", "spoken": "아", "span": [0, 1]},
            {"text": "빠", "spoken": "빠", "span": [1, 3]},
        ]
        alignment = [
            ("a", "a", "match"),
            ("p", "p͈", "sub"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, appa, matrix)

        assert reports[1].status != "no_comment"
        assert not reports[1].slips[0].muted

    def test_the_models_own_confusion_is_muted(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        appa = [
            {"text": "아", "spoken": "아", "span": [0, 1]},
            {"text": "빠", "spoken": "빠", "span": [1, 3]},
        ]
        alignment = [
            ("a", "a", "match"),
            ("p", "p͈", "sub"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, appa, matrix, errors)

        assert reports[1].status == "no_comment"
        slip = reports[1].slips[0]
        assert slip.muted
        assert slip.observations == 10
        assert slip.model_error_rate == pytest.approx(10 / 83)

    def test_a_pair_the_model_does_not_make_still_reports(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        # s->t is not in the error table: this model does not do it alone
        alignment = [
            ("t", "s", "sub"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix, errors)

        assert reports[0].status == "blurred"
        assert reports[0].slips[0].model_error_rate is None

    def test_glide_deletion_is_muted(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        # /w/ goes missing in 3.5% of its chances - the README's
        # "활음 손실" limit, which is the model's and not the child's
        alignment = [
            ("s", "s", "match"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("", "w", "del"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix, errors)

        assert reports[1].status == "no_comment"
        assert reports[1].slips[0].muted

    def test_a_pair_seen_once_is_not_a_rate(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        # s->tɕ appears once in 76 chances. One observation is a
        # coincidence, so it must not buy silence.
        alignment = [
            ("tɕ", "s", "sub"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix, errors)

        assert reports[0].slips[0].observations == 1
        assert not reports[0].slips[0].muted

    def test_insertions_never_blame_the_child(
        self, matrix: ConfusionMatrix
    ) -> None:
        # Nothing measures how often this model invents a phoneme, so an
        # extra one cannot be attributed to anybody.
        alignment = [
            ("s", "s", "match"),
            ("a", "a", "match"),
            ("h", "", "ins"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        slips = [s for r in reports for s in r.slips]
        assert [s.kind for s in slips] == ["ins"]
        assert slips[0].muted

    def test_an_insertion_does_not_cost_a_correct_syllable_its_verdict(
        self, matrix: ConfusionMatrix
    ) -> None:
        # 사 produced both its phonemes; an extra /h/ landing beside them
        # is model noise and must not pull it down to "no_comment".
        alignment = [
            ("s", "s", "match"),
            ("h", "", "ins"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("w", "w", "match"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix)

        assert reports[0].slips  # the insertion is still recorded
        assert [r.status for r in reports] == ["clean", "clean"]

    def test_a_muted_deletion_still_withholds_the_verdict(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        # Unlike an insertion, a muted deletion means a target phoneme
        # really did not come back - we just cannot say whose doing it
        # was, which is "no_comment" and not "clean".
        alignment = [
            ("s", "s", "match"),
            ("a", "a", "match"),
            ("k", "k", "match"),
            ("", "w", "del"),
            ("a", "a", "match"),
        ]

        reports = explain(alignment, SAGWA, matrix, errors)

        assert reports[1].status == "no_comment"


class TestAgainstTheRealMatcher:
    """The alignment must come from the scorer, not from hand-written
    tuples, or the feedback could describe a match the scorer never
    made."""

    def test_scored_utterance_attributes_to_the_right_syllable(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        matcher = Matcher(matrix)
        result = matcher.best_match(["t", "a", "k", "w", "a"], [
            {"id": "apple", "text": "사과",
             "phonemes": SAGWA_PHONEMES, "threshold": 0.7}
        ])

        assert result.passed
        reports = explain(result.alignment, SAGWA, matrix, errors)

        assert reports[0].status == "blurred"
        assert reports[0].slips[0].heard == "t"
        assert reports[1].status == "clean"

    def test_surrounding_noise_does_not_reach_the_syllables(
        self, matrix: ConfusionMatrix, errors: ModelErrors
    ) -> None:
        # "음... 사과" - the preamble sits outside the matched window, so
        # it is skipped, not blamed on 사.
        matcher = Matcher(matrix)
        result = matcher.best_match(["ɯ", "m", "h", "s", "a", "k", "w", "a"], [
            {"id": "apple", "text": "사과",
             "phonemes": SAGWA_PHONEMES, "threshold": 0.7}
        ])

        assert result.passed
        reports = explain(result.alignment, SAGWA, matrix, errors)

        assert [r.status for r in reports] == ["clean", "clean"]


class TestSpansFromTheBuild:
    """`syllable_spans` and `explain` have to agree about what a span
    means, so the feedback is exercised against what the build emits."""

    def test_build_spans_feed_explain(self, matrix: ConfusionMatrix) -> None:
        groups = [("사", ["s", "a"]), ("과", ["k", "w", "a"])]
        spans = syllable_spans("사과", groups, SAGWA_PHONEMES)

        reports = explain(
            [(p, p, "match") for p in SAGWA_PHONEMES], spans, matrix
        )

        assert [r.text for r in reports] == ["사", "과"]

    def test_resyllabified_word_keeps_both_spellings(self) -> None:
        # 먹어요 -> 머거요: the ㄱ of 먹 is spoken at the start of 거
        groups = [("머", ["m", "ʌ"]), ("거", ["k", "ʌ"]), ("요", ["j", "o"])]
        spans = syllable_spans(
            "먹어요", groups, ["m", "ʌ", "k", "ʌ", "j", "o"]
        )

        assert [s["text"] for s in spans] == ["먹", "어", "요"]
        assert [s["spoken"] for s in spans] == ["머", "거", "요"]
        # The ㄱ lands on 어, whose spoken form 거 is the one that has it
        assert spans[1]["span"] == [2, 4]

    def test_grouping_that_drifted_is_refused(self) -> None:
        groups = [("사", ["s", "a"]), ("과", ["k", "a"])]  # lost the glide

        assert syllable_spans("사과", groups, SAGWA_PHONEMES) is None

    def test_syllable_count_mismatch_is_refused(self) -> None:
        groups = [("사", ["s", "a", "k", "w", "a"])]

        assert syllable_spans("사과", groups, SAGWA_PHONEMES) is None
