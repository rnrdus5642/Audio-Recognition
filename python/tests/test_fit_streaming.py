"""The rule that turns a tuning grid into the settings that ship.

Picking the top single-word score alone chose skip_cost 0.08, which led
by 1.3 points on train - 0.8 standard errors at that sample size, and
gone by the test split. The same profile scored 80.2% once anything
surrounded the word, against 89.8%. Nobody re-running the tool would
have known: it would have overwritten the thresholds with the worse
setting and reported a number that looked fine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "child_tuning"
sys.path.insert(0, str(TOOLS))

from fit_streaming import choose, standard_error  # noqa: E402


def row(skip, coverage, context, detection, surround, rate):
    return {"skip_cost": skip, "coverage": coverage,
            "context_mult": context, "consecutive": 2,
            "detection": detection, "surround": surround, "rate": rate}


# The grid actually measured on 472 single-word utterances from 52
# children, with 900 of the same words inside sentences.
GRID = [
    row(0.005, 0.8, 4.0, 0.847, 0.923, 0.0023),
    row(0.005, 0.8, 6.0, 0.847, 0.951, 0.0023),
    row(0.005, 0.8, 8.0, 0.847, 0.955, 0.0023),
    row(0.005, 0.9, 8.0, 0.845, 0.958, 0.0026),
    row(0.02, 0.8, 8.0, 0.850, 0.952, 0.0025),
    row(0.05, 0.8, 4.0, 0.856, 0.909, 0.0032),
    row(0.08, 0.8, 4.0, 0.860, 0.870, 0.0034),
]


class TestChoose:
    def test_picks_the_shipped_profile(self) -> None:
        best = choose(GRID, n_isolated=472, n_surround=900)
        assert (best["skip_cost"], best["coverage"],
                best["context_mult"]) == (0.005, 0.8, 8.0)

    def test_leader_is_within_noise(self) -> None:
        """The row it does not pick is ahead by less than one error."""
        margin = standard_error(0.860, 472)
        assert 0.860 - 0.847 < margin

    def test_real_gap_wins(self) -> None:
        """A lead outside the noise is taken, whatever the other columns."""
        grid = GRID + [row(0.03, 0.8, 8.0, 0.95, 0.60, 0.0040)]
        best = choose(grid, n_isolated=472, n_surround=900)
        assert best["detection"] == 0.95

    def test_false_accepts_break_a_double_tie(self) -> None:
        """Equal on both conditions, the quieter setting wins."""
        grid = [row(0.005, 0.8, 8.0, 0.847, 0.955, 0.0023),
                row(0.02, 0.9, 8.0, 0.847, 0.955, 0.0051)]
        assert choose(grid, 472, 900)["skip_cost"] == 0.005

    def test_single_row(self) -> None:
        assert choose([GRID[0]], 472, 900) is GRID[0]


class TestStandardError:
    def test_shrinks_with_sample_size(self) -> None:
        assert standard_error(0.9, 1000) < standard_error(0.9, 100)

    def test_survives_certainty(self) -> None:
        """p of exactly 0 or 1 must not divide by zero."""
        assert standard_error(1.0, 100) > 0
        assert standard_error(0.0, 100) > 0

    def test_survives_empty(self) -> None:
        assert standard_error(0.5, 0) == pytest.approx(0.5)
