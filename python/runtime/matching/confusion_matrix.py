"""Confusion Matrix for phoneme comparison.

Loads a language-specific JSON confusion matrix and exposes substitution,
insertion and deletion costs to the edit-distance engine. All costs are in
[0, 1] where 0 = "no penalty" and 1 = "full penalty" (i.e., comparable to
a completely unrelated phoneme).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def _normalize_pair_key(key: str) -> tuple[str, str] | None:
    """Convert a 'a|b' key into a sorted tuple. Returns None for non-pair keys
    (e.g., '_comment_*' or already-sorted defaults).
    """
    if "|" not in key:
        return None
    parts = key.split("|")
    if len(parts) != 2:
        return None
    return tuple(sorted(parts))  # type: ignore[return-value]


class ConfusionMatrix:
    """Phoneme-level confusion costs.

    Substitutions are commutative ('s|t' covers both directions).
    Comment keys starting with '_' are ignored.
    """

    def __init__(
        self,
        matrix_id: str,
        language: str,
        version: str,
        substitutions: dict[tuple[str, str], float],
        deletions: dict[str, float],
        insertions: dict[str, float],
        default_substitution: float,
        default_deletion: float,
        default_insertion: float,
    ) -> None:
        self.matrix_id = matrix_id
        self.language = language
        self.version = version
        self._substitutions = substitutions
        self._deletions = deletions
        self._insertions = insertions
        self.default_substitution = default_substitution
        self.default_deletion = default_deletion
        self.default_insertion = default_insertion

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "ConfusionMatrix":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        # Strip comment keys (anything starting with '_')
        def clean(d: dict) -> dict:
            return {k: v for k, v in d.items() if not k.startswith("_")}

        raw_subs = clean(data.get("substitutions", {}))
        substitutions: dict[tuple[str, str], float] = {}
        for key, cost in raw_subs.items():
            pair = _normalize_pair_key(key)
            if pair is None:
                continue
            substitutions[pair] = float(cost)

        deletions = {
            k: float(v) for k, v in clean(data.get("deletions", {})).items()
        }
        insertions = {
            k: float(v) for k, v in clean(data.get("insertions", {})).items()
        }

        return cls(
            matrix_id=data.get(
                "matrix_id", path.stem  # fallback to filename
            ),
            language=data.get("language", "unknown"),
            version=data.get("version", "0.0.0"),
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            default_substitution=float(
                data.get("default_substitution", 0.8)
            ),
            default_deletion=float(data.get("default_deletion", 0.6)),
            default_insertion=float(data.get("default_insertion", 0.6)),
        )

    # ------------------------------------------------------------------
    # Cost lookups
    # ------------------------------------------------------------------

    def sub_cost(self, a: str, b: str) -> float:
        """Cost of substituting phoneme `a` with phoneme `b`."""
        if a == b:
            return 0.0
        return self._substitutions.get(
            tuple(sorted((a, b))), self.default_substitution
        )

    def del_cost(self, phoneme: str) -> float:
        """Cost of deleting `phoneme` from the user output (target had it,
        user did not produce it)."""
        return self._deletions.get(phoneme, self.default_deletion)

    def ins_cost(self, phoneme: str) -> float:
        """Cost of inserting `phoneme` (user produced it, target didn't)."""
        return self._insertions.get(phoneme, self.default_insertion)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def known_substitutions(self) -> Iterable[tuple[str, str, float]]:
        for (a, b), cost in self._substitutions.items():
            yield a, b, cost

    def __repr__(self) -> str:
        return (
            f"ConfusionMatrix(id={self.matrix_id!r}, "
            f"lang={self.language!r}, "
            f"subs={len(self._substitutions)}, "
            f"dels={len(self._deletions)}, "
            f"ins={len(self._insertions)})"
        )
