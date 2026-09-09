"""Model-aware review of differences in an utterance that already passed.

`Matcher` answers one question - did the child say the word - and the
product needs that answer to stay binary. This module reads the same
result a second time to answer a different one: which part of the word
came out unlike the target, and how far off was it.

The alignment records the recognizer's differences, not measured errors
in the child's articulation. Costs describe the scorer's tolerance only.

The trap is that the alignment describes the *recogniser*, not the
child. The child model still makes an 8.71% character error, and some
phonemes it confuses on its own no matter who is speaking: it writes
/p͈/ as /p/ in 12% of its chances. Reported blindly, one confirmation
in eight of 아빠 would tell a child their ㅃ was soft when it was not.
Telling a child about a mistake they did not make is worse than saying
nothing, so anything the recogniser is known to get wrong by itself is
muted - see `ModelErrors` and `shared/reference/reference_errors.json`.

That table is thin (697 utterances, most pairs seen once or twice), so
muting is deliberately conservative and a quiet syllable reads as "no
comment", never as "correct". Insertions are muted unconditionally:
the table has no insertion counts at all, and a CTC decoder emits
spurious phonemes often enough that blaming the child for one would be
guessing.

Missing pairs and insufficient observations also withhold guidance.
The same review is implemented in Unity's ModelFeedback.cs; scoring is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .confusion_matrix import ConfusionMatrix


# Severity, read off the confusion matrix's own cost scale rather than
# invented alongside it. The tiers are where ko_child_v1/v2 already
# cluster: 0.1-0.2 is the same sound made slightly differently (불파음,
# 평음/격음/경음), 0.25-0.4 is the place of articulation moving (s->t,
# l->d), and past that the vowels start swapping and it stops sounding
# like the target at all.
NEAR_COST = 0.2
BLURRED_COST = 0.4

# Provisional display threshold. This frequency is not a confidence
# estimate for an individual child or utterance.
DEFAULT_MUTE_ABOVE = 0.02

# A pair seen once is a coincidence, not a rate. Matches the cut
# `child_tuning/derive_thresholds.py` uses on the same file.
DEFAULT_MIN_OBSERVATIONS = 2


@dataclass
class Slip:
    """One place the alignment differs from the target."""

    kind: str                       # "sub" | "del" | "ins"
    target: str                     # target phoneme ("" for ins)
    heard: str                      # recognised phoneme ("" for del)
    cost: float                     # confusion-matrix price of the gap
    model_error_rate: float | None  # how often this model does it alone
    observations: int               # times that was seen in the table
    muted: bool
    muted_because: str = ""

    @property
    def severity(self) -> str:
        """Where `cost` falls on the matrix's scale."""
        if self.cost <= NEAR_COST:
            return "near"
        if self.cost <= BLURRED_COST:
            return "blurred"
        return "different"


@dataclass
class SyllableReport:
    """One syllable of the target, and what happened to it."""

    text: str          # as printed on screen
    spoken: str        # as pronounced once the rules run
    span: tuple[int, int]
    slips: list[Slip] = field(default_factory=list)

    # "clean"      nothing differed
    # "near"       differed by a sound the matrix prices as nearly equal
    # "blurred"    place of articulation moved
    # "different"  came out as another sound entirely
    # "no_comment" everything that differed is something the model does
    #              on its own, so there is nothing we can honestly say
    status: str = "clean"

    @property
    def resyllabified(self) -> bool:
        """True when this syllable is spoken as a different one.

        먹어요 is spoken 머거요, so the ㄱ written at the end of 먹 is
        spoken at the start of 거 and its slips land here, on 어. Say
        `spoken` when describing the sound or the reader will look for a
        ㄱ in 어 and not find one.
        """
        return self.text != self.spoken


class ModelErrors:
    """How often a recogniser mis-transcribes a phoneme by itself.

    Built by `child_tuning/build_reference.py` from children reading
    known text: entries compare model output with reference text. They
    do not establish the cause of an individual difference. `derive_thresholds.py` reads the same file to
    ask whether a word can be passed at all; here it answers the
    opposite question - whether a slip is worth mentioning.

    Tied to one recogniser. `matrix_id` names which; applying a table
    from another model would mute the wrong things and, worse, leave the
    ones this model actually makes unmuted.
    """

    def __init__(
        self,
        matrix_id: str,
        substitutions: dict[tuple[str, str], tuple[int, int]],
        deletions: dict[str, tuple[int, int]],
        utterances: int,
    ) -> None:
        if not isinstance(matrix_id, str) or not matrix_id.strip() or matrix_id == "unknown":
            raise ValueError("Invalid model error profile identity")
        if type(utterances) is not int or utterances < 0:
            raise ValueError("Invalid model error utterance count")
        for count, total in list(substitutions.values()) + list(deletions.values()):
            if (type(count) is not int or type(total) is not int
                    or count < 0 or total < 0 or count > total):
                raise ValueError("Invalid model error observation counts")
        self.matrix_id = matrix_id
        self.utterances = utterances
        self._substitutions = substitutions
        self._deletions = deletions

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelErrors":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        substitutions: dict[tuple[str, str], tuple[int, int]] = {}
        for key, value in data.get("substitutions", {}).items():
            if key.startswith("_") or "|" not in key:
                continue
            wanted, said = key.split("|", 1)
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError("Model error observations must be [count, total]")
            substitutions[(wanted, said)] = (value[0], value[1])

        deletions = {
            key: tuple(value)
            for key, value in data.get("deletions", {}).items()
            if not key.startswith("_")
        }

        return cls(
            matrix_id=data.get("matrix_id", path.stem),
            substitutions=substitutions,
            deletions=deletions,
            utterances=data.get("utterances", 0),
        )

    def substitution(self, target: str, heard: str) -> tuple[int, float]:
        """(times observed, rate) for the model writing `heard` for
        `target`. Direction matters - the table is not symmetric the way
        confusion-matrix costs are."""
        return self._lookup(self._substitutions.get((target, heard)))

    def deletion(self, target: str) -> tuple[int, float]:
        """(times observed, rate) for the model dropping `target`."""
        return self._lookup(self._deletions.get(target))

    def statistics(self, operation: str, target: str, heard: str):
        """Directional counts; None means unmeasured, not zero errors."""
        if operation == "sub":
            return self._substitutions.get((target, heard))
        if operation == "del":
            return self._deletions.get(target)
        return None

    @staticmethod
    def _lookup(entry: tuple[int, int] | None) -> tuple[int, float]:
        if entry is None:
            return (0, 0.0)
        count, total = entry
        return (count, count / total if total else 0.0)

    def __repr__(self) -> str:
        return (
            f"ModelErrors(matrix_id={self.matrix_id!r}, "
            f"substitutions={len(self._substitutions)}, "
            f"deletions={len(self._deletions)}, "
            f"utterances={self.utterances})"
        )


def explain(
    alignment: list[tuple[str, str, str]],
    syllables: list[dict],
    matrix: ConfusionMatrix,
    model_errors: ModelErrors | None = None,
    *,
    mute_above: float = DEFAULT_MUTE_ABOVE,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> list[SyllableReport]:
    """Attribute an alignment's slips to the syllables they landed in.

    `alignment` is `MatchResult.alignment` - (user, target, op) per step,
    covering the whole target. `syllables` is the target's `syllables`
    entry from targets.json.

    Missing or mismatched evidence withholds guidance. A pair absent from
    a sparse table has not been measured; it is not proof of a child's error.
    """
    profile_matches = bool(model_errors and model_errors.matrix_id == matrix.matrix_id
                           and matrix.matrix_id not in ("", "unknown"))
    reports = [
        SyllableReport(
            text=s["text"],
            # targets.json built before syllables carried a spoken form
            spoken=s.get("spoken", s["text"]),
            span=(s["span"][0], s["span"][1]),
        )
        for s in syllables
    ]
    if not reports:
        return []

    def owner(target_index: int) -> SyllableReport:
        for report in reports:
            if report.span[0] <= target_index < report.span[1]:
                return report
        # An insertion trailing the last target phoneme belongs to the
        # syllable it followed.
        return reports[-1]

    consumed = 0
    for user_ph, target_ph, op in alignment:
        if op == "match":
            consumed += 1
            continue

        report = owner(consumed)
        if op == "sub":
            stats = model_errors.statistics("sub", target_ph, user_ph) if profile_matches else None
            slip = _slip(
                "sub", target_ph, user_ph,
                matrix.sub_cost(target_ph, user_ph),
                stats, profile_matches, mute_above, min_observations,
            )
            consumed += 1
        elif op == "del":
            stats = model_errors.statistics("del", target_ph, "") if profile_matches else None
            slip = _slip(
                "del", target_ph, "",
                matrix.del_cost(target_ph),
                stats, profile_matches, mute_above, min_observations,
            )
            consumed += 1
        else:  # "ins" - consumes no target phoneme
            slip = Slip(
                kind="ins", target="", heard=user_ph,
                cost=matrix.ins_cost(user_ph),
                model_error_rate=None, observations=0,
                muted=True,
                muted_because="이 모델의 삽입 오류율은 측정된 적이 없음",
            )
        report.slips.append(slip)

    if consumed != reports[-1].span[1]:
        raise ValueError(
            f"alignment covers {consumed} target phonemes but the "
            f"syllables cover {reports[-1].span[1]}; they describe "
            "different words"
        )

    for report in reports:
        # An insertion never moves the verdict. Every target phoneme of
        # the syllable was still produced - something extra turned up
        # beside them - and since nothing measures how often this model
        # invents a phoneme, that extra sound is no evidence about the
        # child in either direction. Letting it count would quietly take
        # "좋아요" away from a syllable that was said correctly, on the
        # strength of noise we have just admitted we cannot read.
        judged = [s for s in report.slips if s.kind != "ins"]
        audible = [s for s in judged if not s.muted]
        if any(s.muted for s in judged):
            report.status = "no_comment"
        elif audible:
            report.status = max(audible, key=lambda s: s.cost).severity
        elif judged:
            report.status = "no_comment"
        else:
            report.status = "clean"

    return reports


def _slip(
    kind: str,
    target: str,
    heard: str,
    cost: float,
    statistics,
    profile_matches: bool,
    mute_above: float,
    min_observations: int,
) -> Slip:
    from .model_feedback import withholding_reason
    count, total = statistics or (0, 0)
    rate = count / total if total else 0.0
    reason = withholding_reason(kind, statistics, profile_matches,
                                mute_above=mute_above, min_observations=min_observations)
    return Slip(
        kind=kind,
        target=target,
        heard=heard,
        cost=cost,
        model_error_rate=rate if count else None,
        observations=count,
        muted=bool(reason),
        muted_because=reason,
    )


__all__ = [
    "NEAR_COST",
    "BLURRED_COST",
    "DEFAULT_MUTE_ABOVE",
    "DEFAULT_MIN_OBSERVATIONS",
    "Slip",
    "SyllableReport",
    "ModelErrors",
    "explain",
]

# Shared web/Unity confirmation contract; keep the model-aware API above intact.
from .confirmation import PhonemeComparison, PronunciationFeedback

__all__ += ["PhonemeComparison", "PronunciationFeedback"]
