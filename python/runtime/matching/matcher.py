"""High-level phoneme matching.

Implements weighted Levenshtein edit distance over IPA phoneme sequences,
normalized to a [0, 1] similarity score. The score is the primary signal
used by the application to decide pass / retry.

This module is kept in pure Python with no exotic dependencies so that
porting to C# for Unity is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .confusion_matrix import ConfusionMatrix
from .acoustic import AcousticEvidence, validate_acoustic_policy


@dataclass
class MatchResult:
    """Single match outcome."""

    score: float                # [0.0, 1.0], higher = closer
    distance: float             # raw weighted edit distance
    target_id: str | None       # matched target answer id (if any)
    target_text: str | None     # matched target text (if any)
    target_phonemes: list[str] | None
    user_phonemes: list[str]
    passed: bool                # threshold met AND not vetoed by another action
    # Substring matching: the slice of user_phonemes that actually matched
    # the target. For exact mode this is [0, len(user_phonemes)].
    window_start: int = 0
    window_end: int = 0
    # Per-step alignment (user_ph, target_ph, op) within the window.
    alignment: list[tuple[str, str, str]] | None = None
    threshold: float = 0.0
    # Set when a passing answer was refused because a DIFFERENT action
    # explains the same speech better (see `Matcher.best_match`). False
    # for every answer that ships no contrast set, which is every answer
    # in a catalog built before contrasts existed - so a caller that
    # never reads these fields behaves exactly as it did.
    competitor_rejected: bool = False
    competitor_reason: str = ""
    competitor_text: str | None = None
    competitor_score: float | None = None
    accepted_whole_score: float | None = None
    whole_rejected: bool = False
    used_short_utterance_policy: bool = False
    acoustic_evidence: AcousticEvidence | None = None


#: `MatchResult.competitor_reason` when another action scored higher.
OTHER_ACTION_SCORED_HIGHER = "other_action_scored_higher"
CONTEXT_PREFIX_MIN_SCORE = 0.80


REBUILD = ("rebuild targets.json with "
           "python -m python.build.build_targets")


def _named(owner: object) -> str:
    """How to refer to the answer an error came from."""
    text = owner.get("text") if isinstance(owner, dict) else None
    return f"'{text}'" if text else "an answer"


def check_action_group(value: object, owner: object = None) -> str | None:
    """The answer's action group, or None when it has none.

    Absent and null are legacy data: no group, no contrast, judged the
    way every answer was judged before groups existed. Anything else has
    to be a name - a non-empty string with no whitespace - because groups
    are compared for equality, and " lift" is a group of one that
    protects nothing. The C# loader and runtime refuse the same values.
    """
    if value is None:
        return None
    if (not isinstance(value, str) or not value
            or any(ch.isspace() for ch in value)):
        raise ValueError(
            f"{_named(owner)} has an action_group that is not a name: "
            f"{value!r}; {REBUILD}")
    return value


def check_alternatives(
    value: object, owner: object = None
) -> list[tuple[str, list[str], str]] | None:
    """The embedded action alternatives as (text, phonemes, group).

    None for absent or null - legacy data, nothing to compare against.
    An empty list is allowed and says the build found nothing to embed.
    Every other non-list - False, 0, "", {} - is a file claiming to
    configure the protection while switching it off, which is exactly the
    failure this must not read as "absent". Elements are checked the same
    way here and in C#, so one bad file cannot load in the web tool and
    throw in Unity.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"{_named(owner)} has contrastive_alternatives that are not a "
            f"list: {value!r}; {REBUILD}")

    out: list[tuple[str, list[str], str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{_named(owner)} has a contrastive alternative that is "
                f"not an object: {entry!r}; {REBUILD}")
        text = entry.get("text")
        phonemes = entry.get("phonemes")
        if (not isinstance(text, str) or not text
                or not isinstance(phonemes, list) or not phonemes
                or not all(isinstance(p, str) and p for p in phonemes)):
            raise ValueError(
                f"{_named(owner)} has a contrastive alternative missing "
                f"text or phonemes; {REBUILD}")
        group = check_action_group(entry.get("action_group"), owner)
        if group is None:
            raise ValueError(
                f"{_named(owner)} has a contrastive alternative with no "
                f"action_group; {REBUILD}")
        out.append((text, list(phonemes), group))
    return out


def validate_contrast_metadata(
    answer: dict,
) -> tuple[str | None, list[tuple[str, list[str], str]] | None]:
    """Refuse an answer whose contrast metadata cannot be honoured.

    Alternatives without a group name nothing to be told apart from and
    nothing to anchor against, so they are a build error rather than a
    quiet no-op. A group on its own is fine: it says which action this
    answer is, and a catalog may hold only one.

    Returns the parsed `(action_group, alternatives)` so the scorer reads
    the metadata once and reads it the same way this check did.
    """
    group = check_action_group(answer.get("action_group"), answer)
    alternatives = check_alternatives(
        answer.get("contrastive_alternatives"), answer)
    if alternatives is not None and group is None:
        raise ValueError(
            f"{_named(answer)} ships contrastive alternatives but no "
            f"action_group, so nothing says which action it is; {REBUILD}")
    return group, alternatives


def validate_catalog(answers: list[dict]) -> list[dict]:
    """Every answer's contrast metadata, checked at load.

    Called where a catalog is read rather than only where it is scored,
    so a file that Unity would refuse cannot quietly half-work in the web
    tool - and so the error names the build, not the child's frame.
    """
    for answer in answers:
        validate_contrast_metadata(answer)
        validate_word_policy(answer)
    return answers


def validate_word_policy(answer: dict) -> None:
    """Optional presentation tuning; absent fields preserve the old policy."""
    validate_acoustic_policy(answer)
    frames = answer.get("confirmation_frames")
    if frames is not None and (
        type(frames) is not int or not 1 <= frames <= 6
    ):
        raise ValueError(f"{_named(answer)} confirmation_frames must be an integer in [1, 6]")
    for key in ("min_whole_score", "contrast_margin"):
        value = answer.get(key, 0.0)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0.0 <= value <= 1.0):
            raise ValueError(f"{_named(answer)} {key} must be a finite number in [0, 1]")
    contexts = answer.get("context_prefixes")
    if contexts is not None:
        if not isinstance(contexts, list):
            raise ValueError(f"{_named(answer)} context_prefixes must be a list")
        for context in contexts:
            if (not isinstance(context, dict)
                    or not isinstance(context.get("text"), str) or not context["text"].strip()
                    or any(not isinstance(context.get(key), list) or not context[key]
                           or not all(isinstance(p, str) and p.strip() for p in context[key])
                           for key in ("phonemes",))):
                raise ValueError(f"{_named(answer)} context_prefixes requires text and phonemes")
    short = answer.get("short_utterance_policy")
    if short is not None:
        if not isinstance(short, dict):
            raise ValueError(f"{_named(answer)} short_utterance_policy must be an object")
        for key in ("threshold", "min_whole_score"):
            value = short.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0.0 < value <= 1.0):
                raise ValueError(f"{_named(answer)} short_utterance_policy {key} must be in (0, 1]")
        extra = short.get("max_extra_phonemes")
        if type(extra) is not int or not 0 <= extra <= 6:
            raise ValueError(f"{_named(answer)} short_utterance_policy max_extra_phonemes must be an integer in [0, 6]")


def substring_edit_distance(
    user: list[str],
    target: list[str],
    matrix: ConfusionMatrix,
    skip_cost: float = 0.0,
) -> tuple[float, int, int, list[tuple[str, str, str]]]:
    """Find the best substring of `user` that matches `target`.

    Unlike full edit distance, this lets `user` carry noise before and/or
    after the actual target word: any prefix/suffix is skipped at
    `skip_cost` per phoneme. Crucial when the ASR captures background
    noise or coughs as extra phonemes around the real utterance.

    `skip_cost=0.0` (the default) makes skipping free - the textbook
    semi-global alignment. Note that free skipping makes the score
    monotonically non-decreasing in `len(user)`: appending phonemes only
    ever adds candidate windows, so a long enough utterance clears any
    threshold. Callers that score real user speech should pass a small
    positive `skip_cost` (see `Matcher`); it must stay well below
    `matrix.ins_cost` so genuine surrounding noise is still cheaper to
    skip than to align.

    Returns:
        (distance, window_start, window_end, ops)

    where:
        * `user[window_start:window_end]` is the best-matching substring
        * `distance` is the weighted edit distance from that substring
          to the full target
        * `ops` is the per-step alignment trace
          [(user_phoneme | "", target_phoneme | "", op)] with
          op in {"match", "sub", "ins", "del"}

    Algorithm: standard DP, but `dp[0][j] = j * skip_cost` (starting the
    window at any position in user costs only the skipped prefix), and we
    minimise `dp[m][j] + (n - j) * skip_cost` over j (stopping anywhere,
    paying for the skipped suffix).
    """
    n, m = len(user), len(target)

    if m == 0:
        return (0.0, 0, 0, [])
    if n == 0:
        d = sum(matrix.del_cost(t) for t in target)
        return (d, 0, 0, [("", t, "del") for t in target])

    INF = float("inf")
    dp: list[list[float]] = [[INF] * (n + 1) for _ in range(m + 1)]
    parent: list[list[tuple[int, int, str] | None]] = [
        [None] * (n + 1) for _ in range(m + 1)
    ]

    # Window start: an empty target prefix costs only the user prefix skipped
    for j in range(n + 1):
        dp[0][j] = j * skip_cost

    # Target prefix vs empty user prefix: must delete every target phoneme
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + matrix.del_cost(target[i - 1])
        parent[i][0] = (i - 1, 0, "del")

    for i in range(1, m + 1):
        ti = target[i - 1]
        for j in range(1, n + 1):
            uj = user[j - 1]
            sub_op = "match" if ti == uj else "sub"
            sub_val = dp[i - 1][j - 1] + matrix.sub_cost(ti, uj)
            ins_val = dp[i][j - 1] + matrix.ins_cost(uj)
            del_val = dp[i - 1][j] + matrix.del_cost(ti)

            # Tie-breaking order: prefer match/sub > del > ins so that the
            # backtrack tends toward a contiguous window rather than
            # zig-zagging through inserts.
            best_val = sub_val
            best_par = (i - 1, j - 1, sub_op)
            if del_val < best_val:
                best_val = del_val
                best_par = (i - 1, j, "del")
            if ins_val < best_val:
                best_val = ins_val
                best_par = (i, j - 1, "ins")

            dp[i][j] = best_val
            parent[i][j] = best_par

    # Find the best ending position, paying for the skipped user suffix
    best_j = 0
    best_dist = dp[m][0] + n * skip_cost
    for j in range(1, n + 1):
        total = dp[m][j] + (n - j) * skip_cost
        if total < best_dist:
            best_dist = total
            best_j = j

    # Backtrack from (m, best_j) until we reach i=0; that j is the start
    ops: list[tuple[str, str, str]] = []
    i, j = m, best_j
    while i > 0:
        par = parent[i][j]
        if par is None:
            break
        pi, pj, op = par
        if op == "match" or op == "sub":
            ops.append((user[j - 1], target[i - 1], op))
        elif op == "ins":
            ops.append((user[j - 1], "", "ins"))
        else:  # "del"
            ops.append(("", target[i - 1], "del"))
        i, j = pi, pj

    start_j = j
    ops.reverse()
    return (best_dist, start_j, best_j, ops)


def weighted_edit_distance(
    user: list[str],
    target: list[str],
    matrix: ConfusionMatrix,
) -> float:
    """Levenshtein with confusion-matrix costs.

    Costs are looked up from `matrix`:
      * sub_cost(a, b) for substitution (0 if a == b)
      * del_cost(t)    for "target had it, user dropped it"
      * ins_cost(u)    for "user inserted extra phoneme"

    Returns the raw weighted distance (>= 0).
    """
    n, m = len(user), len(target)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return sum(matrix.del_cost(t) for t in target)
    if m == 0:
        return sum(matrix.ins_cost(u) for u in user)

    # dp[i][j] = min cost to transform user[:i] into target[:j]
    # Rows = user, columns = target.
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]

    # First column: deleting target prefix (no user phonemes produced yet)
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + matrix.ins_cost(user[i - 1])
    # First row: target had phonemes that user dropped
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + matrix.del_cost(target[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + matrix.sub_cost(user[i - 1], target[j - 1])
            ins = dp[i - 1][j] + matrix.ins_cost(user[i - 1])
            dele = dp[i][j - 1] + matrix.del_cost(target[j - 1])
            dp[i][j] = min(sub, ins, dele)

    return dp[n][m]


def similarity_score(
    distance: float,
    target_length: int,
    user_length: int | None = None,
) -> float:
    """Convert raw distance to a [0, 1] similarity score.

    Normalizes by max(target_length, 1) so that:
      * distance == 0           -> score == 1.0
      * distance >= length      -> score == 0.0

    The denominator is target length (not max(user, target)) because the
    target is the reference: we ask "how much of the target did the user
    correctly reproduce?".

    Length sanity penalty:
      When `user_length` is provided AND the user produced wildly less
      (or more) than the target length, multiply the base score by a
      gentle penalty. This guards against the "ASR truncation -> any
      short word matches" failure mode where a 1-phoneme user output
      accidentally clears the threshold for many candidate words.

      No penalty when |user - target| <= 1.
      Penalty kicks in when ratio is outside [0.5, 2.0].
    """
    if target_length <= 0:
        return 0.0 if distance > 0 else 1.0

    raw = 1.0 - distance / target_length
    score = max(0.0, min(1.0, raw))

    if user_length is None or score == 0.0:
        return score

    # Free zone: small absolute diffs are normal (1 phoneme drop/extra,
    # common for terminal coda deletion or extra glide)
    if abs(user_length - target_length) <= 1:
        return score

    ratio = user_length / target_length
    if ratio < 0.5:
        # Heavy under-shoot (user said much less than target).
        # This is the dominant ASR-truncation failure mode and the source
        # of most false-accepts: a 1-phoneme output accidentally clears
        # the threshold for some random short word in a different
        # segment. Penalty is steep so truncated output loses to a
        # genuine match elsewhere.
        #   ratio 0.0  -> mult 0.20
        #   ratio 0.25 -> mult 0.45
        #   ratio 0.5  -> mult 0.70  (boundary)
        mult = 0.20 + ratio
        score *= mult
    elif ratio > 2.0:
        # Heavy over-shoot (user said much more than target).
        # Less common but still a sign of misalignment; penalize gently.
        mult = 0.5 + 1.0 / ratio
        score *= mult

    return max(0.0, min(1.0, score))


class Matcher:
    """Match a user phoneme sequence against the acceptable answers for
    one question, preferring answers that clear their own thresholds.

    Two scoring modes:

    * `mode="substring"` (default): the target must appear as a
      best-matching window inside the user phoneme sequence. Prefix and
      suffix of the user output (typically ASR noise) are skipped at
      `skip_cost` per phoneme - cheap enough to tolerate incidental
      phonemes from coughs, background and breath, but not free, so a
      long unrelated utterance cannot clear the threshold just by
      offering the matcher more windows to choose from.

    * `mode="exact"`: full sequence-vs-sequence weighted Levenshtein
      with a length-mismatch penalty. Use when you trust the user
      sequence to contain exactly the target word with no surrounding
      noise.

    `skip_cost` defaults to the matrix's own value; pass an explicit
    number to override it (used when tuning against the golden set).

    `coverage` is the window-coverage threshold (see `score_against`) and
    `context_mult`, when set, keeps only the most recent
    `context_mult * len(target) + 3` user phonemes before scoring. Both
    exist because continuous listening needs stricter settings than
    batch scoring - use `Matcher.for_streaming` rather than passing them
    by hand.
    """

    #: Extra phonemes kept beyond `context_mult * len(target)`, so that a
    #: target sitting at the very end of the window is not clipped.
    CONTEXT_PAD = 3

    def __init__(
        self,
        matrix: ConfusionMatrix,
        mode: str = "substring",
        skip_cost: float | None = None,
        coverage: float = 0.5,
        context_mult: float | None = None,
    ) -> None:
        if mode not in ("substring", "exact"):
            raise ValueError(
                f"mode must be 'substring' or 'exact', got {mode!r}"
            )
        if not 0.0 < coverage <= 1.0:
            raise ValueError(f"coverage must be in (0, 1], got {coverage!r}")
        self.matrix = matrix
        self.mode = mode
        self.skip_cost = (
            matrix.skip_cost if skip_cost is None else float(skip_cost)
        )
        self.coverage = float(coverage)
        self.context_mult = (
            None if context_mult is None else float(context_mult)
        )

    @classmethod
    def for_streaming(cls, matrix: ConfusionMatrix) -> "Matcher":
        """Matcher tuned for continuous listening.

        Reads `streaming_profile` from the matrix. Batch scoring keeps a
        lenient window-coverage threshold so a partly-recognised word
        (아빠 -> "앞") still passes; that leniency lets fragments of
        other words clear the bar when the user is talking freely, so
        streaming raises it and bounds how much context is scored.
        """
        p = matrix.streaming_profile
        return cls(
            matrix,
            skip_cost=p.get("skip_cost", matrix.skip_cost),
            coverage=p.get("coverage", 0.5),
            context_mult=p.get("context_mult") or None,
        )

    def context_slice(
        self,
        user: list[str],
        target: list[str],
    ) -> tuple[list[str], int]:
        """The part of `user` that is actually scored, and how much was cut.

        Returns `(scored, dropped)` where `scored` is the trailing slice
        kept under `context_mult` and `dropped` is the number of leading
        phonemes discarded. With no context limit nothing is dropped.

        The limit is coupled to how long a confirmation takes. A target
        only scores while it sits inside this slice, so if the slice
        holds less speech than `consecutive * hop` seconds, a user who
        keeps talking can never accumulate the streak - the answer slides
        out of context before it is confirmed. Korean runs about 10
        phonemes/second, so budget accordingly when tuning.
        """
        if self.context_mult is None or not target:
            return list(user), 0
        keep = int(self.context_mult * len(target)) + self.CONTEXT_PAD
        return list(user[-keep:]), max(0, len(user) - keep)

    def score_against(
        self,
        user: list[str],
        target: list[str],
    ) -> tuple[float, float, int, int, list[tuple[str, str, str]]]:
        """Return (distance, score, window_start, window_end, alignment)
        for a single target.
        """
        if self.mode == "substring":
            # Bound how far back we look. Without this, every extra
            # second of speech adds candidate windows and the best of
            # them drifts upward regardless of what was actually said.
            # Window indices must stay relative to the caller's list even
            # though we score a truncated copy, otherwise anything that
            # slices `user_phonemes` with them points at the wrong
            # phonemes.
            user, offset = self.context_slice(user, target)

            d, ws, we, ops = substring_edit_distance(
                user, target, self.matrix, self.skip_cost
            )
            ws, we = ws + offset, we + offset
            base = max(0.0, 1.0 - d / max(len(target), 1))

            # Window-coverage penalty.
            #
            # Without this, a 1-phoneme user output (e.g., just "a")
            # vs a 7-phoneme target can score ~0.6 because the DP
            # "freely" deletes the other 6 target phonemes - they were
            # never really there, but our deletion costs are small to
            # tolerate 종성 탈락. The penalty kicks in only when the
            # matched window is meaningfully shorter than the target:
            #
            #   window_ratio >= coverage -> no penalty (legit partial
            #                               drop like 종성 탈락)
            #   window_ratio == 0        -> mult 0 (nothing matched)
            #   linear interpolation in between
            target_len = len(target)
            window_len = we - ws
            if target_len > 0 and window_len < self.coverage * target_len:
                mult = (window_len / target_len) / self.coverage
                base *= mult

            s = max(0.0, min(1.0, base))
            return d, s, ws, we, ops

        # Exact mode (legacy)
        d = weighted_edit_distance(user, target, self.matrix)
        s = similarity_score(d, len(target), user_length=len(user))
        return d, s, 0, len(user), []

    def best_match(
        self,
        user: list[str],
        candidates: list[dict],
        acoustic_evidence: dict[str, AcousticEvidence] | None = None,
    ) -> MatchResult:
        """Return the highest-scoring candidate that passes its own threshold.

        If none pass, return the highest-scoring failure for progress display.
        Ties within the same pass/fail group keep the earlier candidate.
        Only the supplied answers are considered; a higher-scoring failure
        must not hide an acceptable answer with a lower pass mark.

        Each candidate dict must have keys:
            id (str), text (str), phonemes (list[str]), threshold (float)

        A candidate may also carry `action_group` and the
        `contrastive_alternatives` the build embedded for it. Those
        change nothing about who wins or what they score - they can only
        take a pass away afterwards, see `_veto_by_other_action`.
        """
        if not candidates:
            return MatchResult(
                score=0.0,
                distance=0.0,
                target_id=None,
                target_text=None,
                target_phonemes=None,
                user_phonemes=user,
                passed=False,
                window_start=0,
                window_end=0,
                alignment=[],
            )

        best_dist = float("inf")
        best_score = -1.0
        best_passed = False
        best = candidates[0]
        best_ws = 0
        best_we = len(user)
        best_ops: list[tuple[str, str, str]] = []
        best_whole = None
        best_whole_rejected = False
        best_threshold = float(best.get("threshold", 0.6))
        best_short = False
        best_acoustic = None
        for cand in candidates:
            validate_word_policy(cand)
            d, s, ws, we, ops = self.score_against(
                user, cand["phonemes"]
            )
            threshold = float(cand.get("threshold", 0.6))
            passed = s >= threshold
            used_short = False
            used_acoustic = None
            whole = None
            whole_rejected = False
            if passed and cand.get("min_whole_score", 0.0) > 0.0:
                whole = self.accepted_whole_score(user, candidates, cand)
                whole_rejected = whole < cand["min_whole_score"]
                passed = not whole_rejected
            short = cand.get("short_utterance_policy")
            if not passed and short and s >= short["threshold"]:
                short_whole = self.short_utterance_score(user, cand)
                if short_whole >= short["min_whole_score"]:
                    passed, used_short, whole_rejected = True, True, False
                    threshold, whole = short["threshold"], short_whole
            evidence = (acoustic_evidence or {}).get(cand["text"])
            if not passed and evidence and evidence.permits(cand, len(user), s):
                whole = self.accepted_whole_score(user, candidates, cand)
                if whole >= cand.get("min_whole_score", 0.0):
                    passed, whole_rejected = True, False
                    threshold = cand["acoustic_keyword_policy"]["min_score"]
                    used_acoustic = evidence
            if ((passed and not best_passed)
                    or (passed == best_passed and s > best_score)):
                best_score = s
                best_passed = passed
                best_dist = d
                best = cand
                best_ws = ws
                best_we = we
                best_ops = ops
                best_whole = whole
                best_whole_rejected = whole_rejected
                best_threshold, best_short = threshold, used_short
                best_acoustic = used_acoustic

        result = MatchResult(
            score=best_score,
            distance=best_dist,
            target_id=best["id"],
            target_text=best["text"],
            target_phonemes=best["phonemes"],
            user_phonemes=user,
            passed=best_passed,
            window_start=best_ws,
            window_end=best_we,
            alignment=best_ops,
            threshold=best_threshold,
            accepted_whole_score=best_whole,
            whole_rejected=best_whole_rejected,
            used_short_utterance_policy=best_short,
            acoustic_evidence=best_acoustic,
        )
        self._veto_by_other_action(result, user, candidates, best)
        return result

    def whole_score(self, user: list[str], target: list[str]) -> float:
        """`user` against `target`, whole against whole.

        No window and no context limit: every phoneme the recogniser
        produced is paid for. This is the existing exact-mode arithmetic,
        used only to compare an answer against the actions it has to be
        told apart from. It never accepts anything and is never reported
        as an answer's score.
        """
        d = weighted_edit_distance(user, target, self.matrix)
        return similarity_score(d, len(target), user_length=len(user))

    def accepted_whole_score(self, user: list[str], candidates: list[dict],
                             winner: dict) -> float:
        """Whole utterance against this answer and aliases of its own group.

        A minimum here complements the substring threshold: a fragment in
        a long unrelated sentence cannot pass just because its local match
        is good. Ungrouped candidates never borrow another answer's score.
        """
        group, alternatives = validate_contrast_metadata(winner)
        validate_word_policy(winner)
        sounds = [winner["phonemes"]]
        if group is not None:
            sounds += [c["phonemes"] for c in candidates
                       if c.get("action_group") == group]
            sounds += [p for _, p, g in alternatives or [] if g == group]
        best = max(self.whole_score(user, p) for p in sounds)
        for tail in self.context_tails(user, winner):
            best = max(best, *(self.whole_score(tail, p) for p in sounds))
        return best

    def context_tails(self, user: list[str], winner: dict) -> list[list[str]]:
        """Verb evidence following a configured, independently matched prefix.

        Only the remainder is compared with accepted and rival actions. The
        many phonemes in an object name must not inflate a weak verb's score.
        """
        tails = []
        contexts = winner.get("context_prefixes") or []
        if contexts:
            _, _, word_start, word_end, _ = self.score_against(user, winner["phonemes"])
            if word_end > word_start:
                for context in contexts:
                    _, prefix_score, start, end, _ = self.score_against(user, context["phonemes"])
                    # A noun's final syllable must not double as the verb:
                    # prefix evidence and answer evidence cannot overlap.
                    if prefix_score >= CONTEXT_PREFIX_MIN_SCORE and start < end <= word_start:
                        tails.append(user[end:])
        return tails

    def short_utterance_score(self, user: list[str], winner: dict) -> float:
        """Whole-word evidence for the optional short-input fallback, never an alias."""
        limit = len(winner["phonemes"]) + winner["short_utterance_policy"]["max_extra_phonemes"]
        return max((self.whole_score(u, winner["phonemes"])
                    for u in [user, *self.context_tails(user, winner)] if 0 < len(u) <= limit), default=-1.0)

    def _veto_by_other_action(
        self,
        result: MatchResult,
        user: list[str],
        candidates: list[dict],
        winner: dict,
    ) -> None:
        """Take the pass back when another ACTION explains this better.

        The decision above is left exactly as it was - same winner, same
        score, same window, same alignment. This can only refuse it, and
        it never promotes a rival: the rivals are not answers to the
        question that was asked, and confirming one would answer a
        question nobody asked. There is no second ranking pass.

        Why it is needed: substring scoring may start its window
        anywhere, so a short answer whose phonemes are the tail of a
        longer word passes on the longer word (들기 inside 흔들기, 0.997).
        Whole-sequence scoring sees the difference, but as a THRESHOLD it
        creates the mirror failure - the long word passing on the short
        one - so it is used here only as a comparison between two
        readings of the same speech: if a word this question does not
        accept fits everything the child said better than anything it
        does accept, the pass is not earned.

        The anchor is what the ACTION that won is worth, read whole: the
        winner, the aliases of its group the question accepted, AND the
        aliases it did not ask for, which the build embedded for exactly
        this. A lesson may ask for 두드려 alone while the child says
        두드리기 - the same instruction - and without the unrequested
        alias in the anchor the winner is measured on a form nobody said
        and any passing rival takes the answer away. Another accepted
        action never raises the anchor: a good score for a different
        instruction is not evidence that this one was said, and letting
        it in would defend a poor winner with the wrong word.

        Skipped rivals, both decided by catalog data rather than by any
        word this code knows:

          * an action the question already accepts - asking for 흔들기
            and hearing 흔들어 is a right answer, not a competitor;
          * a rival whose phonemes an accepted answer already has, which
            is the same sound arriving under another name.

        Ties never veto: equal scores are not evidence that the other
        action was meant.
        """
        # Read before the early return, so metadata a caller built by
        # hand cannot switch the protection off by being unreadable.
        own_group, alternatives = validate_contrast_metadata(winner)
        if not result.passed or not alternatives:
            return
        accepted_groups = {check_action_group(c.get("action_group"), c)
                           for c in candidates}
        accepted_groups.discard(None)
        accepted_sounds = {tuple(c.get("phonemes") or ()) for c in candidates}
        best_accepted = self.accepted_whole_score(user, candidates, winner)
        tails = self.context_tails(user, winner)
        margin = winner.get("contrast_margin", 0.0)

        challenger: tuple[str, float] | None = None
        for text, phonemes, group in alternatives:
            if group in accepted_groups or tuple(phonemes) in accepted_sounds:
                continue
            score = self.whole_score(user, phonemes)
            for tail in tails:
                score = max(score, self.whole_score(tail, phonemes))
            if score > best_accepted + margin and (challenger is None
                                          or score > challenger[1]):
                challenger = (text, score)

        if challenger is None:
            return
        result.passed = False
        result.competitor_rejected = True
        result.competitor_reason = OTHER_ACTION_SCORED_HIGHER
        result.competitor_text = challenger[0]
        result.competitor_score = challenger[1]
        result.accepted_whole_score = best_accepted
