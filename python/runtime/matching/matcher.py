"""High-level phoneme matching.

Implements weighted Levenshtein edit distance over IPA phoneme sequences,
normalized to a [0, 1] similarity score. The score is the primary signal
used by the application to decide pass / retry.

This module is kept in pure Python with no exotic dependencies so that
porting to C# for Unity is mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

from .confusion_matrix import ConfusionMatrix


@dataclass
class MatchResult:
    """Single match outcome."""

    score: float                # [0.0, 1.0], higher = closer
    distance: float             # raw weighted edit distance
    target_id: str | None       # matched target answer id (if any)
    target_text: str | None     # matched target text (if any)
    target_phonemes: list[str] | None
    user_phonemes: list[str]
    passed: bool                # score >= target's threshold
    # Substring matching: the slice of user_phonemes that actually matched
    # the target. For exact mode this is [0, len(user_phonemes)].
    window_start: int = 0
    window_end: int = 0
    # Per-step alignment (user_ph, target_ph, op) within the window.
    alignment: list[tuple[str, str, str]] | None = None


def substring_edit_distance(
    user: list[str],
    target: list[str],
    matrix: ConfusionMatrix,
) -> tuple[float, int, int, list[tuple[str, str, str]]]:
    """Find the best substring of `user` that matches `target`.

    Unlike full edit distance, this lets `user` carry arbitrary noise
    before and/or after the actual target word - any prefix/suffix is
    skipped for free. Crucial when the ASR captures background noise
    or coughs as extra phonemes around the real utterance.

    Returns:
        (distance, window_start, window_end, ops)

    where:
        * `user[window_start:window_end]` is the best-matching substring
        * `distance` is the weighted edit distance from that substring
          to the full target
        * `ops` is the per-step alignment trace
          [(user_phoneme | "", target_phoneme | "", op)] with
          op in {"match", "sub", "ins", "del"}

    Algorithm: standard DP, but `dp[0][j] = 0` for all j (matching
    an empty target prefix is free at any starting position in user),
    and we minimise `dp[m][j]` over j (free to stop anywhere in user).
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

    # Free start: target prefix of length 0 has cost 0 at any user position
    for j in range(n + 1):
        dp[0][j] = 0.0

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

    # Find the best ending position (free end)
    best_j = 0
    best_dist = dp[m][0]
    for j in range(1, n + 1):
        if dp[m][j] < best_dist:
            best_dist = dp[m][j]
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
    """Match a user phoneme sequence against a set of candidate targets
    (e.g., the answers in one segment) and return the best one.

    Two scoring modes:

    * `mode="substring"` (default): the target must appear as a
      best-matching window inside the user phoneme sequence. Prefix and
      suffix of the user output (typically ASR noise) are skipped for
      free. Use when the recogniser tends to include incidental
      phonemes from coughs, background, breath, etc.

    * `mode="exact"`: full sequence-vs-sequence weighted Levenshtein
      with a length-mismatch penalty. Use when you trust the user
      sequence to contain exactly the target word with no surrounding
      noise.
    """

    def __init__(
        self,
        matrix: ConfusionMatrix,
        mode: str = "substring",
    ) -> None:
        if mode not in ("substring", "exact"):
            raise ValueError(
                f"mode must be 'substring' or 'exact', got {mode!r}"
            )
        self.matrix = matrix
        self.mode = mode

    def score_against(
        self,
        user: list[str],
        target: list[str],
    ) -> tuple[float, float, int, int, list[tuple[str, str, str]]]:
        """Return (distance, score, window_start, window_end, alignment)
        for a single target.
        """
        if self.mode == "substring":
            d, ws, we, ops = substring_edit_distance(
                user, target, self.matrix
            )
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
            #   window_ratio >= 0.5 -> no penalty (legit partial drop
            #                                       like 종성 탈락)
            #   window_ratio == 0   -> mult 0    (nothing matched)
            #   linear interpolation in between
            target_len = len(target)
            window_len = we - ws
            if target_len > 0 and window_len < 0.5 * target_len:
                mult = (window_len / target_len) / 0.5
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
    ) -> MatchResult:
        """Find the candidate with the highest similarity score.

        Each candidate dict must have keys:
            id (str), text (str), phonemes (list[str]), threshold (float)
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
        best = candidates[0]
        best_ws = 0
        best_we = len(user)
        best_ops: list[tuple[str, str, str]] = []
        for cand in candidates:
            d, s, ws, we, ops = self.score_against(
                user, cand["phonemes"]
            )
            if s > best_score:
                best_score = s
                best_dist = d
                best = cand
                best_ws = ws
                best_we = we
                best_ops = ops

        return MatchResult(
            score=best_score,
            distance=best_dist,
            target_id=best["id"],
            target_text=best["text"],
            target_phonemes=best["phonemes"],
            user_phonemes=user,
            passed=best_score >= float(best.get("threshold", 0.6)),
            window_start=best_ws,
            window_end=best_we,
            alignment=best_ops,
        )
