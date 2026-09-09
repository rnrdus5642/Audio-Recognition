using System;
using System.Collections.Generic;

namespace DomiCube.PhonemeMatching
{
    public enum AlignOp
    {
        Match,
        Substitute,
        Insert,
        Delete
    }

    /// <summary>One step of the alignment between user and target.</summary>
    public readonly struct AlignStep
    {
        public readonly string UserPhoneme;   // empty for Delete
        public readonly string TargetPhoneme; // empty for Insert
        public readonly AlignOp Op;

        public AlignStep(string userPhoneme, string targetPhoneme, AlignOp op)
        {
            UserPhoneme = userPhoneme;
            TargetPhoneme = targetPhoneme;
            Op = op;
        }
    }

    /// <summary>Outcome of matching a user sequence against candidates.</summary>
    public sealed class MatchResult
    {
        public double Score;
        public double Distance;
        public double Threshold;
        public string TargetId;
        public string TargetText;
        public List<string> TargetPhonemes;
        public List<string> UserPhonemes;
        // Threshold met AND not vetoed by another action. Do not derive
        // this flag from Score >= Threshold in callers.
        public bool Passed;

        /// <summary>
        /// Slice of <see cref="UserPhonemes"/> that matched the target.
        /// Indices are relative to the caller's list even when a context
        /// limit made scoring look at a shorter copy.
        /// </summary>
        public int WindowStart;
        public int WindowEnd;

        public List<AlignStep> Alignment = new List<AlignStep>();

        /// <summary>
        /// Set when a passing answer was refused because a DIFFERENT
        /// action explains the same speech better. False for every
        /// answer that ships no contrast set, which is every answer in a
        /// catalog built before contrasts existed.
        /// </summary>
        public bool CompetitorRejected;

        /// <summary>Why, or empty. See <see cref="Matcher.OtherActionScoredHigher"/>.</summary>
        public string CompetitorReason = string.Empty;

        /// <summary>The action that fitted better, or null.</summary>
        public string CompetitorText;

        /// <summary>Its whole-sequence score, or null.</summary>
        public double? CompetitorScore;

        /// <summary>
        /// The best whole-sequence score within the winner's action group,
        /// including embedded aliases that were not requested, or null.
        /// Reported only on veto, beside the competitor. Comparison-only:
        /// neither a probability nor the answer's thresholded Score.
        /// </summary>
        public double? AcceptedWholeScore;
        public bool WholeRejected;
        public bool UsedShortUtterancePolicy;
        public AcousticEvidence AcousticEvidence;
    }

    public enum MatchMode
    {
        /// <summary>
        /// The target must appear as a best-matching window inside the
        /// user sequence; surrounding phonemes are skipped at a small
        /// per-phoneme cost.
        /// </summary>
        Substring,

        /// <summary>Full sequence-vs-sequence weighted Levenshtein.</summary>
        Exact
    }

    /// <summary>
    /// Weighted phoneme matching, mirroring
    /// python/runtime/matching/matcher.py.
    ///
    /// Kept free of Unity types and exotic dependencies so it can be
    /// unit-tested outside the editor and stays comparable, line for
    /// line, with the Python reference the thresholds were tuned against.
    /// </summary>
    public sealed class Matcher
    {
        /// <summary>
        /// Extra phonemes kept beyond ContextMult * target length, so a
        /// target sitting at the very end of the window is not clipped.
        /// </summary>
        public const int ContextPad = 3;

        /// <summary>
        /// <see cref="MatchResult.CompetitorReason"/> when the pass was
        /// taken back because another action fitted the speech better.
        /// </summary>
        public const string OtherActionScoredHigher = "other_action_scored_higher";
        public const double ContextPrefixMinScore = 0.80;

        public ConfusionMatrix Matrix { get; }
        public MatchMode Mode { get; }
        public double SkipCost { get; }
        public double Coverage { get; }
        public double? ContextMult { get; }

        public Matcher(
            ConfusionMatrix matrix,
            MatchMode mode = MatchMode.Substring,
            double? skipCost = null,
            double coverage = 0.5,
            double? contextMult = null)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            if (coverage <= 0.0 || coverage > 1.0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(coverage), coverage, "coverage must be in (0, 1]");
            }

            Matrix = matrix;
            Mode = mode;
            SkipCost = skipCost ?? matrix.SkipCost;
            Coverage = coverage;
            ContextMult = contextMult;
        }

        /// <summary>
        /// Matcher tuned for continuous listening, reading the matrix's
        /// streaming profile. Batch scoring keeps a lenient coverage
        /// threshold so a partly recognised word still passes; that
        /// leniency lets fragments of other words clear the bar during
        /// free speech, so streaming raises it and bounds how much
        /// context is scored.
        /// </summary>
        public static Matcher ForStreaming(ConfusionMatrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException(nameof(matrix));
            }

            var p = matrix.Streaming;
            return new Matcher(
                matrix,
                MatchMode.Substring,
                p.SkipCost ?? matrix.SkipCost,
                p.Coverage ?? 0.5,
                p.ContextMult.HasValue && p.ContextMult.Value > 0.0
                    ? p.ContextMult
                    : null);
        }

        /// <summary>
        /// The part of <paramref name="user"/> that is actually scored and
        /// how many leading phonemes were dropped.
        ///
        /// The limit is coupled to how long a confirmation takes: a target
        /// only scores while it sits inside this slice, so if the slice
        /// holds less speech than (consecutive frames x hop) seconds, a
        /// user who keeps talking can never accumulate a streak - the
        /// answer slides out of context first. Korean runs about 10
        /// phonemes per second, so budget accordingly when tuning.
        /// </summary>
        public void ContextSlice(
            IReadOnlyList<string> user, IReadOnlyList<string> target,
            out int start, out int dropped)
        {
            start = 0;
            dropped = 0;
            if (!ContextMult.HasValue || target == null || target.Count == 0)
            {
                return;
            }

            int keep = (int)(ContextMult.Value * target.Count) + ContextPad;
            if (user.Count > keep)
            {
                dropped = user.Count - keep;
                start = dropped;
            }
        }

        /// <summary>
        /// Best-matching window of <paramref name="user"/> against the
        /// full target.
        ///
        /// Prefix and suffix outside the window cost
        /// <paramref name="skipCost"/> per phoneme. A cost of zero is the
        /// textbook semi-global alignment, but it makes the score
        /// monotonically non-decreasing in user length: appending
        /// phonemes only ever adds candidate windows, so a long enough
        /// utterance clears any threshold. Keep it well below
        /// <see cref="ConfusionMatrix.InsCost"/> so genuine surrounding
        /// noise is still cheaper to skip than to align.
        /// </summary>
        public static double SubstringEditDistance(
            IReadOnlyList<string> user,
            IReadOnlyList<string> target,
            ConfusionMatrix matrix,
            double skipCost,
            out int windowStart,
            out int windowEnd,
            List<AlignStep> ops)
        {
            ops?.Clear();
            int n = user.Count;
            int m = target.Count;
            windowStart = 0;
            windowEnd = 0;

            if (m == 0)
            {
                return 0.0;
            }

            if (n == 0)
            {
                double total = 0.0;
                for (int i = 0; i < m; i++)
                {
                    total += matrix.DelCost(target[i]);
                    ops?.Add(new AlignStep(string.Empty, target[i],
                        AlignOp.Delete));
                }

                return total;
            }

            var dp = new double[m + 1, n + 1];
            var parentI = new int[m + 1, n + 1];
            var parentJ = new int[m + 1, n + 1];
            var parentOp = new AlignOp[m + 1, n + 1];
            var hasParent = new bool[m + 1, n + 1];

            // Window start: an empty target prefix costs only the skipped
            // user prefix.
            for (int j = 0; j <= n; j++)
            {
                dp[0, j] = j * skipCost;
            }

            // Target prefix against an empty user prefix: delete them all.
            for (int i = 1; i <= m; i++)
            {
                dp[i, 0] = dp[i - 1, 0] + matrix.DelCost(target[i - 1]);
                parentI[i, 0] = i - 1;
                parentJ[i, 0] = 0;
                parentOp[i, 0] = AlignOp.Delete;
                hasParent[i, 0] = true;
            }

            for (int i = 1; i <= m; i++)
            {
                string ti = target[i - 1];
                for (int j = 1; j <= n; j++)
                {
                    string uj = user[j - 1];
                    var subOp = string.Equals(ti, uj, StringComparison.Ordinal)
                        ? AlignOp.Match
                        : AlignOp.Substitute;

                    double subVal = dp[i - 1, j - 1] + matrix.SubCost(ti, uj);
                    double insVal = dp[i, j - 1] + matrix.InsCost(uj);
                    double delVal = dp[i - 1, j] + matrix.DelCost(ti);

                    // Tie-break match/sub > del > ins so the backtrack
                    // tends toward a contiguous window.
                    double best = subVal;
                    int pi = i - 1, pj = j - 1;
                    var op = subOp;

                    if (delVal < best)
                    {
                        best = delVal;
                        pi = i - 1;
                        pj = j;
                        op = AlignOp.Delete;
                    }

                    if (insVal < best)
                    {
                        best = insVal;
                        pi = i;
                        pj = j - 1;
                        op = AlignOp.Insert;
                    }

                    dp[i, j] = best;
                    parentI[i, j] = pi;
                    parentJ[i, j] = pj;
                    parentOp[i, j] = op;
                    hasParent[i, j] = true;
                }
            }

            // Best ending position, paying for the skipped user suffix.
            int bestJ = 0;
            double bestDist = dp[m, 0] + n * skipCost;
            for (int j = 1; j <= n; j++)
            {
                double total = dp[m, j] + (n - j) * skipCost;
                if (total < bestDist)
                {
                    bestDist = total;
                    bestJ = j;
                }
            }

            // Backtrack to find where the window started.
            int ci = m, cj = bestJ;
            var trace = ops != null ? new List<AlignStep>() : null;
            while (ci > 0)
            {
                if (!hasParent[ci, cj])
                {
                    break;
                }

                int ni = parentI[ci, cj];
                int nj = parentJ[ci, cj];
                var op = parentOp[ci, cj];

                if (trace != null)
                {
                    switch (op)
                    {
                        case AlignOp.Match:
                        case AlignOp.Substitute:
                            trace.Add(new AlignStep(
                                user[cj - 1], target[ci - 1], op));
                            break;
                        case AlignOp.Insert:
                            trace.Add(new AlignStep(
                                user[cj - 1], string.Empty, op));
                            break;
                        default:
                            trace.Add(new AlignStep(
                                string.Empty, target[ci - 1], op));
                            break;
                    }
                }

                ci = ni;
                cj = nj;
            }

            windowStart = cj;
            windowEnd = bestJ;

            if (trace != null)
            {
                trace.Reverse();
                ops.AddRange(trace);
            }

            return bestDist;
        }

        /// <summary>Full weighted Levenshtein (exact mode).</summary>
        public static double WeightedEditDistance(
            IReadOnlyList<string> user,
            IReadOnlyList<string> target,
            ConfusionMatrix matrix)
        {
            int n = user.Count;
            int m = target.Count;

            if (n == 0 && m == 0)
            {
                return 0.0;
            }

            if (n == 0)
            {
                double d = 0.0;
                for (int j = 0; j < m; j++)
                {
                    d += matrix.DelCost(target[j]);
                }

                return d;
            }

            if (m == 0)
            {
                double d = 0.0;
                for (int i = 0; i < n; i++)
                {
                    d += matrix.InsCost(user[i]);
                }

                return d;
            }

            var dp = new double[n + 1, m + 1];
            for (int i = 1; i <= n; i++)
            {
                dp[i, 0] = dp[i - 1, 0] + matrix.InsCost(user[i - 1]);
            }

            for (int j = 1; j <= m; j++)
            {
                dp[0, j] = dp[0, j - 1] + matrix.DelCost(target[j - 1]);
            }

            for (int i = 1; i <= n; i++)
            {
                for (int j = 1; j <= m; j++)
                {
                    double sub = dp[i - 1, j - 1]
                        + matrix.SubCost(user[i - 1], target[j - 1]);
                    double ins = dp[i - 1, j] + matrix.InsCost(user[i - 1]);
                    double del = dp[i, j - 1] + matrix.DelCost(target[j - 1]);
                    dp[i, j] = Math.Min(sub, Math.Min(ins, del));
                }
            }

            return dp[n, m];
        }

        /// <summary>
        /// Raw distance to a [0, 1] similarity, normalised by target
        /// length, with a penalty when the user produced wildly more or
        /// less than the target. Exact mode only.
        /// </summary>
        public static double SimilarityScore(
            double distance, int targetLength, int? userLength = null)
        {
            if (targetLength <= 0)
            {
                return distance > 0 ? 0.0 : 1.0;
            }

            double raw = 1.0 - distance / targetLength;
            double score = Math.Max(0.0, Math.Min(1.0, raw));

            if (!userLength.HasValue || score == 0.0)
            {
                return score;
            }

            int u = userLength.Value;
            if (Math.Abs(u - targetLength) <= 1)
            {
                return score;
            }

            double ratio = (double)u / targetLength;
            if (ratio < 0.5)
            {
                score *= 0.20 + ratio;
            }
            else if (ratio > 2.0)
            {
                score *= 0.5 + 1.0 / ratio;
            }

            return Math.Max(0.0, Math.Min(1.0, score));
        }

        /// <summary>Score one target. Window indices are relative to
        /// <paramref name="user"/>.</summary>
        public double ScoreAgainst(
            IReadOnlyList<string> user,
            IReadOnlyList<string> target,
            out double score,
            out int windowStart,
            out int windowEnd,
            List<AlignStep> alignment = null)
        {
            if (Mode == MatchMode.Substring)
            {
                ContextSlice(user, target, out int start, out int dropped);
                IReadOnlyList<string> scoped = start == 0
                    ? user
                    : Slice(user, start);

                double d = SubstringEditDistance(
                    scoped, target, Matrix, SkipCost,
                    out windowStart, out windowEnd, alignment);
                windowStart += dropped;
                windowEnd += dropped;

                int targetLen = target.Count;
                double baseScore = Math.Max(
                    0.0, 1.0 - d / Math.Max(targetLen, 1));

                // Coverage penalty. Deletion costs are small so that coda
                // dropping stays cheap, which also lets a 1-phoneme output
                // score surprisingly well against a long target. Penalise
                // only when the matched window is meaningfully shorter
                // than the target.
                int windowLen = windowEnd - windowStart;
                if (targetLen > 0 && windowLen < Coverage * targetLen)
                {
                    baseScore *= windowLen / (double)targetLen / Coverage;
                }

                score = Math.Max(0.0, Math.Min(1.0, baseScore));
                return d;
            }

            double dist = WeightedEditDistance(user, target, Matrix);
            score = SimilarityScore(dist, target.Count, user.Count);
            windowStart = 0;
            windowEnd = user.Count;
            alignment?.Clear();
            return dist;
        }

        private static List<string> Slice(IReadOnlyList<string> src, int start)
        {
            var result = new List<string>(src.Count - start);
            for (int i = start; i < src.Count; i++)
            {
                result.Add(src[i]);
            }

            return result;
        }

        /// <summary>
        /// Highest-scoring candidate that passes its own threshold. If none
        /// pass, returns the highest-scoring failure for progress display.
        /// Ties within the same pass/fail group keep the earlier candidate.
        /// Only the supplied answers are considered; a higher-scoring failure
        /// must not hide an acceptable answer with a lower pass mark.
        ///
        /// A candidate may carry an action group and the contrast set the
        /// build embedded for it. Those change nothing about who wins or
        /// what they score - they can only take a pass away afterwards,
        /// see <see cref="VetoByOtherAction"/>.
        /// </summary>
        public MatchResult BestMatch(
            IReadOnlyList<string> user, IReadOnlyList<Answer> candidates,
            IReadOnlyDictionary<string, AcousticEvidence> acousticEvidence = null)
        {
            var userList = new List<string>(user ?? new List<string>());

            if (candidates == null || candidates.Count == 0)
            {
                return new MatchResult
                {
                    Score = 0.0,
                    Distance = 0.0,
                    UserPhonemes = userList,
                    Passed = false
                };
            }

            double bestScore = -1.0;
            bool bestPassed = false;
            double bestDistance = double.PositiveInfinity;
            Answer best = candidates[0];
            int bestWs = 0, bestWe = userList.Count;
            var bestOps = new List<AlignStep>();
            double? bestWhole = null;
            bool bestWholeRejected = false;
            double bestThreshold = best.Threshold;
            bool bestShort = false;
            AcousticEvidence bestAcoustic = null;

            foreach (var cand in candidates)
            {
                AnswerPolicy.Validate(cand);
                var ops = new List<AlignStep>();
                double d = ScoreAgainst(
                    userList, cand.Phonemes, out double s,
                    out int ws, out int we, ops);

                double threshold = cand.Threshold;
                bool passed = s >= threshold;
                bool usedShort = false;
                AcousticEvidence usedAcoustic = null;
                double? whole = null;
                bool wholeRejected = false;
                if (passed && cand.MinWholeScore > 0.0)
                {
                    whole = AcceptedWholeScore(userList, candidates, cand);
                    wholeRejected = whole.Value < cand.MinWholeScore;
                    passed = !wholeRejected;
                }
                var shortPolicy = cand.ShortUtterancePolicy;
                if (!passed && shortPolicy != null && s >= shortPolicy.Threshold)
                {
                    double shortWhole = ShortUtteranceScore(userList, cand);
                    if (shortWhole >= shortPolicy.MinWholeScore)
                    {
                        passed = usedShort = true;
                        wholeRejected = false;
                        threshold = shortPolicy.Threshold;
                        whole = shortWhole;
                    }
                }
                if (!passed && acousticEvidence != null
                    && acousticEvidence.TryGetValue(cand.Text, out var evidence)
                    && evidence != null && evidence.Permits(cand, userList.Count, s))
                {
                    whole = AcceptedWholeScore(userList, candidates, cand);
                    if (whole >= cand.MinWholeScore)
                    {
                        passed = true;
                        wholeRejected = false;
                        threshold = cand.AcousticKeywordPolicy.MinScore;
                        usedAcoustic = evidence;
                    }
                }
                if ((passed && !bestPassed)
                    || (passed == bestPassed && s > bestScore))
                {
                    bestScore = s;
                    bestPassed = passed;
                    bestDistance = d;
                    best = cand;
                    bestWs = ws;
                    bestWe = we;
                    bestOps = ops;
                    bestWhole = whole;
                    bestWholeRejected = wholeRejected;
                    bestThreshold = threshold;
                    bestShort = usedShort;
                    bestAcoustic = usedAcoustic;
                }
            }

            var result = new MatchResult
            {
                Score = bestScore,
                Distance = bestDistance,
                Threshold = bestThreshold,
                TargetId = best.Id,
                TargetText = best.Text,
                TargetPhonemes = best.Phonemes,
                UserPhonemes = userList,
                Passed = bestPassed,
                WindowStart = bestWs,
                WindowEnd = bestWe,
                Alignment = bestOps,
                AcceptedWholeScore = bestWhole,
                WholeRejected = bestWholeRejected,
                UsedShortUtterancePolicy = bestShort,
                AcousticEvidence = bestAcoustic
            };

            VetoByOtherAction(result, userList, candidates, best);
            return result;
        }

        /// <summary>
        /// <paramref name="user"/> against <paramref name="target"/>, whole
        /// against whole: no window and no context limit, so every phoneme
        /// the recogniser produced is paid for. The existing exact-mode
        /// arithmetic, used only to compare an answer with the actions it
        /// must be told apart from. It accepts nothing and is never
        /// reported as an answer's score.
        /// </summary>
        public double WholeScore(
            IReadOnlyList<string> user, IReadOnlyList<string> target)
        {
            double distance = WeightedEditDistance(user, target, Matrix);
            return SimilarityScore(distance, target.Count, user.Count);
        }

        /// <summary>The whole utterance against the answer and its own group's aliases.</summary>
        public double AcceptedWholeScore(IReadOnlyList<string> user,
            IReadOnlyList<Answer> candidates, Answer winner)
        {
            ActionContrast.Validate(winner);
            AnswerPolicy.Validate(winner);
            double best = WholeScore(user, winner.Phonemes);
            if (winner.ActionGroup != null)
            {
                foreach (var candidate in candidates)
                    if (candidate.ActionGroup == winner.ActionGroup)
                        best = Math.Max(best, WholeScore(user, candidate.Phonemes));
                if (winner.ContrastiveAlternatives != null)
                    foreach (var alias in winner.ContrastiveAlternatives)
                        if (alias.ActionGroup == winner.ActionGroup)
                            best = Math.Max(best, WholeScore(user, alias.Phonemes));
            }
            foreach (var tail in ContextTails(user, winner))
            {
                best = Math.Max(best, WholeScore(tail, winner.Phonemes));
                if (winner.ActionGroup == null) continue;
                foreach (var candidate in candidates)
                    if (candidate.ActionGroup == winner.ActionGroup)
                        best = Math.Max(best, WholeScore(tail, candidate.Phonemes));
                if (winner.ContrastiveAlternatives != null)
                    foreach (var alias in winner.ContrastiveAlternatives)
                        if (alias.ActionGroup == winner.ActionGroup)
                            best = Math.Max(best, WholeScore(tail, alias.Phonemes));
            }
            return best;
        }

        /// <summary>Compare the remaining verb, so a long noun cannot inflate its score.</summary>
        private List<IReadOnlyList<string>> ContextTails(IReadOnlyList<string> user, Answer winner)
        {
            var tails = new List<IReadOnlyList<string>>();
            if (winner.ContextPrefixes != null && winner.ContextPrefixes.Count > 0)
            {
                ScoreAgainst(user, winner.Phonemes, out _, out int wordStart, out int wordEnd);
                if (wordEnd > wordStart)
                    foreach (var context in winner.ContextPrefixes)
                    {
                        ScoreAgainst(user, context.Phonemes, out double prefixScore,
                            out int start, out int end);
                        // The noun's final syllable cannot also count as the verb.
                        if (prefixScore >= ContextPrefixMinScore && start < end && end <= wordStart)
                            tails.Add(Slice(user, end));
                    }
            }
            return tails;
        }

        /// <summary>Short-input fallback compares the word itself, never an alias.</summary>
        public double ShortUtteranceScore(IReadOnlyList<string> user, Answer winner)
        {
            int limit = winner.Phonemes.Count + winner.ShortUtterancePolicy.MaxExtraPhonemes;
            double best = user.Count > 0 && user.Count <= limit ? WholeScore(user, winner.Phonemes) : -1.0;
            foreach (var tail in ContextTails(user, winner))
                if (tail.Count > 0 && tail.Count <= limit)
                    best = Math.Max(best, WholeScore(tail, winner.Phonemes));
            return best;
        }

        /// <summary>
        /// Take the pass back when another ACTION explains this better.
        ///
        /// The decision above is left exactly as it was - same winner,
        /// same score, same window, same alignment. This can only refuse
        /// it, and it never promotes a rival: the rivals are not answers
        /// to the question that was asked. There is no second ranking.
        ///
        /// Substring scoring may start its window anywhere, so a short
        /// answer whose phonemes are the tail of a longer word passes on
        /// the longer word. Whole-sequence scoring sees the difference,
        /// but as a THRESHOLD it creates the mirror failure, so it is
        /// used here only to compare two readings of the same speech: if
        /// an action this question does not accept fits everything the
        /// child said better than anything it does accept, the pass was
        /// not earned. Ties never veto.
        ///
        /// The anchor is what the ACTION that won is worth, read whole:
        /// the winner, the aliases of its group the question accepted,
        /// and the aliases it did not ask for, which the build embedded
        /// for exactly this. A lesson may ask for 두드려 alone while the
        /// child says 두드리기 - the same instruction - and without the
        /// unrequested alias the winner is measured on a form nobody
        /// said. Another accepted action never raises the anchor.
        ///
        /// Both skips are decided by catalog data, never by knowing any
        /// particular word: an action the question already accepts, and a
        /// rival whose phonemes an accepted answer already has.
        /// </summary>
        private void VetoByOtherAction(
            MatchResult result,
            IReadOnlyList<string> user,
            IReadOnlyList<Answer> candidates,
            Answer winner)
        {
            // Checked before anything else, so metadata a caller built in
            // code cannot switch the protection off by being unreadable.
            ActionContrast.Validate(winner);
            var rivals = winner.ContrastiveAlternatives;
            if (!result.Passed || rivals == null || rivals.Count == 0)
            {
                return;
            }

            var acceptedGroups = new HashSet<string>(StringComparer.Ordinal);
            // The winner is always part of its own action.
            double bestAccepted = AcceptedWholeScore(user, candidates, winner);
            var tails = ContextTails(user, winner);
            foreach (var candidate in candidates)
            {
                string group = ActionContrast.CheckGroup(
                    candidate.ActionGroup, candidate.Text ?? candidate.Id);
                if (group != null)
                {
                    acceptedGroups.Add(group);
                }

            }

            string challengerText = null;
            double challengerScore = 0.0;
            foreach (var rival in rivals)
            {
                if (acceptedGroups.Contains(rival.ActionGroup)
                    || IsAlreadyAccepted(rival.Phonemes, candidates))
                {
                    continue;
                }

                double score = WholeScore(user, rival.Phonemes);
                foreach (var tail in tails)
                    score = Math.Max(score, WholeScore(tail, rival.Phonemes));
                if (score > bestAccepted + winner.ContrastMargin
                    && (challengerText == null || score > challengerScore))
                {
                    challengerText = rival.Text;
                    challengerScore = score;
                }
            }

            if (challengerText == null)
            {
                return;
            }

            result.Passed = false;
            result.CompetitorRejected = true;
            result.CompetitorReason = OtherActionScoredHigher;
            result.CompetitorText = challengerText;
            result.CompetitorScore = challengerScore;
            result.AcceptedWholeScore = bestAccepted;
        }

        /// <summary>Whether an accepted answer already has these sounds.</summary>
        private static bool IsAlreadyAccepted(
            IReadOnlyList<string> phonemes, IReadOnlyList<Answer> candidates)
        {
            foreach (var candidate in candidates)
            {
                var other = candidate.Phonemes;
                if (other == null || other.Count != phonemes.Count)
                {
                    continue;
                }

                bool same = true;
                for (int i = 0; i < other.Count && same; i++)
                {
                    same = string.Equals(other[i], phonemes[i],
                        StringComparison.Ordinal);
                }

                if (same)
                {
                    return true;
                }
            }

            return false;
        }
    }
}
