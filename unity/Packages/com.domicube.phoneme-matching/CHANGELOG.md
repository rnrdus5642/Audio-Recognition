# Changelog

## 0.8.5 — 2026-09-10 (local distribution)

- Add opt-in acoustic evidence for 잡아 only. Reuse the same Sentis inference and CTC sequence likelihoods for the target, raw decoded text and 작아; keep raw text/phonemes and two-frame confirmation intact.
- Recover four of five supplied adult recordings at the original replay timing, while rejecting both supplied 작아 recordings. Across four onset offsets, recover 15/20 positive replays and accept 0/8 negative replays; these are seven recordings, not 28 independent samples. Recording 47 still fails.
- No additional accepts in the evaluated 4,995 reference negatives, 698 existing child negatives or 180 new near-word controls. Existing false accepts remain, including early acceptance inside longer 잡아먹어/잡아당겨 sentences. Results are tuning checks, not unseen-speaker accuracy.
- Keep the other eleven answer policies and child model unchanged. Export acoustic support in local diagnostics. Withhold pronunciation criticism when acceptance depends on acoustic recovery; preserve raw differences and the presentation message 발음에 문제 없어요.
- PronunciationSession/PronunciationListener connect the evidence automatically; custom StreamingMatcher callers must supply current-frame evidence explicitly. Legacy recognizers and phoneme-only calls retain their previous decisions.

## 0.8.4 — 2026-09-10 (local distribution)

- Accept configured object prefixes before 잡아, 쳐, 때려, 연주해 and 내려놔. Prefix and verb evidence must not overlap; compare the remaining action to prevent an object name from inflating its score.
- Reduce 잡아 confirmation from three frames to two. Add calibrated short-input policies for 잡아, 때려, 연주해 and 내려놔 while retaining the normal sentence policy and competing-action checks.
- Add `OnFrameScoredDetailed`, `LastFrameScored`, recent frame history and local JSON diagnostic export with raw recognition, input levels, effective threshold and rejection reason.
- Keep the child acoustic model and reviewed pronunciation guidance. Update the bundled profile to `music_demo_2026_09_10`; ordinary catalogs without the new fields retain their previous policy.

## 0.8.3 — 2026-09-09 (local distribution)

- Show `발음에 문제 없어요.` when model review leaves no pronunciation hints to display, as requested for the presentation.
- Show filtered syllables as green `문제 없음` in the web UI; keep remaining hints for other syllables.
- Keep the internal withheld reasons, raw comparisons, answer decisions and the 0.8.2 music policy unchanged.

## 0.8.2 — 2026-09-09 (local distribution)

- Apply the revised nine-stage answer table; add 잡아, 연주해 and 내려놔, and remove 실로폰, 탬버린 and 북 as answers.
- Accept 때려 in stages 2 and 5, and 쳐 in stage 9, as well as their existing stages.
- Calibrate the new expressions using complete child recordings, synthetic word recordings and existing negative controls.
- Keep 내려놔 explicitly registered; 놔 alone does not satisfy the whole-utterance check for that longer expression.
- Retain removed object names and confusing different actions as comparison-only entries.
- Preserve the nine retained words' thresholds and confirmation counts, and the schema 3 reviewed pronunciation guidance.

## 0.8.1 — 2026-09-09 (local distribution)

- Withhold pronunciation guidance for frequent ASR confusions, unmeasured differences, insufficient evidence, and missing/mismatched model profiles.
- Bundle the child model error observations and connect them to Listener, music sessions and the editor test window.
- Share the same guidance review in Python and C#. Schema 3 adds RawKoreanFeedback and FeedbackReview; KoreanFeedback is now reviewed for display.
- Keep raw phoneme comparisons, answer scores, thresholds and confirmation timing unchanged.
- Show reviewed guidance in MusicPresentationLesson and the microphone test window.

## 0.8.0 — 2026-09-08 (local distribution)

- Bundle the music presentation policy (12 words, 9 stages), child matrix and vocabulary in package Resources.
- Add music setup, a stage selector in the microphone test window, and a ready-to-use MusicPresentationLesson.
- Add MusicPresentation.ApplyTo and MusicPresentationProfile.CreateSession for existing games.
- Reuse an installed child model; package updates carry the music policy without overwriting project JSON files.
- Include the September 7 per-word confirmation, whole-utterance and competing-action controls.
- Fix LessonExample compilation before an inference engine is installed.
- Retain existing listener events and ordinary base/child profiles.

This local artifact does not create or update a remote Git tag.
