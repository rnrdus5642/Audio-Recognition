# Changelog

## 0.9.1-preview.4 — 2026-11-18

- Sentis 1에 `CreatePreparedAsync`를 추가했습니다. 외부 모델 파일은 백그라운드에서 읽고, GPU 생성·워밍업은 Unity 스레드에서 진행합니다. 준비가 끝난 인식기를 재사용하세요.
- 추론 스케줄을 프레임당 최대 128개 레이어·CPU 스케줄 시간 약 2ms로 조정했습니다. 이전 64개 제한으로 30fps에서 0.5초 입력 간격보다 늦게 완료되는 문제를 줄입니다. 단일 레이어 실행 시간과 GPU 초기 업로드는 이 시간 상한으로 나눌 수 없습니다.
- 모델·정답 목록·후처리 v5·연속 확인 규칙·오디오 창의 순서는 변경하지 않았습니다. 이번 통합 대상은 Flower이며 Unity 6/Sentis 2에는 적용하지 않습니다.

## 0.9.1-preview.3 — 2026-11-17

- `PronunciationListener.ListenFrom(AudioWindowBuffer)`로 앱이 유지하는 마이크의 PCM을 받습니다. 질문마다 버퍼와 판정 상태를 초기화하며, 질문 종료가 외부 마이크를 끄지 않습니다.
- 기존 `Listen()`의 마이크 소유 방식과 모델·정답 정책·비동기 GPU 추론은 유지합니다. 외부 입력 경로는 화분 프로젝트에 먼저 연결합니다.

## 0.9.1-preview.2 — 2026-11-12

- Sentis 1.x의 실시간 추론을 프레임별 64개 레이어/예약 시간 1ms로 분산하고 GPU 결과를 비동기로 읽습니다. 기존 동기 API와 시작 시 Warmup은 유지합니다.
- `IAsyncPhonemeRecognizer`, `RecognitionResult`, `PronunciationSession.TryAdvance`를 추가합니다. 리스너는 대기 중에도 마이크·제한 시간을 확인하고 완료된 오디오 창을 순서대로 처리합니다.
- 문항 전환·정지 후 늦은 결과를 버리고 진행 중 GPU 작업의 해제를 완료 이후로 미룹니다. 읽기/추론 실패 시 청취를 중단해 게임의 기존 입력 오류·재시도 경로로 전환하고, 프로젝트 어댑터가 다음 준비에서 실패한 모델을 재생성할 수 있도록 `IsFaulted`를 제공합니다.
- 오디오 유효성·순서·취소·재진입·시간 초과·실패 시 종료에 대한 검사를 추가합니다. 모델, v5 점수·정답 임계값, 2.5초 창/0.5초 간격, PNG 저장 코드는 변경하지 않습니다.

## 0.9.0-preview.1 — 2026-10-31

- 최종 3에포크 모델로 갱신하고 새 Release·SHA-256을 고정합니다. 문서 공용 폴더에 버전별 파일을 설치해 기존 2에포크 앱과 공존합니다.
- 웹의 `orthographic-word-context-v5`를 C#에 이식합니다. 문장 중간의 단어와 명시한 조사·서술형·인용 표현을 개별 후보와 비교합니다.
- 문장 중간 한 음절의 발음 일치, 동작어 대비, 부정 표현, 동음 후보 동점을 제한하고 서로 떨어진 정확한 정답이 여러 개면 마지막 단어를 선택합니다.
- 원본 인식 문장·기본 0.85 기준·2회 확인·2.5초 창·0.5초 간격을 유지합니다. 새로운 선택 이유와 비교 토큰 위치를 진단 결과에 포함합니다.
- 기존 v4 프로젝트 정답 JSON을 그대로 읽습니다. 정답 목록과 장면은 패키지 갱신으로 변경하지 않습니다.
- 웹의 11,804개 사례·98,494회 검사에 대한 C# 비교 자료와 문맥·합성어·기존 설정 호환 테스트를 갱신합니다.

## 0.8.9-preview.5 — 2026-10-30

- Windows Player 시작 씬에서 문서 공용 모델을 자동 확인·다운로드·검증한 뒤 원래 게임 씬을 로드합니다.
- 정상 설치 기록은 파일 메타데이터만 확인해 재사용하며 전체 해시는 다시 계산하지 않습니다.
- 설치 진행률, 취소, 실패 재시도, 종료를 표시합니다. Editor와 Player는 동일 설치기를 사용합니다.
- 초기 세팅이 빌드 시작 씬과 게임 씬 설정을 구성하며, 시작 씬 누락/지원하지 않는 빌드 대상은 빌드 오류로 안내합니다.

## 0.8.8-preview.7 — 2026-10-30

- Show initial setup progress in a centered Unity progress dialog, including the current download/verification phase, percentage and cancellation. Close it on completion, failure, cancellation, Play entry or domain reload.
- Show inference-engine installation as an indeterminate phase; UPM does not provide a download percentage. Setup still starts immediately from the menu without a settings panel.

## 0.8.8-preview.5 ? 2026-10-30

- Use one verified HuBERT model in the current Windows user Documents folder across projects. Initial setup downloads the published model when needed; no manual registration or project-local lookup.
- Run initial setup immediately from the menu without a separate panel; report progress through Unity Progress.
- Provide three unified menus: initial setup, speech testing and project-owned stage answers.
- Stream and verify downloads before atomic install; serialize concurrent installers and cancel stalled response reads. Store a verification receipt so runtime avoids hashing 3.85 GB on the game thread.
- Preserve web v4 matching, GPUCompute and project-specific 82/76/41 expressions. Retain unsaved answer edits through reload with Undo and close confirmation.

## 0.8.6 — 2026-09-12 (local distribution)

- Lower the music profile's 북채 threshold from 0.78 to 0.74; retain its whole-utterance floor of 0.65 and two-frame confirmation.
- Adjust 잡아 acoustic recovery to a minimum target log probability of -14 and maximum raw-transcript log gap of 13. Retain the 작아 contrast guard, phoneme/length limits, ordinary thresholds and two fresh confirming frames.
- Bundle profile `music_demo_2026_09_12` for both fresh installations and existing music integrations. Recognition code, model files, the other ten word policies and pronunciation guidance retain their existing behavior.
- 들어, 놓아, 놔 and 내려놔 still have known recognition failures. This update does not resolve the model's missing 놔 output token or loosen their competing-action checks.

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
