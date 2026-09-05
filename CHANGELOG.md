# 변경 기록

## Unreleased — 2026-09-06

이 항목은 아동 파인튜닝과 기존 음절 피드백을 포함한 최신 main 위의 개발 소스 변경이다.
새 배포 태그는 만들지 않았으며 패키지의 `version`은 여전히 `0.7.0`이다. 아래의 결과 데이터 계약 버전 2와
패키지 버전은 서로 다른 값이다.

### 웹 테스트 UI

- 실시간 마이크 화면 하나만 유지한다. 별도의 연속 청취·파일 평가·ASR 단독·녹음 저장·
  정답 빌드 탭과 웹 빌드 도우미를 제거했다. 파일 평가·녹음·빌드용 CLI는 유지한다.
- `targets.json` 없이 즉석 정답 입력으로 실행한다. 웹 실행 옵션 `--targets`,
  `--recordings-dir`은 제거했다.
- 준비 → 청취 → 동일 후보 연속 통과 → 정답 확정/녹음 중단 → 재시작 흐름을 사용한다.
  창 기본 2.5초, 입력 전달 0.5초, 연속 확인 기본 2회를 유지한다.
- 확정 프레임의 쉬운 한글 설명을 먼저 보여 준다. 기존 일치·교체·누락·추가 IPA 표와
  음소 위치는 접힌 상세 보기에 유지하고, 결과 JSON도 함께 제공한다.
- 기존 단어별 임계값·아동 프로필·모델 오류표를 보존했다. 웹의 음절 안내는 모델이 자주
  혼동하는 차이를 보류하며, 공통 JSON에는 원본 음소 차이를 그대로 남긴다.

### Unity·공통 결과 계약

- `PronunciationFeedback`에 확정 프레임의 점수·거리·임계값·정답/인식 IPA·정렬·차이 개수·
  프레임/연속 횟수와 `KoreanFeedback`을 제공한다. JSON 키는 snake_case,
  `schema_version`은 2다.
- `StreamingHit.Feedback`, Listener/TestBench의 `OnConfirmedDetailed`와
  `LastConfirmation`으로 결과를 받는다. 기존 `OnConfirmed(string, float)`는 유지한다.
- 기존 `PronunciationSession`에도 `frame.Feedback`과 `session.LastConfirmation`을
  연결했다. 원본 피드백의 생성으로 추가 ASR 추론은 발생하지 않는다.
- 설명은 채점에 사용한 정렬을 그대로 활용한다. 재인식·재채점·외부 API 호출은 하지 않는다.
  예: `s→t`는 “사→다”, `s→tʰ`는 “사→타”, `과`의 `w` 누락은 “과→가”.
- 표기와 정답 IPA가 정확히 대응할 때만 글자를 지목한다. 음운 변화·삽입·모호한 소리 등은
  무리하게 글자로 바꾸지 않으며, 한글 예시는 실제 발음의 확정 진단이나 ASR 전사가 아니다.
- 현재 웹은 Python, Unity는 C#/Sentis 경로다. 결과 형식·설명은 공통 벡터로 검증하지만
  전체 실행 모듈이 통합된 것은 아니다. C# Session은 이미 구현되어 있으며,
  웹의 C# 호스트 연결·공유 번들은 [별도 설계](docs/SHARED_RUNTIME_ARCHITECTURE.md)에 남아 있다.
  모델 오류표 기반 판단 보류는 기존 Python 화면 정책이며 C# 기본 DTO에는 적용하지 않는다.

### 환경·검증·문서

- Windows 한국어 처리의 구형 `eunjeon` 의존성을 `mecab-ko`와 POS 어댑터로 교체했다.
  실행 중 pip 설치를 하지 않는다. 기본 경로의 mecab-free 한국어 규칙과 IPA 매핑을
  유지하며, 어댑터는 별도 g2pkk 비교 경로에서만 사용한다.
- Python 170개, .NET C# 28개, Unity 6000.3.18f1 PlayMode 29개 테스트를 통과했다.
  공유 결과 계약·24개 한글 설명 사례·전체 한글 11,172음절 매핑을 검사한다.
- 웹 모델 예열과 합성 확정 결과의 HTML/JSON을 확인했다. 실제 마이크로 새 설명을 평가하거나
  동일 PCM을 웹/Unity 전체 경로에 넣어 대조한 검증은 별도다.
- 저장소 위치를 통합하면서 학습 산출물과 모델을 보존하고 주요 파일의 SHA-256을 대조했다.
  이전한 best 아동 체크포인트의 CUDA 추론도 확인했다.
- README, [발음 설명 API](unity/Packages/com.domicube.phoneme-matching/Documentation~/pronunciation-feedback.md),
  공통 런타임 설계, 새 언어 추가 가이드에 구현 상태·한계·검증 방법을 반영했다.
