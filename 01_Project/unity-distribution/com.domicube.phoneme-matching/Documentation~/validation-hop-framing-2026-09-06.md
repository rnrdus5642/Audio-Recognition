# hop 프레이밍 검증 — 2026-09-06

`validation-2026-09-06.md` 의 설치·실행 검증은 그대로 두고, 그 뒤에 들어간
**샘플 기준 프레임 생성**만 따로 기록한다.

## 무엇이 바뀌었나

채점 한 번의 기준이 "콜백/코루틴 한 번" 에서 **"새 샘플 8,000개(0.5초)"** 로
바뀌었다. 같은 16 kHz 모노 PCM 이면 청크를 어떻게 쪼개 넣든 같은 창이 같은
순서로 생기고 같은 샘플에서 확정된다. 한 청크에 hop 이 여러 개 들어 있으면
각각의 경계 시점 창을 순서대로 채점하고 확정된 프레임에서 멈춘다. 부분 hop
에서는 추론을 하지 않는다.

- 파이썬: `python/runtime/audio.py` 의 `HopFramer`, 웹 라이브 콜백이 사용.
- C#: `AudioWindowBuffer` 의 hop 큐(선택)와 `PronunciationSession.TryPush`.
  `PronunciationListener` 와 Editor 발음 테스트 창이 `TryPush` 를 쓰고,
  저장소의 `PronunciationTestBench` 는 `MicrophoneRollingBuffer.TryReadFrame`
  으로 프레임을 꺼내 자기 `StreamingMatcher` 로 채점한다.
- 인식·채점이 실패하면 그 세션을 버린다. 실패한 프레임 앞뒤를 이어 붙여
  "연속 2회" 로 세지 않는다.

창 40,000샘플·hop 8,000샘플·연속 2회 기본값, 점수 계산식, 단어별 임계값,
발음 설명 규칙, 결과 JSON 계약은 바꾸지 않았다.

## 후속 미해결 문제: 짧은 발화의 확정 누락

샘플 기준 프레임 생성의 동일성이 짧은 발화의 검출 성공을 보장하지는 않는다.
추가 아동 음성 비교에서 0.5초 입력은 수정 전후 같았지만, 0.25초 또는
0.4초·0.6초 교대 입력을 가정하면 기존에 확정되던 `책`이 수정 후 한 번만
통과해 미확정됐다. 기존에도 있던 문제지만 **일부 조건의 통과 사례 손실도
확인된 미해결 이슈**로 남긴다. 판정 규칙은 아직 바꾸지 않았다.

추가 C# 비교는 실제 Python 인식 결과를 재사용한 288회 창 생성·채점 재생이며,
아래 초기 합성 인식기 테스트나 실제 Unity/Sentis 실행 검증과 구분한다.
상세 집계와 한계는 [짧은 발화 후속 진단](../../../../docs/SHORT_WORD_CONFIRMATION_2026-09-06.md)을 참고한다.

## 실제 결과

| 검사 | 결과 | 종료 코드 |
|---|---|---|
| `python -m pytest python/tests -q` | 233 passed | 0 |
| `dotnet test csharp/PhonemeMatching.Tests` | 67 passed / 0 failed | 0 |
| Unity 2022.3.62f2 EditMode | 27 passed / 0 failed | 0 |
| Unity 2022.3.62f2 PlayMode | 72 passed / 0 failed | 0 |
| Unity 6000.3.18f1 EditMode | 27 passed / 0 failed | 0 |
| Unity 6000.3.18f1 PlayMode | 72 passed / 0 failed | 0 |

Unity 결과는 임시 소비 프로젝트의 `testables` 에 이 패키지를 넣고 EditMode·
PlayMode 를 각각 돌린 것이다. 패키지 참조는 **로컬 파일 경로**(`file:` 상대
경로)로, 작업 트리의 소스를 그대로 컴파일했다. Unity 6 쪽 프로젝트에는
저장소의 `unity/Assets/Scripts/PronunciationTestBench.cs` 를 그대로 복사해
함께 컴파일했다(루트 Unity 프로젝트가 6000.3.18f1 이고 벤치가
`Unity.InferenceEngine` 을 쓰기 때문). 원본과 사본의 SHA-256 은 같다.

## 이 검증이 말하지 않는 것

- Git URL 설치 경로, Windows Player 빌드, IL2CPP 는 이번에 돌리지 않았다.
- 인식기는 합성 가짜 구현이다. 실제 마이크·GPU 추론·아동 발음 정확도는
  이 결과로 검증되지 않았고, 새 아동 오디오도 쓰지 않았다.
- 보장 범위는 **동일한 정규화 16 kHz 모노 PCM** 이다. 48 kHz 등을 청크로
  나눠 넣을 때의 리샘플링은 이번에 손대지 않았으므로 그 경우의 동일성은
  주장하지 않는다. Unity 의 hop 을 웹과 비교하려면 기본 0.5초를 그대로 쓴다.
- `Push(float[])` 와 인자 없는 `Push()` 는 수동 API 로 남아 있다. 타이머로
  `Push()` 를 돌리는 외부 통합은 여전히 호출 횟수만큼 채점하므로, 청크와
  무관한 동일성이 필요하면 `TryPush` 로 옮겨야 한다.

로그와 임시 소비 프로젝트는 저장소 기준 `.venv/frame-fix-0906-a71c/` 에 있다
(`logs/u22-editmode.xml`, `logs/u22-playmode.xml`, `logs/u6-editmode.xml`,
`logs/u6-playmode.xml`, `logs/pytest.txt`, `logs/dotnet.txt`). Git 에는
포함하지 않았고 다른 PC 에서 자동으로 제공되지 않는다.
