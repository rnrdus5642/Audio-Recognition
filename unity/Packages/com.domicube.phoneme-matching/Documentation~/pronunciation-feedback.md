# 정답 확정 시 쉬운 발음 설명과 음소 비교 데이터

현재 개발 소스의 기능이다. 기존 배포 태그 `v0.7.0`에는 없으므로 변경한 패키지를 프로젝트에 반영해야 한다. 별도 서버나 Python을 Unity에 추가할 필요는 없다.

문서 기준: 2026-09-06, 결과 계약 `schema_version: 2`. 패키지의 배포 버전과 결과 계약 버전은 별개다.

## 결과를 받는 방법

`PronunciationListener`는 기존 `OnConfirmed(string, float)`를 유지하고 다음 API를 추가한다.

- `OnConfirmedDetailed`: `PronunciationFeedback`을 전달하는 UnityEvent.
- `LastConfirmation`: 마지막 확정 결과. 다음 `Listen()` 시도 시 `null`로 초기화한다.
- 직접 `StreamingMatcher.Push()`를 사용하면 반환된 `hit.Feedback`으로 같은 데이터를 받는다.
- 기존 `PronunciationSession.Push()`는 확정 시 `frame.Feedback`을 제공한다. 확정 전에는
  `null`이며 `session.LastConfirmation`에도 보관한다. 다음 `Begin()` 시도에서 초기화한다.
- 저장소의 `PronunciationTestBench`도 `OnConfirmedDetailed`와 `LastConfirmation`을 제공한다. TestBench 시작 시 이전 결과를 비운다.

```csharp
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Newtonsoft.Json;
using UnityEngine;

// listener는 게임에서 사용 중인 PronunciationListener입니다.
listener.OnConfirmedDetailed.AddListener(feedback =>
{
    // 웹과 같은 한글 설명. Unity 화면에 이 문자열을 바로 표시할 수 있습니다.
    Debug.Log(feedback.KoreanFeedback.Summary);
    foreach (var hint in feedback.KoreanFeedback.Items)
        Debug.Log(hint.Message);

    Debug.Log($"{feedback.TargetText}: 교체 {feedback.SubstitutionCount}, "
        + $"누락 {feedback.DeletionCount}, 추가 {feedback.InsertionCount}");

    if (feedback.AlignmentAvailable)
    {
        foreach (var step in feedback.Alignment)
        {
            if (step.Operation == "match") continue;
            // -1은 해당 쪽 음소가 없다는 뜻. 표시할 때만 +1 합니다.
            Debug.Log($"{step.Operation}: 정답[{step.TargetIndex}] "
                + $"{step.TargetPhoneme} / 인식[{step.UserIndex}] {step.UserPhoneme}");
        }
    }

    // 웹의 '확정 결과 데이터'와 같은 snake_case 키를 사용하는 JSON.
    string json = JsonConvert.SerializeObject(feedback);
});
```

Listener의 호출 순서는 **청취 중단 → LastConfirmation 설정 → OnConfirmedDetailed → 기존 OnConfirmed**이다. 기존 콜백이 다음 문제나 씬으로 이동하기 전에 상세 결과를 받을 수 있다. 새 화면으로 넘겨야 한다면 이벤트에서 결과 참조를 보관한다. `LastConfirmation`은 다음 청취 시작 시 비워진다.

이 문서의 snake_case JSON은 위 예제의 `Newtonsoft.Json` 직렬화를 기준으로 한다. Unity 화면에서 C# 객체를 직접 읽을 때는 `KoreanFeedback`, `TargetText` 등 C# 필드명을 그대로 사용한다.

이 계약은 **인식 결과의 원본 비교**다. 웹은 별도로 기존 `ModelErrors/explain`을 사용해
인식기가 자주 혼동하는 차이의 안내를 보류한다. 그 화면 정책은 아직 C#에 포팅되지
않았으며, JSON과 C# DTO의 원본 차이를 제거하거나 점수를 바꾸지는 않는다.

## 데이터 계약 (schema_version 2)

버전 2는 기존 필드를 유지하면서 `korean_feedback`을 추가한다. 이전 결과 파일(버전 1)에는 이 설명이 없다. 이전 버전의 소비자가 알 수 없는 JSON 필드를 무시하면 원래 음소 비교 필드는 계속 사용할 수 있다. 버전 값을 엄격히 검사하는 소비자는 2도 허용해야 한다.

| C# 필드 | JSON 키 | 의미 |
|---|---|---|
| `SchemaVersion` | `schema_version` | 계약 버전, 현재 2 |
| `TargetId`, `TargetText` | `target_id`, `target_text` | 확정한 후보 |
| `Score`, `Distance`, `Threshold`, `Passed` | `score`, `distance`, `threshold`, `passed` | 확정 프레임의 가중 점수·거리·임계값·통과 여부 |
| `Frames`, `Streak` | `frames`, `streak` | 확정까지 채점한 횟수, 동일 후보 연속 통과 횟수 |
| `TargetPhonemes`, `UserPhonemes` | `target_phonemes`, `user_phonemes` | 정답 IPA와 확정 프레임의 전체 인식 IPA |
| `WindowStart`, `WindowEnd` | `window_start`, `window_end` | 인식 IPA에서 매칭한 범위, `[start, end)` |
| `AlignmentAvailable` | `alignment_available` | 정렬 정보 제공 여부. `false`를 ‘차이 없음’으로 표시하지 않음 |
| `Alignment` | `alignment` | 순서대로 정렬된 `PhonemeComparison` 목록 |
| `MatchCount`, `SubstitutionCount`, `DeletionCount`, `InsertionCount` | `match_count`, `substitution_count`, `deletion_count`, `insertion_count` | 비교 구간 안의 일치·교체·누락·추가 개수 |
| `KoreanFeedback` | `korean_feedback` | 바로 표시할 수 있는 한글 설명과 글자별 소리 예시 |

각 `PhonemeComparison`에는 `Operation`, `TargetPhoneme`, `UserPhoneme`, `TargetIndex`, `UserIndex`가 있다. JSON 키는 각각 `operation`, `target_phoneme`, `user_phoneme`, `target_index`, `user_index`이다.

| operation | 의미 | 없는 쪽 |
|---|---|---|
| `match` | 정답과 인식 음소가 일치 | 없음 |
| `sub` | 다른 음소로 인식 | 없음 |
| `del` | 정답 음소가 인식 결과에서 누락 | `user_index = -1`, `user_phoneme = ""` |
| `ins` | 정답에 없는 음소가 비교 구간 안에 추가 | `target_index = -1`, `target_phoneme = ""` |

인덱스는 **0부터 시작하는 IPA 토큰 위치**다. `UserIndex`는 문맥 제한으로 앞부분을 잘랐더라도 원래 `UserPhonemes` 배열 기준이다. 웹 표는 사람이 읽기 쉽게 1부터 표시한다. 한글 글자 위치나 음소의 발음 시각을 뜻하지 않는다. 추가 음소의 정답 쪽 위치는 정렬 목록의 앞뒤 항목으로 확인할 수 있다.

## 한글 설명 사용하기

`KoreanFeedback`은 `Summary`(`summary`), `SyllableMappingAvailable`(`syllable_mapping_available`), `Items`(`items`)를 제공한다. 웹은 이 데이터로 쉬운 설명을 먼저 표시하고 기존 IPA 표는 접힌 상세 보기에 둔다. 같은 점수 정렬을 설명하는 규칙 기반 변환이며, 별도 음성 인식·LLM·웹 API 호출은 없다.

각 `PronunciationHint`의 필드는 다음과 같다.

| C# 필드 | JSON 키 | 의미 |
|---|---|---|
| `Kind` | `kind` | `sub`, `del`, `ins`, 또는 한 글자 안에 여러 종류가 있으면 `mixed` |
| `SyllableIndex` | `syllable_index` | 정답 텍스트의 **한글 음절만 센** 0부터 시작하는 순번. 연결할 수 없으면 -1 |
| `TargetSyllable` | `target_syllable` | 연결된 정답 글자. 연결할 수 없으면 빈 문자열 |
| `HeardSyllable` | `heard_syllable` | 인식 음소로 만든 가까운 소리의 한글 예시. 안전하게 만들 수 없으면 빈 문자열 |
| `Message` | `message` | UI에 바로 표시할 한국어 문장 |
| `AlignmentIndices` | `alignment_indices` | 이 설명이 다루는 차이들의 `Alignment` 내 인덱스. 모두 0부터 시작 |

예를 들어 `사과`의 확정 정렬에 `target=s`, `user=tʰ`, `operation=sub`가 있으면 다음 설명이 된다.

```json
{
  "kind": "sub",
  "syllable_index": 0,
  "target_syllable": "사",
  "heard_syllable": "타",
  "message": "‘사과’의 ‘사’가 ‘타’에 가까운 소리로 인식됐어요.",
  "alignment_indices": [0]
}
```

- 이 프로젝트의 표기에서 `t`는 ㄷ, `tʰ`는 ㅌ이다. 따라서 `s → t`는 `사 → 다`, `s → tʰ`는 `사 → 타`로 설명한다.
- `과`의 `w`가 빠지면 `과 → 가`, `간`의 `n`이 빠지면 `간 → 가`처럼 보여 준다. 한 음절 전체가 빠지면 해당 글자의 소리가 인식 결과에서 빠졌다고 설명한다.
- 같은 글자의 차이는 하나의 설명으로 묶는다. 그러므로 `Items.Count`는 음소 차이 개수와 다를 수 있다. 각 차이는 `AlignmentIndices`에 한 번씩 포함된다.
- `SyllableIndex`는 문자열 인덱스가 아니다. 공백·기호·이모지는 세지 않는다. 예를 들어 `🙂사 사!`에서 두 번째 ‘사’는 1이다. 글자를 강조하려면 정답 텍스트의 한글 음절만 순서대로 대응한다.
- 표기 한글을 기존 자모→IPA 표로 변환한 결과와 실제 정답 IPA가 **완전히 같을 때만** 글자에 연결한다. `학교→학꾜`, `옷이→오시` 같은 음운 변화가 있거나 표기·IPA 메타데이터가 맞지 않으면 `SyllableMappingAvailable=false`이고 “시옷 소리가 티읕 소리로 인식됐어요”처럼 소리만 안내한다. 이 경우 글자 강조를 하지 않는다.
- 삽입 음소는 정답 글자 위치가 없으므로 별도 “더해진 소리” 설명으로 남긴다. 삽입 바로 앞뒤 글자는 완전한 인식 음절로 재구성하지 않는다. 따라서 `SyllableMappingAvailable=true`여도 일부 항목은 `SyllableIndex=-1`이거나 `HeardSyllable`이 비어 있을 수 있다.
- `HeardSyllable`은 **소리 예시이며 ASR 전사나 맞춤법 판정이 아니다**. 바뀐 받침은 대표 받침 소리로 표현하고, 그대로인 받침은 정답 표기를 유지한다. 모음 표기가 여러 개로 해석되거나 한글로 구성할 수 없는 경우에는 음절을 지어내지 않고 소리 설명으로 대체한다.
- 정렬이 없으면 자세한 설명 불가를 알린다. 정렬이 있고 차이가 없을 때만 “비교한 구간에서 정답과 다른 소리가 발견되지 않았어요”라고 안내한다.

## 판정과 표시의 구분

- 결과는 **마지막 확정 프레임의 스냅샷**이다. 연속 프레임의 평균이나 재인식 결과가 아니다. 리스트를 복사하므로 이후 매칭·초기화가 이전 결과를 바꾸지 않는다. 소비자는 결과를 읽기 전용으로 취급한다.
- 정답 처리와 음소 차이는 동시에 존재할 수 있다. 허용 임계값을 넘겼다고 모든 음소가 일치했다는 뜻은 아니다.
- 매칭 구간 밖의 음소는 위 추가 개수에 포함하지 않는다. 점수에는 혼동행렬 비용·주변 문맥·길이 조건도 영향을 주므로 차이 개수와 점수가 일대일 대응하지 않는다.
- 설명은 인식 IPA 전체를 새로 해석하는 것이 아니라 실제 채점 정렬의 결과를 따른다. 매처가 어떤 소리를 구간 밖으로 제외하고 정답 소리를 누락으로 정렬했다면, 그 결과를 임의로 교체로 바꾸어 설명하지 않는다.
- ASR이 인식한 음소의 차이이지 실제 발음 오류를 확정하는 진단은 아니다. 한글 설명도 “가까운 소리로 인식됐다”는 안내이며, 실제 발음을 단정하지 않는다. 위의 엄격한 연결 조건을 만족하지 않으면 특정 글자를 지목하지 않는다.
- 웹의 표시와 JSON은 같은 `hit.feedback`을 사용한다. 웹과 Unity의 데이터 계약은 기준 벡터를 함께 검사하지만, 아직 Python과 C# 전체 런타임이 통합된 것은 아니다. 동일 마이크 음성이 양쪽 추론 백엔드에서 같은 IPA가 되는지는 별도 검증 대상이다.

기준 자료: `Tests/Runtime/feedback_vectors.json`, `Tests/Runtime/korean_feedback_vectors.json`. Python과 C# 양쪽에서 같은 기대값, 설명 문장, 글자·음소 인덱스, 누락/추가 방향, JSON 키를 검증한다. 전체 한글 11,172음절의 기본 매핑도 양쪽에서 검사한다. 전체 런타임 통합 전까지 양쪽 설명 로직을 수정할 때 이 공통 벡터를 함께 통과해야 한다.
