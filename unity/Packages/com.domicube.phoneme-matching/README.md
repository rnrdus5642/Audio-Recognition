# Phoneme Matching (Unity 패키지)

**0.8.5 · 악기놀이 발표용:** Package Manager의 `+ > Add package from tarball…`에서
제공된 `com.domicube.phoneme-matching-0.8.5.tgz`를 선택하고,
`Tools > Phoneme Matching > 악기놀이 발표 설정`을 실행하세요.
필요한 추론 엔진과 아동 모델을 준비하고, **12개 정답의 조정값·9단계가 연결된 테스트 창**을 엽니다.
이미 설치된 아동 모델은 재사용합니다. 처음 받는 모델은 약 1.18GB라 다운로드·임포트 시간이 필요합니다.

발표 설정은 패키지 Resources에 포함됩니다. JSON 복사나 Python 실행이 필요 없으며,
이후 패키지 업데이트에도 새 설정이 함께 반영됩니다. 게임에는
`GameObject > Phoneme Matching > 악기놀이 발표`로 컴포넌트를 추가하세요.
기존 게임의 연결 코드와 업데이트 방법은 [악기놀이 적용 안내](Documentation~/music-presentation.md)를 보세요.

발화가 정답 단어와 **음소(IPA) 수준에서** 충분히 가까운지 판정합니다. STT로 "무슨
단어인지" 맞추는 게 아니라 음소 부분 매칭이라, 발음이 부정확한 사용자(아동·발달지연
포함)에게도 동작하고 STT 환각·마이크 잡음 문제를 함께 회피합니다.

**온디바이스 전용입니다.** 서버도 파이썬 런타임도 필요 없습니다.

2026-09-06 개발 소스에는 기존 JSON을 보존하는 초기 세팅, 합격 후보 우선 선택,
모델별 테스트 창 설정이 반영되어 있습니다. 웹도 기본 2.5초 창·앞쪽 무음 패딩으로 맞췄습니다.
검증된 환경과 남은 제약은 [검증 기록](Documentation~/validation-2026-09-06.md)을 참고하세요.
0.8.2에는 9월 9일 변경된 정답표와 `잡아·연주해·내려놔`의 판정 설정을 포함했습니다.
0.8.3에서는 보류된 차이를 화면에 `문제 없음`으로 표시합니다. 내부 보류 기록과 정답 설정은 유지합니다.
0.8.5는 실제 사용자 녹음으로 확인한 `잡아`의 모델 오인식을 보완합니다. 같은 추론의 CTC 후보 점수를 추가 확인하며, 원래 인식 문구와 연속 2회 조건을 유지합니다. 다른 단어 설정은 0.8.4와 같습니다.

0.8.4는 악기 이름이 붙은 동작 표현, 짧은 발화 보완, `잡아` 연속 2회와 인식 진단 로그를 추가합니다.
0.8.1에서 추가한 모델 혼동을 고려한 발음 안내도 유지합니다. Git URL로 받으려면 이번 변경이
원격에 게시되어야 하며, 기존 `v0.7.0` 태그에서는 새 발표 기능을 받을 수 없습니다.

## 실행 구조

```
[개발 PC · 오프라인]
  words.csv ──(한국어 음운 규칙 + IPA 매핑)──> targets.json

[기기 · 런타임]
  마이크 → IPhonemeRecognizer → IPA → Matcher → 통과/재시도
```

정답 빌드와 인식 결과 모두 한국어 음운 규칙을 적용합니다. Unity에서는
`Korean.Rules.Apply(text)` 후 `JamoIpa.ToPhonemes()`를 호출하며, Python에도
같은 규칙 구현이 있습니다. 런타임에 mecab이나 Python은 필요 없습니다.
g2pkk는 Python의 별도 비교 실험 경로에서만 사용합니다.

## 일반 단어용 설치

**Unity 2022.3 과 Unity 6 을 모두 지원합니다.**

패키지를 넣고 **버튼 하나** 누르면 끝납니다.

```
Window > Package Manager > + > Add package from git URL…
https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching

Tools > Phoneme Matching > 초기 세팅
```

`초기 세팅` 은 없는 것만 차례대로 설치합니다 - 추론 엔진, 데이터 JSON 6개,
음향 모델. 이미 있는 것은 건너뛰므로 두 경우에 같은 버튼을 씁니다.

| 상황 | 실제로 하는 일 |
|---|---|
| 처음 붙이는 프로젝트 | 세 단계 전부 |
| **이미 쓰는 프로젝트를 clone/pull** | **모델만** (나머지는 커밋돼 있음) |

모델은 1.18GB 라 커밋되지 않으므로 각자 한 번 받아야 합니다. 모델이 없는
채로 에디터를 열면 콘솔이 알려줍니다.

엔진 설치가 프로젝트를 재컴파일시키므로 중간에 한 번 멈췄다 이어집니다.
전체는 회선에 따라 5~10분 걸립니다.

끝나면 `Tools > Phoneme Matching > 발음 테스트 (마이크)` 로 확인하세요.

테스트 창은 선택한 모델의 **표준 파일명**에 맞춰 채점 데이터를 자동 연결하며,
창에 실제 사용하는 JSON 파일을 표시합니다.

| 모델 파일명 | matrix | 정답 데이터 |
|---|---|---|
| `wav2vec2_ko.onnx` | `ko_child_v1.json` | `targets.json` |
| `wav2vec2_ko_child.onnx` | `ko_child_v2.json` | `targets_child.json` |

공통 어휘는 `wav2vec2_ko_vocab.json`입니다. 이름을 바꾼 모델이나 별도 모델은
`채점 설정 (직접 지정)`에서 기본/아동 모델용을 선택해야 시작할 수 있으며,
선택은 모델 에셋별로 기억합니다. 자동 구분은 파일명 기준이지 모델 내부를 분석한
결과는 아니므로, 표준 이름으로 다른 모델을 덮어쓴 경우까지 식별하지는 않습니다.
모델·채점 설정을 바꾸면 이전 준비 세션과 결과를 비우고 다시 준비합니다.
JSON이 누락되면 해당 파일을 안내하며 다른 모델용 설정으로 대체하지 않습니다.
기존 JSON 수정 내용도 그대로 읽습니다. 이 자동 연결은 **에디터 테스트 창에만**
적용되며, 게임의 `PronunciationListener`·`PronunciationSession` 설정 API는 그대로입니다.

<details>
<summary>메뉴 전체와 단계별 설명</summary>

```
초기 세팅                    ← 아래 셋을 알아서
────────────────────────
추론 엔진 설치               개별 실행도 됩니다
데이터 파일 설치
음향 모델 내려받기              성인용
음향 모델 파일에서 가져오기…   이미 파일이 있을 때
아동 음향 모델 내려받기         5~7세 전용. 성인용을 대체하지 않습니다
아동 음향 모델 파일에서 가져오기…
────────────────────────
발음 테스트 (마이크)
────────────────────────
단어 목록 열기 (words.csv)    ┐ 정답 단어를 바꿀 때만.
정답 데이터 다시 만들기        ┘ 저장소와 파이썬이 필요합니다
────────────────────────
저장소 경로 지정…
```

순서가 중요합니다 - 추론 엔진이 `.onnx` 임포터를 등록하므로, 엔진 없이
모델을 먼저 넣으면 `ModelAsset` 이 만들어지지 않습니다.

</details>

### 1. 패키지

`Window > Package Manager` > `+` > **Add package from git URL…**

```
https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching
```

`manifest.json` 에 직접 적어도 같습니다:

```json
"com.domicube.phoneme-matching": "https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching"
```

새로 설치할 때는 원격 기본 브랜치를 사용하지만, 설치 후에는 `packages-lock.json`의
커밋이 유지되므로 main 변경이 자동 반영되지는 않습니다. 수정본이 푸시된 뒤
`Add package from git URL…`에 같은 URL을 다시 입력하거나, 뒤에 `#`와 원하는
전체 40자리 커밋 해시 또는 태그를 붙여 갱신하세요. 기존 `v0.7.0` 태그를 지정하면
이번 수정본이 아닌 그 태그의 소스를 받습니다.
[Unity의 Git 의존성 잠금·갱신 설명](https://docs.unity3d.com/2022.3/Documentation/Manual/upm-git.html#git-locks).
로컬에서 고쳐가며 쓰려면 `"file:../../path/to/com.domicube.phoneme-matching"`도 됩니다.

### 2. 추론 엔진

`Tools > Phoneme Matching > 추론 엔진 설치`

기본 패키지와 `LessonExample`은 엔진 설치 전에도 컴파일됩니다.
실제 음성 인식에는 추론 엔진과 모델이 필요합니다. 악기놀이 컴포넌트는 준비가
안 된 상태에서 정답을 만들지 않고, 설치 안내를 표시합니다.

에디터 버전을 보고 알맞은 것을 받습니다 - 2022.3 은
`com.unity.sentis 1.2.0-exp.2`, Unity 6 은
`com.unity.ai.inference 2.6.1`.

패키지가 이것을 의존성으로 선언하지 못하는 이유는 UPM 이 "A 또는 B" 를
표현할 수 없기 때문입니다. 2.x 를 박으면 2022 에서 설치가 실패하고, 1.2 를
박으면 Unity 6 사용자가 중단된 실험 패키지를 강제로 받게 됩니다.

설치된 엔진에 맞는 인식기만 컴파일됩니다 (asmdef 의 `versionDefines` +
`defineConstraints`). 클래스 이름은 양쪽 다
`SentisPhonemeRecognizer` 라서 호출 코드는 동일하고, `ModelAsset` 을
선언할 때의 `using` 만 다릅니다 - 2022 는 `Unity.Sentis`, Unity 6 은
`Unity.InferenceEngine`.

### 3. 데이터 파일

`Tools > Phoneme Matching > 데이터 파일 설치`

`ko_child_v1.json` · `targets.json` · `ko_child_v2.json` ·
`targets_child.json` · `wav2vec2_ko_vocab.json` 을
`Assets/StreamingAssets/` 에 넣고, 런타임이 읽는 방식 그대로 다시 읽어
확인까지 합니다. 이미 있으면 덮어쓸지 물어봅니다.

`초기 세팅`은 누락된 JSON만 추가하고, 이미 있는 단어 목록·판정 설정은 수정하지
않습니다. 반면 별도 `데이터 파일 설치` 메뉴는 확인 후 기존 파일을 패키지 기본
데이터로 교체합니다. 직접 추가한 단어와 수정한 임계값이 있다면 교체 전에 보관하세요.

뒤의 두 개가 아동 전용입니다. 성인용을 대체하지 않고 나란히 놓입니다 —
두 음향 모델이 서로 다르게 잘못 들어서, 한쪽에 맞춘 임계값과 채점 설정이
다른 쪽에는 맞지 않습니다.

> Package Manager 의 샘플 임포트 버튼을 써도 되지만, 그건 파일을
> `Assets/Samples/...` 에 놓기 때문에 `StreamingAssets/` 로 한 번 더
> 옮겨야 합니다. 위 메뉴가 그 단계를 없앤 것입니다.

단어를 바꿨다면 `데이터 파일 설치` 대신
`정답 데이터 다시 만들기` 를 쓰세요 — 저장소의 `words.csv` 로 빌드해서
`targets.json` 만 갱신합니다.

> **패키지만 올리면 데이터는 그대로입니다.** 이미 프로젝트의
> `StreamingAssets/` 에 복사된 `targets_child.json` 은 패키지 갱신으로
> 덮이지 않습니다. 동작 그룹(`action_group` · `contrastive_alternatives`)
> 처럼 카탈로그 형식이 늘어난 갱신을 받으려면 `데이터 파일 설치`로 최신
> 아동 데이터도 설치해야 합니다. `정답 데이터 다시 만들기`는 위 설명처럼
> 성인 `targets.json`만 갱신하므로 아동 데이터 갱신을 대신하지 않습니다.
> 직접 손본 카탈로그가 있다면 덮어쓰기 전에
> 따로 보관하세요. 이 필드가 없는 예전 카탈로그도 그대로 동작합니다 —
> 없으면 비교할 다른 동작이 없다는 뜻이고, 판정은 예전과 같습니다.

2026-09-07 아동 카탈로그에는 악기 동작 표현 10종이 모두 등록됐습니다(총 35개).
같은 그룹은 `들어/들기`, `쳐/치기`, `놓아/놓기`, `흔들어/흔들기`,
`두드려/두드리기`입니다. `Begin`에는 허용할 표현만 기존처럼 넘기면 됩니다.
다른 동작이 발화 전체에 더 잘 맞으면 기존 통과를 취소하며, 허용하지 않은
표현으로 정답을 바꾸지는 않습니다. 호출 시 새로운 필수 인자는 없습니다.

취소 정보는 `FrameScore.Best`의 `CompetitorRejected`, `CompetitorReason`,
`CompetitorText`, `CompetitorScore`, `AcceptedWholeScore`로 확인합니다.
마지막 두 값은 전체 음소열 비교 점수이며 확률이 아닙니다. 취소되지 않은
프레임은 취소 정보가 비어 있습니다. 원래 `Score`·`Threshold`는 그대로이므로
직접 점수만 비교하지 말고 `Passed`·`Confirmed`·확정 이벤트를 사용하세요.
확정 시 발음 피드백 형식은 바뀌지 않았습니다.

실제 아동 녹음의 해당 표현은 여전히 **7개 중 2개만 확정**됐습니다. 새로
등록한 6개 표현은 실제 녹음이 없어 인식률을 검증하지 못했습니다. 이번 수정은
표현 등록 누락과 다른 동작의 오통과를 줄이는 것이며 음향 모델 개선은 아닙니다.

### 4. 음향 모델

`Tools > Phoneme Matching > 음향 모델 내려받기`

1.18GB 라 패키지에 담을 수 없어 GitHub Release 에 따로 올려두었습니다.
`Assets/Resources/Models/wav2vec2_ko.onnx` 로 받고, **크기와 SHA-256 을
확인한 뒤에** Assets 로 옮깁니다 - 잘린 모델은 임포트까지는 되고 추론에서
죽는데, 그때는 원인이 여기라는 단서가 없습니다.

받은 뒤 Unity 임포트에 몇 분 걸립니다. **이 파일을 커밋하지 마세요** -
GitHub 는 100MB 넘는 파일을 거부합니다. 다운로드가 끝나면 프로젝트
`.gitignore` 에 규칙을 넣을지 물어봅니다.

이미 파일을 가지고 있다면(사내 공유 폴더, 다른 프로젝트)
`음향 모델 파일에서 가져오기…` 로 고르면 됩니다. 검증은 똑같이 거칩니다.

### 5. 아동 모델 (선택)

5~7세 아동 음성 200시간으로 재학습한 별도 모델입니다. 아이 목소리에서
문자 오류율이 55.4% 에서 8.7% 로 내려갑니다.

`Tools > Phoneme Matching > 아동 음향 모델 내려받기`

**성인 모델을 대체하지 않습니다.** 둘은 서로 다르게 잘못 듣기 때문에
데이터도 같이 바꿔야 합니다.

```csharp
var listener = GetComponent<PronunciationListener>();
listener.MatrixFileName  = "ko_child_v2.json";
listener.TargetsFileName = "targets_child.json";

var model = Resources.Load<ModelAsset>(
    "Models/wav2vec2_ko_child");
```

셋을 섞으면 안 됩니다. 아동 설정을 성인 모델에 쓰면 아이가 말하지도 않은
세션의 6.47% 를 확정합니다 (예산은 1%).

처음 보는 아이 52명 기준 성능입니다.

| | 단독 단어 | 오확정 |
|---|---|---|
| 성인 모델 + 성인 설정 | 37.3% | 3.50% |
| 아동 모델 + 아동 설정 | **88.1%** | **0.70%** |

> 순서가 중요합니다. **추론 엔진이 먼저** 설치돼 있어야 `.onnx` 가
> `ModelAsset` 으로 임포트됩니다.

#### 모델은 인스펙터에 끌어다 놓지 마세요

`Resources` 아래에 두는 이유가 있습니다. 모델은 커밋되지 않으므로 팀원마다
각자 임포트하고, 그때 **애셋 GUID 가 사람마다 달라집니다.** 씬이나 프리팹의
인스펙터 참조는 GUID 로 저장되니, 한 사람이 끌어다 놓고 커밋하면 다른
사람들에게는 그 칸이 비어 보입니다. `.meta` 를 커밋해도 소용없습니다 -
애셋이 없으면 Unity 가 `.meta` 를 지웁니다.

경로로 불러오면 GUID 를 쓰지 않으므로 아무도 깨지지 않습니다.

```csharp
var model = Resources.Load<ModelAsset>("Models/wav2vec2_ko");
var recognizer = new SentisPhonemeRecognizer(model, vocab);
```

팀원이 저장소를 클론해서 쓰는 경우, **모델만 각자 한 번 받으면 됩니다.**
`manifest.json` 과 `StreamingAssets` 의 JSON 은 커밋되니까요. 모델이 없는
채로 에디터를 열면 콘솔이 한 번 알려줍니다.

직접 만들 수도 있습니다. 저장소에서:

```powershell
python -m python.tools.export_onnx
```

**모델은 정확히 40000 샘플(2.5 초)만 받습니다.** 시간 축이 고정으로
export 되기 때문입니다 - Sentis 1.2 는 동적 축 그래프를 실행하지 못하고
(임포트는 되지만 어떤 길이를 줘도 같은 Reshape 에서 죽습니다), 축을
고정하면 그 계산이 상수로 접혀 사라집니다. Unity 6 은 동적도 실행하지만
파일 하나로 양쪽을 덮으려고 고정으로 통일했습니다. 부수 효과로 런타임
shape 계산이 사라져 프레임당 30ms → 20ms 로 빨라집니다.

`PronunciationListener.WindowSeconds` 를 바꾸면 모델도 그 길이로 다시
뽑아야 합니다:

```powershell
python -m python.tools.export_onnx --static-samples 48000   # 3.0초 창
```

모델 없이 매칭 계층만 쓰는 것도 됩니다 — `IPhonemeRecognizer` 를 직접
구현하면 Sentis 경로는 건드리지 않아도 됩니다.

## 준비 (네 방법 공통)

```csharp
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using Unity.Sentis;          // Unity 6 이면 Unity.InferenceEngine

string Read(string f) => System.IO.File.ReadAllText(
    System.IO.Path.Combine(Application.streamingAssetsPath, f));

var matrix  = PhonemeData.LoadMatrix(Read("ko_child_v1.json"));
var catalog = PhonemeData.LoadTargets(Read("targets.json"));
var vocab   = PhonemeData.LoadCtcVocabulary(Read("wav2vec2_ko_vocab.json"));

// 인스펙터 참조가 아니라 경로로. 이유는 위 "인스펙터에 끌어다 놓지 마세요".
var model = Resources.Load<ModelAsset>("Models/wav2vec2_ko");

var recognizer = new SentisPhonemeRecognizer(model, vocab);
recognizer.Warmup(2.5f);     // 로딩 화면에서. 첫 추론이 ~2초 걸립니다
```

다 쓰면 `recognizer.Dispose()`.

## 어느 방법을 쓸지

오디오를 누가 가지고 있느냐로 갈립니다.

| | 상황 | 방법 |
|---|---|---|
| 1 | 앱에 오디오 캡처가 **없다** | 패키지가 마이크를 엽니다 |
| 2 | 앱이 `Microphone` 으로 캡처 중 | `GetData` 결과를 넘깁니다 |
| 3 | 마이크가 `AudioSource` 로 흐른다 | `OnAudioFilterRead` 에서 넘깁니다 |
| 4 | 보이스 SDK 가 콜백을 준다 | 그 콜백에서 넘깁니다 |

2~4 는 오디오를 넘기는 지점만 다르고 나머지는 같습니다. 마이크를 두 번
열면 충돌하므로, 앱이 이미 캡처 중이라면 1 을 쓰지 마세요.

---

### 방법 1 — 마이크까지 맡기기

컴포넌트를 붙이고 이벤트만 받습니다. 마이크는 `Listen()` 과
`StopListening()` 사이에만 열립니다.

```csharp
var listener = gameObject.AddComponent<PronunciationListener>();
listener.SetRecognizer(recognizer);
listener.MicrophoneDevice = "Headset Microphone (...)";   // 비우면 OS 기본
listener.TargetWords = new[] { "사과" };                  // 여러 개도 가능

listener.OnConfirmed.AddListener((word, score) => 다음문제로());
listener.OnTimedOut.AddListener(() => 다시해볼까());

listener.Listen();          // 마이크 켜짐
listener.StopListening();   // 끄기 (확정·타임아웃에도 자동으로 꺼짐)
```

창 길이·hop·확정 횟수·타임아웃은 인스펙터에서 조정합니다. 장치 이름은
`Microphone.devices` 로 확인하세요 - VR 에서는 OS 기본이 헤드셋 마이크가
아닌 경우가 많습니다.

**정답을 여러 개 둘 수 있습니다.** 하나만 맞아도 확정되고, `OnConfirmed`
의 첫 인자가 어느 단어였는지 알려줍니다.

```csharp
listener.TargetWords = new[] { "엄마", "어머니" };
```

각 프레임에서 **각 정답의 점수를 그 정답의 임계값과 먼저 비교**합니다. 통과한
정답이 있으면 그중 최고 점수를 선택하고, 전부 불합격이면 표시용 최고 점수만
반환합니다. 같은 통과 정답이 연속 2회(기본값) 선택돼야 최종 확정됩니다.

예외가 하나 있습니다. 정답 데이터가 그 단어에 `allow_exact_single_frame`
을 켜 뒀고, `Consecutive` 가 정확히 2이며, 지금 설정이 그 단어의 **보정
스냅샷**(`exact_single_frame_calibration`: 매트릭스 이름·스트리밍
프로파일·임계값·음소열·표기)과 같으면, **점수가 정확히 1.0 인 프레임
하나로 확정**됩니다. 1.0 은 **그 프레임에서 채점에 쓰인 구간**(문맥 제한으로
잘라낸 최근 음소열)이 설정된 정답 음소열과 그대로 같았다는 뜻입니다 — 전사
전체가 정답뿐이었다는 뜻도, 발음이 100% 정확했다는 인증도 아니며, 0.9999 는
해당하지 않습니다. 짧은 단어가 그런 프레임을 딱 한 번만 얻는 경우를 위한
것입니다 — 발화는 다음 hop 에도 창 안에 있지만 창 안에서의 위치가 바뀌면
인식 결과가 달라집니다
([근거와 한계](../../../docs/DEMO_RELIABILITY_2026-09-06.md)).
이때 보고되는 `Streak` 은 진짜 **1** 입니다 — 한 프레임을 두 번 성공한
것처럼 표시하지 않습니다. 플래그가 없는 데이터, 성인용 타깃, 손으로 조정한
프로파일, `Consecutive` 1 또는 3 은 전부 예전 그대로입니다.

매트릭스는 자기 이름을 알아야 합니다. `ko_child_v2.json` 에는 `matrix_id`
가 들어 있어 `PhonemeData.LoadMatrix(json)` 한 인자 호출로도 신원이
확인됩니다. `matrix_id` 가 없는 예전 파일은 `unknown` 으로 읽히고 예외를
받지 못하므로, 필요하면 `LoadMatrix(json, "ko_child_v2")` 로 직접
알려 주세요. 이 예외를 쓰려면 **`targets_child.json` 과 `ko_child_v2.json`
을 함께** 최신으로 갱신해야 합니다.

대조 대상은 매트릭스·스트리밍 프로파일·임계값·음소열까지입니다.
**음향 모델(ONNX)의 신원은 확인하지 않습니다** — 보정은 아동 파인튜닝
모델에서 잰 값이므로, 다른 모델을 쓸 거라면 다시 보정하거나 플래그를 끄는
것은 호출자의 책임입니다.
정답이 바뀌면 연속 횟수는 1부터, 모두 불합격이면 0부터 다시 시작합니다.
`FrameScore.Best`, 이벤트의 단어·점수, 확정 피드백은 모두 이 선택된 정답 기준입니다.
`Begin(...)`·이벤트·결과 JSON 형식은 바뀌지 않습니다.

다만 **오확정 확률이 대략 곱해집니다.** 개수보다 어느 단어를 넣느냐가
문제입니다. 아래 수치는 이전 후보 선택 규칙에서의 측정 예시이며, 현재 규칙의
정확도 수치로 사용하면 안 됩니다.

```
엄마/아빠                  3/100
엄마/아빠/할아버지          5/100
엄마/아빠/할머니           21/100    ← 할머니 하나가 19
```

`할머니` 처럼 긴 단어는 다른 말 안에 통과할 구간이 생기기 쉬워 혼자
오확정을 다 만듭니다. 동의어가 정말 필요할 때만 쓰세요.

---

### 방법 2~4 공통 — 앱이 오디오를 준다

리듬이 둘로 나뉩니다. **오는 대로 붓고, 0.5초치가 모이면 채점합니다.**

```
앱의 오디오 ──Append()──▶ [ 2.5초 창 ] ──TryPush()──▶ 채점
            오는 대로 계속              새 샘플 0.5초마다
```

```csharp
// 한 번 만들어 계속 씁니다. Begin ~ End 한 번이 문제 하나입니다.
// hop 을 주면 버퍼가 0.5초치 새 샘플마다 프레임을 하나씩 쌓습니다.
var buffer  = new AudioWindowBuffer(2.5f, 0.5f);
var session = new PronunciationSession(matrix, catalog, recognizer,
                                       buffer: buffer);

// 문제 낼 때 — 창을 비우고 시작합니다
session.Begin("사과");                      // 이 단어만 채점
// session.Begin(new[] { "사과", "능금" }); // 둘 중 아무거나 정답

// 아무 때나 자주 불러도 됩니다. 0.5초가 안 모였으면 아무것도 안 합니다.
while (session.TryPush(out var frame))
{
    if (frame.Confirmed) { 다음문제로(); break; }
}

session.End();   // 중단 — 안 읽은 프레임과 남은 부분 hop 도 같이 버립니다
```

**왜 `TryPush` 인가.** `Push()` 는 부를 때마다 지금 창을 채점하므로,
채점 횟수가 오디오가 아니라 **호출 횟수**를 따라갑니다. 같은 1초를
8000샘플씩 두 번 주면 2회, 4000샘플씩 네 번 주면 4회 채점돼 확정 시점이
달라졌습니다. `TryPush` 는 새 샘플 수로 프레임을 끊으므로 청크를 어떻게
쪼개 넣든 같은 창이 같은 순서로 나옵니다. 한 청크에 hop 이 여러 개 들어
있으면 각각의 경계 시점 창을 순서대로 돌려주고, 확정된 프레임에서 멈춥니다.
(보장 범위는 **같은 16 kHz 모노 PCM** 입니다. 다른 샘플레이트를 넣을 때의
리샘플링은 종전과 같습니다.)

`Push()` 는 직접 창을 넘기는 수동 API 로 그대로 둡니다. 이미 자체 클럭을
돌리는 코드는 바꾸지 않아도 됩니다.

`TryPush()` 는 메인 스레드에서 부르세요 - 20~30ms 걸리고 GPU 를 씁니다.
오답 신호는 따로 없습니다. 시간을 재서 앱이 판단하세요 (방법 1 은 이
타이머가 내장돼 있습니다).

아래 세 방법은 **`Append` 한 줄만** 다릅니다.

---

### 방법 2 — 앱이 Microphone 으로 캡처 중

```csharp
int _last = 0;

void Update()
{
    int pos = Microphone.GetPosition(device);
    int available = pos - _last;
    if (available < 0) available += clip.samples;   // 링 버퍼가 한 바퀴
    if (available <= 0) return;

    var raw = new float[available];
    clip.GetData(raw, _last % clip.samples);
    _last = pos % clip.samples;

    buffer.Append(raw, clip.frequency);
}
```

`clip.frequency` 를 넘기는 게 중요합니다. 장치가 16 kHz 를 거절하면
Unity 가 다른 레이트로 열어주고, 요청한 값으로 리샘플링하면 오디오가
시간 축으로 어긋납니다 - 에러가 아니라 발음이 나쁜 것처럼 보입니다.

---

### 방법 3 — AudioSource 를 지나는 오디오

```csharp
void OnAudioFilterRead(float[] data, int channels)
{
    buffer.AppendInterleaved(data, channels, AudioSettings.outputSampleRate);
}
```

오디오 스레드에서 불리지만 `AudioWindowBuffer` 는 그래도 안전합니다.
`AppendInterleaved` 가 모노로 섞고 16 kHz 로 변환합니다.

---

### 방법 4 — 보이스 SDK 콜백

```csharp
// float[] 로 주는 경우
void OnFrame(float[] frame) => buffer.Append(frame, 48000);

// short[] (16비트 PCM) 로 주는 경우
void OnFrame(short[] pcm)
{
    var f = new float[pcm.Length];
    for (int i = 0; i < pcm.Length; i++) f[i] = pcm[i] / 32768f;
    buffer.Append(f, 48000);
}
```

청크 크기는 아무래도 좋습니다. 창이 알아서 이어붙이고 오래된 것을
밀어냅니다.

---

## 결과 읽기

정답 확정 시에는 `frame.Feedback`과 `session.LastConfirmation`으로
쉬운 한글 설명·음소별 차이·점수·임계값·연속 횟수를 받습니다.
마이크 Listener는 `OnConfirmedDetailed`와 `LastConfirmation`을 제공하고,
기존 `OnConfirmed(word, score)`도 유지합니다.
[발음 설명 API](Documentation~/pronunciation-feedback.md)에 전체 필드와 예제가 있습니다.

```csharp
listener.OnConfirmedDetailed.AddListener(feedback =>
{
    foreach (var hint in feedback.KoreanFeedback.Items)
        Debug.Log(hint.Message);
});
```

`KoreanFeedback`은 웹·Unity 모두 모델 오류표를 확인한 표시용 안내입니다.
모델 혼동이 잦거나 자료가 없거나 부족한 차이는 보류합니다. `놔 → 와`의 ㄴ 누락도
발음 오류로 지적하지 않습니다. `RawKoreanFeedback`·`Alignment`에는 원본 차이가 남습니다.
결과 계약은 `schema_version: 3`이며 정답 점수·임계값·확정 횟수는 그대로입니다.

방법 2~4 는 `Push()` 가 돌려주고, 방법 1 은 `OnFrameScored` 로 옵니다.

```csharp
frame.Confirmed        // true 면 확정 - 이 순간 세션 종료
frame.Streak           // 연속 몇 번째 (기본 2 면 확정)
frame.Best.TargetText  // 합격 후보 우선으로 선택된 단어 (전부 불합격이면 최고 점수)
frame.Best.Score       // 0~1
frame.Best.Passed      // 이번 창이 임계값을 넘었는지
frame.Best.Alignment   // 음소별로 무엇이 어긋났는지
frame.Text             // 인식된 한글
```

`Alignment` 는 `(사용자음소, 정답음소, 연산)` 의 나열입니다 - 어느 소리를
어떻게 틀렸는지 보여주는 피드백에 쓰세요.

**청크를 따로 인식해 이어붙이지 마세요.** wav2vec2는 문맥 모델이라 짧은 조각에서
무너집니다 — 4.4초 발화에서 0.5초 청크는 43음소가 31음소로 줄고 단어가 깨졌습니다.
매 프레임 **최근 창 전체를 재인식**해야 합니다. 위 방법들은 모두 그렇게 합니다.

### 오답 신호는 없습니다

물어본 단어만 채점하므로, 아이가 딴말을 하면 **확정이 안 될 뿐**입니다.
"틀렸다"는 판정은 시간으로 앱이 정합니다.

```csharp
if (Time.time - _asked > 10f) { 다시해볼까(); judge.End(); }
```

방법 1 은 이 타이머가 들어 있어 `OnTimedOut` 으로 옵니다.

---

## 통째로 붙여 쓰는 예제

### 방법 1 — 패키지가 마이크를 연다

```csharp
using System.IO;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using UnityEngine;
using Unity.Sentis;                  // Unity 6 이면 Unity.InferenceEngine

[RequireComponent(typeof(PronunciationListener))]
public sealed class Lesson : MonoBehaviour
{
    public string[] Words = { "사과", "엄마", "토끼" };

    PronunciationListener _listener;
    SentisPhonemeRecognizer _recognizer;
    int _index;

    void Awake()
    {
        var vocabPath = Path.Combine(
            Application.streamingAssetsPath, "wav2vec2_ko_vocab.json");
        var vocab = PhonemeData.LoadCtcVocabulary(File.ReadAllText(vocabPath));

        // 인스펙터 참조 대신 경로로 — 모델은 커밋되지 않아 GUID 가
        // 사람마다 다릅니다.
        var model = Resources.Load<ModelAsset>("Models/wav2vec2_ko");
        _recognizer = new SentisPhonemeRecognizer(model, vocab);

        _listener = GetComponent<PronunciationListener>();
        _listener.SetRecognizer(_recognizer);
        _listener.MicrophoneDevice = "";          // 비우면 OS 기본 장치
        _listener.TimeoutSeconds = 15f;

        _listener.OnConfirmed.AddListener((word, score) =>
        {
            Debug.Log($"정답 {word} {score:F2}");
            _index++;
            Invoke(nameof(Ask), 1f);
        });

        _listener.OnTimedOut.AddListener(() => Debug.Log("다시 해볼까?"));

        _listener.OnFrameScored.AddListener((word, score, streak) =>
            Debug.Log($"{word} {score:F2} ({streak}/{_listener.Consecutive})"));
    }

    System.Collections.IEnumerator Start()
    {
        yield return null;                  // 배너 한 프레임 그리고
        _recognizer.Warmup(_listener.WindowSeconds);   // ~2초 멈춤
        Ask();
    }

    public void Ask()
    {
        if (_index >= Words.Length) return;
        _listener.TargetWords = new[] { Words[_index] };
        _listener.Listen();                 // 마이크 켜짐
    }

    public void Stop() => _listener.StopListening();   // 언제든 끄기

    void OnDestroy() => _recognizer?.Dispose();
}
```

`PronunciationListener` 는 마이크·타이밍·타임아웃만 맡고, 판정은 아래
`PronunciationSession` 에 그대로 위임합니다. 두 방법의 채점 결과는 같습니다.

### 방법 2~4 — 앱이 오디오를 준다

```csharp
using System.IO;
using DomiCube.PhonemeMatching;
using DomiCube.PhonemeMatching.Unity;
using UnityEngine;
using Unity.Sentis;                  // Unity 6 이면 Unity.InferenceEngine

public sealed class Lesson : MonoBehaviour
{
    public string[] Words = { "사과", "엄마", "토끼" };

    AudioWindowBuffer _buffer;
    PronunciationSession _judge;
    SentisPhonemeRecognizer _recognizer;
    Coroutine _loop;
    int _index;

    void Awake()
    {
        string Read(string f) => File.ReadAllText(
            Path.Combine(Application.streamingAssetsPath, f));

        var matrix  = PhonemeData.LoadMatrix(Read("ko_child_v1.json"));
        var catalog = PhonemeData.LoadTargets(Read("targets.json"));
        var vocab   = PhonemeData.LoadCtcVocabulary(Read("wav2vec2_ko_vocab.json"));

        // 인스펙터 참조 대신 경로로 — 모델은 커밋되지 않아 GUID 가
        // 사람마다 다릅니다.
        var model = Resources.Load<ModelAsset>("Models/wav2vec2_ko");
        _recognizer = new SentisPhonemeRecognizer(model, vocab);
        _buffer = new AudioWindowBuffer(2.5f, 0.5f);   // 창 2.5초 · hop 0.5초
        _judge  = new PronunciationSession(
            matrix, catalog, _recognizer, buffer: _buffer);
    }

    System.Collections.IEnumerator Start()
    {
        yield return null;
        _recognizer.Warmup(2.5f);
        Ask();
    }

    // ── 앱의 오디오를 여기로 부으세요 (오디오 스레드에서 불러도 안전) ──
    void OnAudioFilterRead(float[] data, int channels)
    {
        _buffer.AppendInterleaved(data, channels, AudioSettings.outputSampleRate);
    }

    public void Ask()
    {
        if (_index >= Words.Length) return;
        _judge.Begin(Words[_index]);        // 창을 비우고 시작
        _loop = StartCoroutine(Score());
    }

    System.Collections.IEnumerator Score()
    {
        float started = Time.time;

        while (_judge.IsActive)
        {
            yield return null;

            // 0.5초치가 모였을 때만 돕니다 (20~30ms, 메인 스레드)
            while (_judge.TryPush(out var frame))
            {
                if (frame.Confirmed)
                {
                    Debug.Log($"정답 {frame.Best.TargetText} {frame.Best.Score:F2}");
                    _index++;
                    Invoke(nameof(Ask), 1f);
                    yield break;
                }
            }

            if (Time.time - started > 15f)  // 오답 판정은 앱 몫
            {
                Debug.Log("다시 해볼까?");
                _judge.End();
            }
        }
    }

    public void Stop()
    {
        _judge.End();
        if (_loop != null) StopCoroutine(_loop);
    }

    void OnDestroy() => _recognizer?.Dispose();
}
```

오디오 출처가 `OnAudioFilterRead` 가 아니라면 그 줄만 바꾸세요 (방법 2·4 참고).

---

## API 요약

### `PronunciationSession` — 판정 (엔진·마이크 무관)

| 멤버 | 설명 |
|---|---|
| `new PronunciationSession(matrix, catalog, recognizer, consecutive = 2, buffer = null)` | 앱 수명 내내 하나 만들어 재사용 |
| `Begin(string word)` | 문제 시작. 그 단어만 채점 |
| `Begin(IReadOnlyList<string> words)` | 여러 정답 중 아무거나 |
| `Push(float[] window)` | 창 하나 채점 → `FrameScore` |
| `Push()` | (수동) 지금 버퍼 상태를 그대로 채점 — 부른 만큼 채점됨 |
| `TryPush(out FrameScore)` | hop 이 하나 찼으면 채점. 아니면 추론 없이 false |
| `End()` | 중단. 버퍼의 안 읽은 프레임·부분 hop 도 비움 (`LastConfirmation` 은 유지) |
| `IsActive` | `Begin` 과 확정/`End` 사이 |
| `TargetText`, `Candidates` | 지금 묻고 있는 것 |

`FrameScore.EndSampleIndex` 는 그 창이 끝난 16 kHz 샘플 번호입니다
(`TryPush` 로 얻은 프레임만; 직접 창을 넘겼으면 -1). 프레임 큐가 넘쳐
`TryPush` 가 예외를 던지면, 또는 **인식·채점 자체가 예외를 던지면** 세션도
무효가 됩니다. 예외를 잡고 계속 먹여도 옛 연속 횟수는 이어지지 않으며
(버퍼만 `Reset()` 해도 마찬가지), 다시 `Begin()` 해야 합니다 — 실패한
프레임 앞뒤를 붙여 "연속 2회" 라고 부를 수는 없기 때문입니다.
`LastConfirmation` 은 이때도 지우지 않습니다.

`Begin` 은 `buffer` 를 넘겼다면 그것도 비웁니다 — 직전 답이 다음 문제에
섞이지 않도록.

### `FrameScore` — `Push` 의 반환값

| 필드 | 설명 |
|---|---|
| `Confirmed` | 확정됨 (세션 종료) |
| `Feedback` | 확정 프레임의 PronunciationFeedback, 확정 전에는 null |
| `Streak` | 연속 횟수 |
| `Text` | 인식된 한글 (`""` 면 무음) |
| `Phonemes` | 인식된 IPA |
| `Best.TargetText` / `Best.Score` / `Best.Passed` | 단어·점수·이번 창 통과 여부 |
| `Best.Alignment` | `(사용자음소, 정답음소, 연산)` 목록 |
| `Best.WindowStart` / `WindowEnd` | 사용자 음소 중 실제로 매칭된 구간 |

### `AudioWindowBuffer` — 롤링 창 (마이크 안 엶)

| 멤버 | 설명 |
|---|---|
| `new AudioWindowBuffer(2.5f)` | 창 길이 = 모델이 받는 길이. hop 없음 = 종전과 동일 |
| `new AudioWindowBuffer(2.5f, 0.5f)` | hop 프레이밍 켜기 (새 샘플 0.5초 = 프레임 1개) |
| `Append(float[] mono, int rate)` | 모노 샘플 추가. 리샘플링 자동 |
| `AppendInterleaved(float[] data, int channels, int rate)` | 인터리브 다채널 |
| `Snapshot()` | 항상 40000개 (모자라면 앞을 무음으로) |
| `TryReadFrame(out HopFrame)` | 다음 hop 프레임. 아직 안 찼으면 false |
| `Reset()` | 오디오·대기 프레임·경계 위상·오류 상태 모두 비우기 |
| `IsFull` | 실제 오디오로 창이 다 찼는지 |
| `PendingFrames` / `Overflowed` | 안 읽은 프레임 수 / 큐가 넘쳤는지 |

`HopFrame` 은 `Window`(그 경계 시점의 창)와 `EndSampleIndex`(그때까지
들어온 16 kHz 샘플 수)를 가집니다. 프레임을 읽지 않고 쌓기만 하면
`maxPendingFrames`(기본 64, 기본 hop 으로 32초)에서 큐가 **비워지고
`TryReadFrame` 이 예외를 던집니다** — 조용히 버리면 아무도 채점하지 않은
오디오 위에 연속 횟수가 이어지기 때문입니다. `Reset()` 해야 풀립니다.

`Append` 는 오디오 스레드에서 불러도 안전합니다. `Snapshot`/`Push`/
`TryPush` 는 메인 스레드에서 부르세요.
메인 스레드에서 부르세요.

### `PronunciationListener` — 마이크까지 맡기는 컴포넌트

| 멤버 | 설명 |
|---|---|
| `SetRecognizer(IPhonemeRecognizer)` | `Listen` 전에 한 번 |
| `Listen()` / `StopListening()` | 마이크 켜기·끄기 |
| `IsListening`, `Session` | 상태, 내부 판정 객체 |
| `TargetWords` | 물어볼 단어. 여러 개면 그중 하나만 맞아도 확정 |
| `MicrophoneDevice` | 비우면 OS 기본 |
| `WindowSeconds` `HopSeconds` `Consecutive` `TimeoutSeconds` | 2.5 / 0.5 / 2 / 30 |
| `OnConfirmed(word, score)` | 확정 |
| `OnConfirmedDetailed(feedback)` / `LastConfirmation` | 확정 원본 비교·한글 설명, 다음 Listen 시도 시 초기화 |
| `OnTimedOut()` | 시간 초과 |
| `OnFrameScored(word, score, streak)` | 매 프레임 (진행 표시용) |

### `SentisPhonemeRecognizer` — 음향 모델

| 멤버 | 설명 |
|---|---|
| `new SentisPhonemeRecognizer(modelAsset, vocab)` | GPU 백엔드 고정 |
| `Warmup(float windowSeconds)` | 로딩 화면에서. 첫 추론 ~2초 |
| `Recognize` / `RecognizeWithText` | 오디오 → IPA (직접 부를 일은 드묾) |
| `Dispose()` | 워커 해제 |

### `PhonemeData` — JSON 로더

| 멤버 | 설명 |
|---|---|
| `LoadMatrix(json)` | confusion matrix |
| `LoadTargets(json)` | `targets.json` → `TargetCatalog` |
| `LoadCtcVocabulary(json)` | CTC 어휘 |

`TargetCatalog.Find(idOrText)` 로 단어가 등록돼 있는지 미리 확인할 수
있습니다 (없으면 `Begin` 이 예외를 던집니다).

---

## 자주 걸리는 것

| 증상 | 원인 |
|---|---|
| `targets.json has no 'answers' list` | 구버전 데이터. **데이터 파일 설치** 를 다시 누름 |
| `'포도' is not in targets.json` | `words.csv` 에 추가 후 **정답 데이터 다시 만들기** 를 안 누름 |
| `.onnx` 를 넣었는데 `ModelAsset` 으로 안 잡힘 | 추론 엔진보다 모델을 먼저 넣음. 엔진 설치 후 재임포트 |
| 첫 발화에서 2초 멈춤 | `Warmup()` 을 로딩 중에 안 부름 |
| 계속 무음, 점수 0 | 마이크 장치가 엉뚱하거나 Windows 개인정보 설정에서 차단 |
| `inference failed on N samples` | 창 길이와 모델 길이 불일치. `--static-samples` 로 다시 export |
| 말했는데 확정 안 됨 | 처음 통과한 프레임 뒤로 **`(연속횟수-1) × hop` = 0.5초** 를 더 들어야 함. 발화가 끝난 시점부터 1초라는 뜻이 아니라, 통과가 시작된 시점부터입니다 |
| 점수가 이상하게 낮음 | 앱이 넘기는 샘플레이트를 잘못 알려줌 (`Append` 의 두 번째 인자) |

## 배치와 스트리밍은 설정이 다릅니다

| | 배치 | 성인 모델 스트리밍 (v1) | 아동 모델 스트리밍 (v2) |
|---|---|---|---|
| `skip_cost` | 0.15 | 0.05 | 0.005 |
| 윈도우 커버리지 | 0.5 | 0.8 | 0.8 |
| 문맥 제한 | 없음 | 최근 `6×정답음소수+3` | 최근 `8×정답음소수+3` |
| 연속 확인 | 없음 | 2회 | 2회 |

현재 값은 각 matrix의 `streaming_profile`에서 읽습니다. 아래 골든셋 수치는
초기 성인 모델 실험이며, 아동 모델의 별도 검증은 위 아동 모델 항목을 참고하세요.

**하나로 통일할 수 없습니다.** 배치 설정을 스트리밍에 쓰면 정답 4개 중 3개를 놓치고,
스트리밍 설정을 배치에 쓰면 positives가 69.4%→47.2%로 무너집니다. 배치는 커버리지가
느슨해야 아빠→"앞" 같은 부분 인식을 살리는데, 자유 발화에서는 그 느슨함 때문에 다른
단어의 파편(과일→사과)이 임계값을 넘습니다.

### 연속 확인이 필요한 이유

스트리밍은 프레임마다 채점하므로 우연히 한 번 넘길 기회가 계속 생깁니다. 같은 답을
연속 N회 요구하면 그게 무너집니다 — 우연한 일치는 다음 프레임에 사라지지만, 진짜
발화는 롤링 창에 남아 계속 이깁니다.

골든 클립 36개로 스트리밍 세션을 만들어 스윕한 결과입니다(0.5초 hop, 각 세션에서
자기 단어와 나머지 17단어를 모두 시도).

| 연속 | 검출 | 오발동 | 확정까지 |
|---|---|---|---|
| 1회 | 18/36 | 19/612 | 1.5초 |
| **2회** | **18/36** | **16/612** | **2.1초** |
| 3회 | 15/36 | 10/612 | 2.5초 |

이 성인 TTS 실험에서는 2회가 1회와 검출은 같고 오발동은 적었습니다.
3회는 오발동을 약 1%p 줄이는 대신 검출이 약 8%p 낮았습니다. 현재 성인·아동
기본값은 모두 2회이며, 아동 모델은 별도의 실제 아동 발화로 설정을 검증했습니다.

### 문맥 제한은 연속 횟수와 묶여 있습니다

정답은 문맥 창 **안에 있는 동안만** 점수를 얻습니다. 첫 통과 뒤에도
`(연속횟수 - 1) × hop` 동안 같은 후보가 통과해야 합니다. 말을 계속 덧붙이면
정답 음소가 문맥 범위 밖으로 밀려날 수 있으므로, 연속 횟수를 올릴 때에는
문맥 제한과 실제 발화 속도를 함께 확인하세요. 현재 문맥 배수는 성인 6, 아동 8입니다.

## 음향 모델

`SentisPhonemeRecognizer` 가 wav2vec2 를 Sentis 로 돌립니다.

```csharp
var vocab = PhonemeData.LoadCtcVocabulary(vocabJson);
var recognizer = new SentisPhonemeRecognizer(modelAsset, vocab);
recognizer.Warmup(2.5f);        // 로딩 중에 부를 것 (아래)
listener.SetRecognizer(recognizer);
```

**첫 추론은 ~1.9초 걸립니다** — 셰이더 컴파일과 1.2GB 가중치 업로드입니다.
`Warmup()` 을 로딩 화면에서 부르지 않으면 사용자의 첫 발화가 그 시간을
기다립니다. 다 쓰면 `Dispose()` 로 워커를 놓아주세요.

백엔드는 **GPUCompute 고정이고 CPU 폴백이 없습니다.** 2.5초 창 실측(유휴
PC, RTX 5060 Ti): GPU 22~32ms, CPU 461~523ms. hop 예산이 500ms인데 CPU 는
그걸 거의 다 쓰면서 코어 6개를 점유합니다. VRAM 은 가중치로 1.2GB 를
씁니다.

다른 모델을 쓰려면 `IPhonemeRecognizer` 를 구현하면 됩니다. 주의: 출력 음소는
targets 를 만든 것과 **같은 표기 체계**여야 합니다. 한국어는 인식기의 한글
출력에 `Korean.Rules.Apply()` 후 `JamoIpa.ToPhonemes()`를 적용하면 됩니다.

## 테스트

2026-09-06: .NET C# 46개, Unity 2022.3.62f2 / 6000.3.18f1 각각 Editor 27개·PlayMode 47개 통과.
동일한 JSON 피드백 벡터와 24개 한글 설명 사례·전체 11,172음절 매핑을 Python/C#에서 대조합니다.
두 Unity 버전의 Windows x64 Mono 빌드·실제 GPU 추론도 통과했습니다.
초기 버퍼를 맞춘 웹의 아동 모델 결과는 58개 합성 음성/199개 창에서 앞선 Unity 실행 기록과
일치했습니다. 새 마이크·사용자의 게임 씬·Windows IL2CPP는 미검증이며, Unity 6 Editor
추론 종료 시 외부 엔진의 정적 버퍼 경고가 남아 있습니다. 자세한 범위와 설치 검증 방식은
[검증 기록](Documentation~/validation-2026-09-06.md)에 정리했습니다.

```bash
dotnet test csharp/PhonemeMatching.Tests    # Unity 없이, 초 단위
```

Unity Test Runner로도 같은 파일이 돕니다(NUnit).

`Tests/Runtime/parity_vectors.json`은 **파이썬 구현이 낸 기준값**입니다. 임계값·
`skip_cost`·스트리밍 프로파일은 전부 파이썬에서 녹음 데이터로 튜닝했으므로, C#이 조금이라도
다른 값을 내면 그 측정치가 무의미해집니다. 양쪽 다 그럴듯한 숫자를 반환하니 조용히
어긋나고요. 매처나 matrix를 고쳤다면 다시 생성하세요:

```bash
python -m python.tools.export_parity_vectors
```

`Tests/Runtime/ctc_vectors.json` 도 같은 목적입니다 — CTC 디코더가 파이썬
`batch_decode` 와 같은 글자를 내는지 43케이스로 확인합니다. 36개는 실제 ONNX
가 골든 클립에 대해 뱉은 토큰 id 라, 반복·blank 같은 까다로운 패턴이 실제
모델 출력에서 나옵니다. 모델이나 어휘를 바꿨다면:

```bash
python -m python.tools.export_ctc_vocab
```
