# Phoneme Matching (Unity 패키지)

발화가 정답 단어와 **음소(IPA) 수준에서** 충분히 가까운지 판정합니다. STT로 "무슨
단어인지" 맞추는 게 아니라 음소 부분 매칭이라, 발음이 부정확한 사용자(아동·발달지연
포함)에게도 동작하고 STT 환각·마이크 잡음 문제를 함께 회피합니다.

**온디바이스 전용입니다.** 서버도 파이썬 런타임도 필요 없습니다.

## 실행 구조

```
[개발 PC · 오프라인]
  words.csv ──(파이썬 g2pkk)──> targets.json     ← 앱에 동봉하는 데이터

[기기 · 런타임]
  마이크 → IPhonemeRecognizer → IPA → Matcher → 통과/재시도
```

음운 규칙(g2pkk)은 **빌드 타임에만** 씁니다. 런타임은 유니코드 한글 분해 + 자모→IPA
테이블만 쓰므로 mecab이 필요 없습니다. 규칙 생략의 대가는 측정했습니다 — 골든셋에서
18단어 중 3개만 달라지고 negatives 거절율이 91.7%→90.3%로 1.4%p 떨어질 뿐입니다.
달라지는 세 쌍(`ɾ/l`, `k/k̚`, `k̚/k͈`)이 confusion matrix가 이미 거의 무료로 처리하는
쌍이기 때문입니다.

## 설치

`Packages/manifest.json` 에 한 줄. Sentis 와 Newtonsoft 는 패키지가 의존성으로
선언하므로 UPM 이 알아서 끌어옵니다.

**Unity 2022.3 과 Unity 6 을 모두 지원합니다.** 패키지 줄은 동일하고,
추론 엔진 한 줄만 에디터에 맞춰 고릅니다.

Unity 2022.3:

```json
{
  "dependencies": {
    "com.domicube.phoneme-matching": "https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching#v0.3.0",
    "com.unity.sentis": "1.2.0-exp.2"
  }
}
```

Unity 6:

```json
{
  "dependencies": {
    "com.domicube.phoneme-matching": "https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching#v0.3.0",
    "com.unity.ai.inference": "2.6.1"
  }
}
```

엔진을 패키지 의존성에 넣지 않은 이유는 UPM 이 "A 또는 B" 를 표현하지
못하기 때문입니다. 2.x 를 박으면 2022 에서 설치가 실패하고, 1.2 를 박으면
Unity 6 사용자가 중단된 실험 패키지를 강제로 받게 됩니다.

설치된 엔진에 맞는 인식기만 컴파일됩니다 (asmdef 의 `versionDefines` +
`defineConstraints`). 클래스 이름은 양쪽 다
`SentisPhonemeRecognizer` 라서 호출 코드는 동일하고, `ModelAsset` 을
선언할 때의 `using` 만 다릅니다 - 2022 는 `Unity.Sentis`, Unity 6 은
`Unity.InferenceEngine`.

`#v0.3.0` 을 빼면 항상 main 최신을 당겨옵니다. 로컬에서 고쳐가며 쓰려면
`"file:../../path/to/com.domicube.phoneme-matching"` 도 됩니다.

### 데이터와 모델은 따로 옵니다

코드만 받아서는 동작하지 않습니다. 두 가지를 더 넣어야 합니다.

1. **데이터** — Package Manager 에서 이 패키지의 샘플
   `Korean data (ko_child_v1)` 을 임포트한 뒤, 세 JSON 을
   `Assets/StreamingAssets/` 로 복사하세요.
2. **음향 모델** — 1.18GB 라 패키지에 없습니다. 저장소에서 만드세요:

```powershell
python -m python.tools.export_onnx
```

`shared/models/wav2vec2_ko.onnx` 가 나옵니다. `Assets/` 아래 두면 Sentis
가 임포트합니다.

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

var recognizer = new SentisPhonemeRecognizer(Model, vocab);   // Model = ModelAsset
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
listener.TargetText = "사과";

listener.OnConfirmed.AddListener((word, score) => 다음문제로());
listener.OnTimedOut.AddListener(() => 다시해볼까());

listener.Listen();          // 마이크 켜짐
listener.StopListening();   // 끄기 (확정·타임아웃에도 자동으로 꺼짐)
```

창 길이·hop·확정 횟수·타임아웃은 인스펙터에서 조정합니다. 장치 이름은
`Microphone.devices` 로 확인하세요 - VR 에서는 OS 기본이 헤드셋 마이크가
아닌 경우가 많습니다.

---

### 방법 2~4 공통 — 앱이 오디오를 준다

리듬이 둘로 나뉩니다. **오는 대로 붓고, 0.5초마다 채점합니다.**

```
앱의 오디오 ──Append()──▶ [ 2.5초 창 ] ──Push()──▶ 채점
            오는 대로 계속                0.5초마다
```

```csharp
// 한 번 만들어 계속 씁니다. Begin ~ End 한 번이 문제 하나입니다.
var buffer  = new AudioWindowBuffer(2.5f);
var session = new PronunciationSession(matrix, catalog, recognizer,
                                       buffer: buffer);

// 문제 낼 때 — 창을 비우고 시작합니다
session.Begin("사과");                      // 같은 세그먼트가 경쟁
// session.Begin(new[] { "사과", "빵" });   // 지정한 것만 경쟁

// 0.5초마다
var frame = session.Push();
if (frame.Confirmed) 다음문제로();

session.End();   // 중단
```

`Push()` 는 메인 스레드에서 부르세요 - 20~30ms 걸리고 GPU 를 씁니다.
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

방법 2~4 는 `Push()` 가 돌려주고, 방법 1 은 `OnFrameScored` 로 옵니다.

```csharp
frame.Confirmed        // true 면 확정 - 이 순간 세션 종료
frame.Streak           // 연속 몇 번째 (기본 3 이면 확정)
frame.Best.TargetText  // 지금 가장 유력한 단어
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

## 배치와 스트리밍은 설정이 다릅니다

| | 배치 | 스트리밍 |
|---|---|---|
| `skip_cost` | 0.15 | 0.05 |
| 윈도우 커버리지 | 0.5 | 0.8 |
| 문맥 제한 | 없음 | 최근 `4×정답음소수+3` |
| 연속 확인 | 없음 | 3회 |

**하나로 통일할 수 없습니다.** 배치 설정을 스트리밍에 쓰면 정답 4개 중 3개를 놓치고,
스트리밍 설정을 배치에 쓰면 positives가 69.4%→47.2%로 무너집니다. 배치는 커버리지가
느슨해야 아빠→"앞" 같은 부분 인식을 살리는데, 자유 발화에서는 그 느슨함 때문에 다른
단어의 파편(과일→사과)이 임계값을 넘습니다.

### 연속 확인이 필요한 이유

스트리밍은 프레임마다 채점하므로 기회가 `프레임 수 × 후보 수`만큼 생깁니다. 0.4초
hop으로 10초, 후보 4개면 100회이고, 프레임당 오검출률 1%도 세션 단위로는 63%가 됩니다.
같은 단어로 연속 N회를 요구하면 이게 무너집니다 — 우연한 일치는 다음 프레임에 다른
후보로 옮겨가지만, 진짜 발화는 롤링 창에 남아 계속 이깁니다.

### 문맥 제한은 연속 횟수와 묶여 있습니다

정답은 문맥 창 **안에 있는 동안만** 점수를 얻습니다. 창이 `연속횟수 × hop`초보다 짧으면
말을 계속하는 사용자는 확정이 산술적으로 불가능합니다. 한국어는 약 10음소/초라
`context_mult 2.0`(사과 기준 13음소 ≈ 1.3초 ≈ 2.6프레임)으로는 연속 3회를 못 채웠고,
4.0(≈2.3초 ≈ 4.6프레임)으로 해결했습니다. 값을 바꿀 땐 이 관계를 먼저 확인하세요.

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
출력을 `JamoIpa.ToPhonemes()` 에 통과시키면 됩니다.

## 테스트

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
