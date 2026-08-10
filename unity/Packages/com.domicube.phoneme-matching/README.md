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

**Unity 2022.3 과 Unity 6 을 모두 지원합니다.**

패키지를 넣은 뒤 `Tools > Phoneme Matching` 메뉴를 **위에서 아래로** 세 번
누르면 끝납니다. 순서대로 해야 합니다 - 추론 엔진이 있어야 `.onnx` 가
`ModelAsset` 으로 임포트됩니다.

```
추론 엔진 설치              ← 에디터 버전에 맞는 것을 알아서 고릅니다
데이터 파일 설치            ← JSON 3개를 StreamingAssets 로
음향 모델 내려받기           ← 1.18GB, Release 에서
음향 모델 파일에서 가져오기…  ← 이미 파일이 있을 때
────────────────────────
발음 테스트 (마이크)         ← 여기까지 되면 설치 완료
────────────────────────
단어 목록 열기 (words.csv)   ┐ 정답 단어를 바꿀 때만.
정답 데이터 다시 만들기       ┘ 저장소와 파이썬이 필요합니다
────────────────────────
저장소 경로 지정…
```

### 1. 패키지

`Window > Package Manager` > `+` > **Add package from git URL…**

```
https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching
```

`manifest.json` 에 직접 적어도 같습니다:

```json
"com.domicube.phoneme-matching": "https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching"
```

항상 main 최신을 받습니다. 특정 버전에 고정하려면 뒤에 `#v0.5.0` 처럼
태그를 붙이세요. 로컬에서 고쳐가며 쓰려면
`"file:../../path/to/com.domicube.phoneme-matching"` 도 됩니다.

### 2. 추론 엔진

`Tools > Phoneme Matching > 추론 엔진 설치`

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

`ko_child_v1.json` · `targets.json` · `wav2vec2_ko_vocab.json` 을
`Assets/StreamingAssets/` 에 넣고, 런타임이 읽는 방식 그대로 다시 읽어
확인까지 합니다. 이미 있으면 덮어쓸지 물어봅니다.

> Package Manager 의 샘플 임포트 버튼을 써도 되지만, 그건 파일을
> `Assets/Samples/...` 에 놓기 때문에 `StreamingAssets/` 로 한 번 더
> 옮겨야 합니다. 위 메뉴가 그 단계를 없앤 것입니다.

단어를 바꿨다면 `데이터 파일 설치` 대신
`정답 데이터 다시 만들기` 를 쓰세요 — 저장소의 `words.csv` 로 빌드해서
`targets.json` 만 갱신합니다.

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
session.Begin("사과");                      // 이 단어만 채점
// session.Begin(new[] { "사과", "능금" }); // 둘 중 아무거나 정답

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
frame.Streak           // 연속 몇 번째 (기본 2 면 확정)
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
        _listener.TargetText = Words[_index];
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
        _buffer = new AudioWindowBuffer(2.5f);
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
        var wait = new WaitForSeconds(0.5f);
        float started = Time.time;

        while (_judge.IsActive)
        {
            yield return wait;

            var frame = _judge.Push();      // 20~30ms, 메인 스레드

            if (frame.Confirmed)
            {
                Debug.Log($"정답 {frame.Best.TargetText} {frame.Best.Score:F2}");
                _index++;
                Invoke(nameof(Ask), 1f);
                yield break;
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
| `Push()` | 생성자에 넘긴 버퍼에서 가져와 채점 |
| `End()` | 중단 |
| `IsActive` | `Begin` 과 확정/`End` 사이 |
| `TargetText`, `Candidates` | 지금 묻고 있는 것 |

`Begin` 은 `buffer` 를 넘겼다면 그것도 비웁니다 — 직전 답이 다음 문제에
섞이지 않도록.

### `FrameScore` — `Push` 의 반환값

| 필드 | 설명 |
|---|---|
| `Confirmed` | 확정됨 (세션 종료) |
| `Streak` | 연속 횟수 |
| `Text` | 인식된 한글 (`""` 면 무음) |
| `Phonemes` | 인식된 IPA |
| `Best.TargetText` / `Best.Score` / `Best.Passed` | 단어·점수·이번 창 통과 여부 |
| `Best.Alignment` | `(사용자음소, 정답음소, 연산)` 목록 |
| `Best.WindowStart` / `WindowEnd` | 사용자 음소 중 실제로 매칭된 구간 |

### `AudioWindowBuffer` — 롤링 창 (마이크 안 엶)

| 멤버 | 설명 |
|---|---|
| `new AudioWindowBuffer(2.5f)` | 창 길이 = 모델이 받는 길이 |
| `Append(float[] mono, int rate)` | 모노 샘플 추가. 리샘플링 자동 |
| `AppendInterleaved(float[] data, int channels, int rate)` | 인터리브 다채널 |
| `Snapshot()` | 항상 40000개 (모자라면 앞을 무음으로) |
| `Reset()` | 비우기 |
| `IsFull` | 실제 오디오로 창이 다 찼는지 |

`Append` 는 오디오 스레드에서 불러도 안전합니다. `Snapshot`/`Push` 는
메인 스레드에서 부르세요.

### `PronunciationListener` — 마이크까지 맡기는 컴포넌트

| 멤버 | 설명 |
|---|---|
| `SetRecognizer(IPhonemeRecognizer)` | `Listen` 전에 한 번 |
| `Listen()` / `StopListening()` | 마이크 켜기·끄기 |
| `IsListening`, `Session` | 상태, 내부 판정 객체 |
| `TargetText` | 물어볼 단어 |
| `MicrophoneDevice` | 비우면 OS 기본 |
| `WindowSeconds` `HopSeconds` `Consecutive` `TimeoutSeconds` | 2.5 / 0.5 / 2 / 30 |
| `OnConfirmed(word, score)` | 확정 |
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
| 말했는데 확정 안 됨 | 발화 후 **1초는 더** 들어야 함 (연속 2회 × hop 0.5초) |
| 점수가 이상하게 낮음 | 앱이 넘기는 샘플레이트를 잘못 알려줌 (`Append` 의 두 번째 인자) |

## 배치와 스트리밍은 설정이 다릅니다

| | 배치 | 스트리밍 |
|---|---|---|
| `skip_cost` | 0.15 | 0.05 |
| 윈도우 커버리지 | 0.5 | 0.8 |
| 문맥 제한 | 없음 | 최근 `4×정답음소수+3` |
| 연속 확인 | 없음 | 2회 |

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

2회는 1회보다 무조건 낫고(검출 같고 오발동만 줄어듦), 3회는 오발동 1%p를 얻는 대신
검출 8%p를 잃습니다. 제대로 말했는데 못 알아듣는 쪽이 더 나쁘므로 기본값은 2입니다.
성인 TTS 기준이라, 아이 발화로는 다시 재봐야 합니다.

### 문맥 제한은 연속 횟수와 묶여 있습니다

정답은 문맥 창 **안에 있는 동안만** 점수를 얻습니다. 창이 `연속횟수 × hop`초보다 짧으면
말을 계속하는 사용자는 확정이 산술적으로 불가능합니다. 한국어는 약 10음소/초라
`context_mult 2.0`(사과 기준 13음소 ≈ 1.3초 ≈ 2.6프레임)으로는 연속 확인을 못 채웠고,
4.0(≈2.3초 ≈ 4.6프레임)으로 해결했습니다. 연속 횟수를 올릴 땐 이 관계를 먼저
확인하세요.

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
