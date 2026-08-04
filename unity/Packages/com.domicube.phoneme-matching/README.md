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
    "com.domicube.phoneme-matching": "https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching#v0.2.0",
    "com.unity.sentis": "1.2.0-exp.2"
  }
}
```

Unity 6:

```json
{
  "dependencies": {
    "com.domicube.phoneme-matching": "https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching#v0.2.0",
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

`#v0.2.0` 을 빼면 항상 main 최신을 당겨옵니다. 로컬에서 고쳐가며 쓰려면
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

## 사용법

### 녹음 하나 채점 (배치)

```csharp
var matrix  = PhonemeData.LoadMatrix(matrixJson);
var catalog = PhonemeData.LoadTargets(targetsJson);
var matcher = new Matcher(matrix);

var result = matcher.BestMatch(userPhonemes, catalog.SegmentOf("사과"));
if (result.Passed) { /* 정답 */ }
```

### 연속 청취 (VR 흐름)

```csharp
var sm = new StreamingMatcher(
    Matcher.ForStreaming(matrix), catalog.SegmentOf("사과"), consecutive: 3);

// 매 hop마다: 최근 2.5초 오디오를 통째로 재인식해서 넣는다
var hit = sm.Push(recognizer.Recognize(recentWindow));
if (hit != null) { StopRecording(); }
```

**청크를 따로 인식해 이어붙이지 마세요.** wav2vec2는 문맥 모델이라 짧은 조각에서
무너집니다 — 4.4초 발화에서 0.5초 청크는 43음소가 31음소로 줄고 단어가 깨졌습니다.
매 프레임 **최근 창 전체를 재인식**해야 합니다.

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
