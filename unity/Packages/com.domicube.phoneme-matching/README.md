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

`Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.domicube.phoneme-matching": "file:../../path/to/com.domicube.phoneme-matching",
    "com.unity.nuget.newtonsoft-json": "3.2.1"
  }
}
```

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

`IPhonemeRecognizer` 구현은 아직 없습니다(ONNX export는 별도 단계). 매칭 계층은 모델
없이도 완성돼 있고, 실제 wav2vec2 출력을 고정한 픽스처로 테스트됩니다.

구현할 때 주의: 출력 음소는 targets를 만든 것과 **같은 표기 체계**여야 합니다. 한국어는
인식기의 한글 출력을 `JamoIpa.ToPhonemes()`에 통과시키면 됩니다.

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
