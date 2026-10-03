# Sentis 대조 검사 (일회용 하네스)

Unity 의 추론 엔진이 파이썬과 **같은 음소**를 내는지 확인하는 도구입니다.
평소에는 쓸 일이 없고, 아래 중 하나가 바뀌었을 때 다시 돌립니다.

- ONNX 모델을 다시 뽑았을 때 (`export_onnx.py`)
- Sentis / Inference Engine 버전을 올렸을 때
- 새 Unity 버전에서 처음 돌려볼 때

임포트가 됐다는 것과 **계산이 맞다는 것은 다른 문제**입니다. 2026-08-04 에
Sentis 1.2 는 동적 축 모델을 임포트까지 해놓고 실행에서 죽었고, 그 사실은
이 검사로 드러났습니다.

## 쓰는 법

**1. 기준값 만들기** (파이썬, ONNX Runtime)

```powershell
python python/tools/sentis_parity/export_static_cases.py
```

골든 클립 4개를 40000 샘플로 맞추고, 정규화하고, ONNX Runtime 으로 돌린
결과를 `audio_cases_static.json` 에 씁니다 - 원본 오디오(base64)와 프레임별
토큰 id, 디코딩된 한글이 함께 들어갑니다.

**2. Unity 에서 대조**

`SentisSmokeTest.cs.txt` 를 검사할 Unity 프로젝트의 `Assets/Editor/` 에
`.cs` 로 복사하고, 파일 안의 두 상수를 그 프로젝트에 맞게 고칩니다.

```csharp
private const string ModelPath = "Assets/Models/wav2vec2_ko_static.onnx";
private const string CasesPath = @"...\audio_cases_static.json";
```

`Tools > Sentis Smoke Test` 를 누르면 GPU·CPU 백엔드 각각에서 124 프레임의
토큰 id 를 전부 파이썬과 비교합니다.

> `.cs` 가 아니라 `.txt` 로 둔 이유: 이 저장소의 Unity 프로젝트가 컴파일하려
> 들면 `Unity.Sentis` 참조가 없어 깨집니다. 필요할 때만 복사해서 쓰세요.

## 마지막 결과 (2026-08-04)

Unity 2022.3.62f1 + Sentis 1.2.0-exp.2, 고정 40000 샘플 모델:

| 백엔드 | 일치 | 정상 프레임 | 첫 실행 |
|---|---|---|---|
| GPUCompute | 4/4 (124프레임 전부) | 30~31ms | 379ms |
| CPU | 4/4 | 385~396ms | 430ms |
