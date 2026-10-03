# 한국어 자모 인식 기반 모델의 Windows / Unity 실험

2026-09-14의 독립 실험 도구다. 현행 인식기나 아동 학습 도구의 교체본이 아니다.
결과와 대용량 파일은 저장소의 `.venv/encoder-screen-0914/`에 둔다.
모델 가중치·음성·Unity 프로젝트 및 실행 파일은 Git에 추가하지 않는다.

**2026-10-03 정리:** 실험 결과·설정·선정 목록·평가 기록·소스는 보존하고,
과거 후보 가중치·시험 학습 저장본·특징 배열·변환 모델·Unity 빌드/Library·추출 음성 사본은 제거했다.
현재 아동 자모 모델이 의존하는 `models/hubertko1b/`는 보존했다.
[현재 보존 범위와 삭제 기록](../../../docs/EXPERIMENT_ARTIFACT_CLEANUP_2026-10-03.md).
아래 명령은 당시 실험의 기록이며, 다시 실행하려면 해당 모델과 캐시를 새로 준비해야 한다.

완료 결과는 [Windows / Unity 실측 보고서](../../../docs/PRETRAINED_ENCODER_WINDOWS_UNITY_RESULTS_2026-09-14.md)에 정리했다.

## 비교 조건

- Windows PC, RTX 5060 Ti 16GB. 실제 Unity GPUCompute 실행을 확인한다.
- 주 환경: Unity 6000.3.18f1 / Inference Engine 2.6.1 / Windows x64 Mono / Direct3D 11.
- 보조 환경: Unity 2022.3.62f2 / Sentis 1.2.0-exp.2. 별도 프로젝트에서 대표 후보를 확인한다.
- 16kHz mono, 2.5초 입력과 0.5초 hop. GPU 결과를 읽어 올 때까지 동기화해 측정한다.
  p95 500ms 이내는 hop을 따라가는 조건이다. 게임 프레임 예산을 만족한다는 뜻이 아니다.
- 아동 자료를 먼저 화자별로 분리하고 5·6·7세를 균등하게 선택했다.
  train 5,999개 / valid 600개 / test 600개. 한 개의 손상 WAV는 모든 모델에서 제외한다.
- 발음 규칙을 적용한 문장을 자모로 분해한다. 자모 40개 + 경계 + CTC blank, 총 42개다.
  사람의 실제 발음 오류를 듣고 붙인 라벨은 아니다. 무음 초성 ㅇ도 포함하므로 JER를 IPA PER로 부르지 않는다.
- 고정 본체 + 선형 CTC 출력층: 같은 1,600 update, batch 32, seed 917,
  learning rate 0.001 / 0.003을 valid로 선택하고 test를 한 번 평가한다.
- 출력의 전체 크기가 모델별로 크게 달라 원래 출력과 학습 자료에서 구한 단일 스칼라로
  크기를 보정한 출력을 각각 비교한다. 채널 간 상대 크기는 유지한다. 두 표현 × 두 학습률,
  총 네 조건 중 valid가 가장 좋은 출력층을 후속 학습에 쓴다. 채널별 중심/분산을 바꾼
  첫 보완 실험은 한국어 HuBERT에서 악화되어 `experiments/per-channel-head/`에 보존했다.
  그 실험에서 시작한 본체 학습은 150 update 이후 중단하고 주 비교에서 제외했다.
- 후속 소규모 본체 학습은 별도 `finetune.py`에서 수행한다. 같은 학습 순서,
  batch 8(1개씩 누적), 400 update, encoder LR 2e-5 / head LR 1e-4,
  CNN 고정, 나머지 인코더와 출력층 갱신, BF16 학습 / FP32 평가다.
  완전 수렴한 모델의 우열을 확정하는 실험이 아니다.
- 보유 단어 53개와 앞뒤 무음·잡음 파생 입력은 학습 자료와 구분한다.
  아동 표본으로 세지 않는다. 사용자 성인 7개, 두 화자의 연령 미확인 46개다.

## 실행 파일과 주요 도구

이 작업 공간의 `.venv-gpu/Scripts/python.exe -X utf8`를 사용했다. 정확한 버전은
실험 폴더의 `environment.json`에 기록한다. ESPnet는 `202412`, typeguard는
`2.13.3`을 의존성 자동 갱신 없이 추가했다. XEUS의 두 보완 모듈은
`vendor-lock.json`의 공식 저장소 revision과 SHA256으로 고정한다.

| 파일 | 용도 |
|---|---|
| `download.py` | 공개 인코더 11개의 가중치와 설정 확보, revision·LFS SHA 검증 |
| `prepare.py` | 기존 Desktop AIHUB정리 index/splits와 AIHUB 원본 tar에서 고정 분할 추출 |
| `probe.py`, `run.py` | 인코더 FP32 특징 추출, 고정 본체의 한국어 CTC 비교 |
| `normalized_probe.py` | 학습 자료만으로 구한 단일 스칼라 보정의 추가 비교 |
| `convert_omni.py` | fairseq2 가중치의 명시적 매핑과 독립 수식 대조 |
| `xeus.py`, `vendor.py` | 가중치 누락을 허용하지 않는 ESPnet 어댑터 |
| `frontend.py`, `export.py` | 파형 전처리를 포함한 ONNX와 전체 logit 비교 자료 생성 |
| `deploy.py` | 독립 Unity 6 프로젝트의 실제 importer와 Windows player 실행 |
| `setup_legacy.py`, `legacy.py` | 별도의 Unity 2022 / Sentis 1.2 호환성 검사 |
| `finetune.py`, `domain.py` | 소규모 본체 학습 및 보유 단어의 별도 검사 |
| `offload.py`, `verify_offload.py` | 1B의 일부 AdamW 상태/연산을 CPU에 두고 동일 갱신 수식 검증 |
| `phase2.py` ~ `phase5.py` | GPU 실험 순차 실행, 대표 모델의 두 번째 학습 seed 확인 |
| `collect.py` | 집계와 화자 단위 탐색적 부트스트랩 |
| `parallel_heads.py` | 작은 출력층 학습 병행, `phase3.py`의 barrier 이후 Unity 속도 측정 |
| `fold_shapes.py`, `phase6.py` | W2v-BERT 고정 크기 연산 진단과 최종 후보의 구형 Unity 검사 |
| `publish.py` | 음성·개별 화자/파일 목록을 제외한 JSON과 결과 표 생성 |
| `deployed_words.py`, `phase7.py` | 최종 후보에 보유 녹음 53개를 두 위치의 2.5초 창으로 넣어 실제 GPU 검사 |

예시:

```powershell
.venv-gpu/Scripts/python.exe -X utf8 python/tools/encoder_screen/prepare.py
.venv-gpu/Scripts/python.exe -X utf8 python/tools/encoder_screen/probe.py hubertko
.venv-gpu/Scripts/python.exe -X utf8 python/tools/encoder_screen/export.py hubertko
.venv-gpu/Scripts/python.exe -X utf8 python/tools/encoder_screen/deploy.py convert hubertko
.venv-gpu/Scripts/python.exe -X utf8 python/tools/encoder_screen/deploy.py player hubertko
.venv-gpu/Scripts/python.exe -X utf8 python/tools/encoder_screen/finetune.py hubertko
```

`deploy.py`는 준비된 독립 Unity 프로젝트/빌드가 필요하다. `unity/`의 Runtime와 Memory
C# 템플릿을 Assets, Editor 템플릿을 Assets/Editor에 놓고 `EncoderScreenEditor.Build`로 빌드한다.
사용자 게임 프로젝트, 기존 배포 패키지, 활성 MCP 프로젝트를 지정하지 않는다.
원본 음성 자료와 기존 분할·라벨 파일이 필요한 작업 공간용 재현 도구다.
`publish.py --draft`는 실험 폴더에 중간 집계를 만든다. 인수 없는 실행은 모든 단계
완료를 확인한 뒤 `docs/data/`에 집계를 기록한다. 공개 집계만으로 원본 음성을 재구성할 수는 없다.

## 결과를 해석할 때

고정 본체의 점수, 400 update의 적응 점수, 최종 제품 정확도는 서로 다른 단계다.
정답률·오확정률은 서비스의 단어 경계, VAD, 임계값, 연속 판정까지 연결한 별도 측정이 필요하다.
문장 전사로 생성한 라벨만으로 발음 오류 안내의 정확성을 입증할 수 없다.

Unity parity 입력은 아동 파일 8개를 2.5초로 자르거나 채운 입력과 변형/비음성 4개다.
문장 전체 인식률을 계산하는 입력이 아니다. 한 개라도 기준 오차를 넘으면 원래 측정치와
실패 표시를 보존한다. 특히 ONNX의 절대 오차, 상대 오차, 프레임 argmax와 CTC 일치는 각각 기록한다.
Runtime JSON의 `success`는 유한한 출력·CTC 일치·0.5초 hop 조건을 합친 값이다.
절대오차 0.02 충족 여부는 별도 항목이므로 `success: true`만 보고 모든 수치 검사를
통과했다고 해석하면 안 된다. 최종 집계에는 원래 절대오차 기준의 충족 여부도 넣는다.
`functional-concurrent`라는 이름의 조기 smoke 결과는 다른 GPU 작업과 겹쳤으므로
속도 비교에는 쓰지 않는다. 정식 지연 측정은 GPU 실험을 순차 실행하는 구간에서 한다.
후반의 작은 출력층 비교는 본체 학습과 병행했으며 파일 lock으로 중복 학습을 막았다.
Unity 실행 직전에 별도 barrier로 이 작업의 종료를 확인한다. 학습 경과 시간에는
다른 출력층·CPU 진단 작업의 영향과 검증 시간이 포함돼 순수 처리량으로 비교하지 않는다.
W2v-BERT 보완 그래프의 조기 기능 검사도 WavLM 학습 중 실행했다. 그 기록은
`functional-concurrent`로 구분하고 지연 비교에서 제외했으며, 이후 별도 실행도 확인했다.

Release/Mono에서 Unity Profiler와 `Process.PrivateMemorySize64`가 0을 반환한 조기 결과가
있다. 이를 메모리 사용 0으로 해석하지 않는다. 최종 실행 파일은 Windows
`GetProcessMemoryInfo`의 작업 집합/전용 커밋과 DXGI `QueryVideoMemoryInfo`의 해당
프로세스 GPU local/nonlocal 사용량을 기록한다. DXGI 값은 쿼리 시점의 사용량이고
순간 최대 VRAM이나 다른 PC에서의 최소 사양을 보장하는 값은 아니다.
Worker 내부의 CPU fallback 할당도 검사해 GPUCompute 지정만으로 성공을 단정하지 않는다.

MMS/XEUS의 NC 조건, WavLM 가중치의 라이선스 확인 사항은 모델 카드와 조사 문서에 남긴다.
인코더 라이선스와 추가 학습 자료의 이용 조건은 별도로 확인해야 한다.
