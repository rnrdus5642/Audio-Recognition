# Audio Recognition (Korean Phoneme Matching)

VR 아동 음성 인식 시스템 프로토타입.

발음이 부정확한 사용자(아동, 발달지연 포함)의 발화가 정답 단어와 음소(IPA) 수준에서 충분히 가까운지 판정합니다. STT가 아니라 음소 기반 부분 매칭으로 환각 문제와 마이크 잡음 문제를 모두 회피합니다.

## 아키텍처

```
정답 단어 (텍스트)                       🎤 마이크 / WAV
       ↓                                       ↓
  g2pkk + jamo + IPA 매핑              wav2vec2 ASR (한글)
       ↓                                       ↓
   정답 IPA            ←─ 동일한 ─→     g2pkk + jamo + IPA 매핑
       ↓                IPA 표기                ↓
       └────────── 가중 부분 문자열 편집 거리 ──────┘
                   + Confusion Matrix
                   + 윈도우 길이 페널티
                          ↓
                   통과 / 재시도
```

핵심: **정답 측과 사용자 측이 동일한 g2pkk + 매핑 테이블**을 통과해서 IPA 표기 체계가 일치. 매칭이 신뢰 가능.

- **웹 UI**: 단어를 즉석에서 g2p 돌려 매칭 (사전 빌드 불필요)
- **CLI 도구**: `words.csv → targets.json` 사전 빌드 후 일괄 평가/녹음에 활용

## 디렉토리 구조

**언어별 코드는 모두 `<lang>/` 폴더로 격리**. 새 언어 추가 시 폴더 추가 + REGISTRY 한 줄 등록. 자세한 가이드: [docs/ADDING_A_LANGUAGE.md](docs/ADDING_A_LANGUAGE.md)

```
AudioProject/
├── python/
│   ├── build/                       # 빌드 타임 (G2P + targets.json 생성)
│   │   ├── g2p/
│   │   │   ├── __init__.py          # G2P_REGISTRY (언어 등록)
│   │   │   ├── base.py              # BaseG2P 인터페이스 (공통)
│   │   │   └── ko/                  # ── 한국어 ──
│   │   │       ├── g2p.py           #    KoreanG2P
│   │   │       └── jamo_ipa.py      #    자모↔IPA 매핑
│   │   └── build_targets.py         # CSV → JSON (CLI 도구용)
│   │
│   ├── runtime/                     # 런타임 (인식 + 매칭)
│   │   ├── audio.py                 # 16kHz mono 로딩 (공통)
│   │   ├── matching/                # 가중 부분 문자열 편집 거리 (공통)
│   │   │   ├── confusion_matrix.py
│   │   │   ├── matcher.py
│   │   │   └── streaming.py         # 연속 청취 (롤링 + 연속 확인)
│   │   └── recognizer/
│   │       ├── __init__.py          # RECOGNIZER_REGISTRY (언어 등록)
│   │       ├── base.py              # BaseRecognizer 인터페이스 (공통)
│   │       └── ko/                  # ── 한국어 ──
│   │           └── asr.py           #    wav2vec2 + g2pkk
│   │
│   ├── tools/                       # 사용자 도구 (대부분 언어 무관)
│   │   ├── web_test.py              # 🌐 웹 UI (권장)
│   │   ├── _web_builder.py          #    정답 데이터 추출 헬퍼
│   │   ├── record_live.py           # 💻 CLI 실시간 녹음
│   │   ├── test_real_audio.py       # 📁 단일/일괄 파일 테스트
│   │   ├── test_streaming.py        # 🔁 연속 청취 시뮬레이션
│   │   ├── diagnose_audio.py        # 🔍 오디오 진단
│   │   ├── generate_golden_audio.py # Edge-TTS 골든셋 생성
│   │   ├── evaluate.py              # 골든셋 정확도 평가
│   │   ├── export_parity_vectors.py # C# 대조용 기준 벡터 생성
│   │   ├── export_onnx.py           # ONNX export + Sentis 호환성 검증
│   │   └── export_ctc_vocab.py      # CTC 어휘 + 디코드 대조 벡터
│   │
│   └── tests/
│       ├── test_g2p_ko.py           # 언어별 (ko)
│       ├── test_jamo_ipa_ko.py      # 언어별 (ko)
│       ├── test_matcher.py          # 공통
│       └── test_streaming.py        # 연속 청취 (실제 ASR 프레임 픽스처)
│
├── shared/                          # 데이터
│   ├── words.csv                    # 정답 단어 (CLI 빌드 입력)
│   ├── targets.json                 # 빌드 산출물 (CLI 도구용)
│   ├── confusion_matrices/
│   │   └── ko_child_v1.json         # 런타임 매칭의 핵심 자원
│   └── models/                      # ONNX 모델 (Phase 3+)
│
├── docs/
│   └── ADDING_A_LANGUAGE.md
│
├── unity/                           # Unity 6000.3 / URP / Sentis 2.6
│   ├── Assets/
│   │   ├── Models/wav2vec2_ko.onnx  # gitignore (1.18GB, export_onnx.py)
│   │   ├── Scripts/PronunciationDemo.cs
│   │   └── StreamingAssets/         # matrix + targets + vocab JSON (앱 동봉)
│   └── Packages/com.domicube.phoneme-matching/   # UPM 패키지 (임베디드)
│       ├── Runtime/                 # 코어 (엔진 참조 없음)
│       │   ├── CtcDecoder.cs        #   greedy CTC (파이썬 batch_decode 대응)
│       │   └── Unity/               #   마이크·리스너·Sentis (별도 asmdef)
│       └── Tests/Runtime/           # 파이썬 대조 벡터
│
└── csharp/PhonemeMatching.Tests/    # Unity 없이 dotnet test
```

**언어 추가 시 변경**:
- 신규: `python/build/g2p/<lang>/`, `python/runtime/recognizer/<lang>/`, `shared/confusion_matrices/<lang>_*.json`
- 수정: `__init__.py` 2개에 2줄씩
- **건드리지 않음**: 매칭 엔진, 도구, Unity 코드, 웹 UI

## 셋업

```powershell
# 가상환경 생성 (Python 3.10~3.13 호환)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Phase 1 의존성 (빌드 + 테스트)
pip install -r requirements.txt

# Phase 2 의존성 (모델 + 웹 UI + 녹음 등)
pip install -r requirements-phase2.txt
pip install gradio sounddevice
```

> 64-bit conda 사용자는 `environment.yml`로도 가능 (참고 파일).

## 사용법

### 🌐 웹 UI (권장)

```powershell
python -m python.tools.web_test
```
브라우저 자동 오픈 (`http://127.0.0.1:7860`). 첫 인식 시 wav2vec2 모델(~1.2GB)이 로드되어 1~2분 걸립니다. 이후는 빠릅니다.

네 개의 탭:

**🎤 발음 테스트** — 단어 매칭
- 모드 라디오: `직접 입력` (정답 단어를 즉석 타이핑) / `ASR 결과만 보기` (한글·IPA만 출력, 점수 없음)
- 마이크 녹음 또는 파일 업로드
- 결과: 통과/실패 배너, 오디오 통계, ASR 한글·IPA, 음소 정렬 시각화, 세션 기록
- 💾 저장 버튼: `recordings/web_YYYYMMDD/`에 wav + 결과 JSON 보관

**🔁 연속 청취** — VR 흐름 재현
- 정답 단어를 여러 개 입력 (한 줄에 하나) + 녹음/업로드
- 녹음을 롤링 윈도우로 재생하며 프레임마다 재인식 → 연속 N회 이기면 확정
- 차트 위: **후보 단어별 임계값** (`사과 0.65 (5음소) · 빵 0.73 (3음소)`)
- 그래프: 시간에 따른 **점수**
- 그래프 아래: 실제 채점된 IPA (문맥 제한 반영, 매칭 윈도우 굵게)
- `프레임 상세`: 시각/연속/최고 후보/점수/임계값 원값
- 파라미터(창 길이·hop·연속 횟수)를 즉석에서 바꿔가며 비교 가능
- 앞에서 딴말을 해도 되지만, **정답 후 `연속횟수 × hop`초는 더 녹음**해야 확정됨

> 임계값을 차트에 선으로 겹치지 않고 위에 목록으로 두는 이유: 임계값은
> 상수가 아니라 **그 프레임의 선두 단어를 따라가서** 선이 오르내립니다.
> 또 스트리밍 갱신에서 다중 계열 색상 인코딩이 조용히 누락돼 임계값 선이
> 아예 렌더링되지 않는 문제도 있었습니다. 단일 계열 + 목록이면 둘 다
> 피할 수 있습니다.

**🔴 실시간** — 마이크 스트리밍 (VR과 가장 가까움)
- ⚡ **준비** 버튼으로 모델 예열 후 마이크 시작 (예열 없이 시작하면 첫 인식에서 1~2분 멈춤)
- 말하는 도중 0.5초마다 채점, 연속 N회 이기면 **녹음이 자동으로 중단됨**
- 그래프·IPA·프레임 표는 연속 청취 탭과 동일
- 처리 속도: 첫 프레임 ~520ms(1회성), 이후 프레임당 142~259ms (예산 500ms)
- hop은 0.5초 고정 — 바꿔가며 비교하려면 🔁 연속 청취 탭 사용

**📝 정답 데이터 만들기** — 단어 → IPA 추출
- 언어 선택 + 단어를 한 줄에 하나씩 입력 (`#` 주석 허용)
- 🚀 추출 & 다운로드 클릭 → `[{text, language, phonemes}, ...]` 형식 JSON 파일

### 💻 CLI 도구

CLI 도구들은 `shared/targets.json` (사전 빌드된 IPA 카탈로그)을 기반으로 동작합니다.

**1. 정답 단어 정의 후 빌드**
`shared/words.csv` 편집:
```csv
answer_id,text,language
mom,엄마,ko
apple,사과,ko
```

Unity에서는 `Tools → Phoneme Matching → 정답 데이터 다시 만들기` 버튼이
이 빌드를 대신 돌리고 `StreamingAssets`까지 복사합니다.

```powershell
python -m python.build.build_targets   # → shared/targets.json
```

**2. 실시간 마이크 녹음**
```powershell
python -m python.tools.record_live --target 사과
python -m python.tools.record_live --target 사과 --pick-device   # 마이크 선택
python -m python.tools.record_live --queue-from-words            # words.csv 전체 순회
```

**3. WAV 파일 / 일괄 평가**
```powershell
python -m python.tools.test_real_audio my.wav --target 사과
python -m python.tools.test_real_audio my.wav --probe            # ASR만
python -m python.tools.test_real_audio my.wav --scan-all         # 전체 검색
python -m python.tools.test_real_audio --manifest tests.csv      # 일괄
```

**3-1. 연속 청취 시뮬레이션** (VR 흐름 재현)
```powershell
python -m python.tools.test_streaming my.wav --target 사과
python -m python.tools.test_streaming my.wav --target 사과 --verbose  # 프레임별
python -m python.tools.test_streaming my.wav                      # 전체 단어 후보
```

**4. 오디오 진단** (잡음? 잘렸나? ASR 한계?)
```powershell
python -m python.tools.diagnose_audio recordings/session_XXX/001_사과.wav
```

**5. 자동 평가 (TTS 골든셋)**
```powershell
python -m python.tools.generate_golden_audio   # 36개 클립 자동 생성
python -m python.tools.evaluate                # 평가 + 캐시
python -m python.tools.evaluate --refresh-cache  # 모델 재추론
```

## 매칭 알고리즘

기본은 **부분 문자열(Substring) 매칭**:
- 사용자 IPA에서 정답이 등장하는 가장 가까운 윈도우를 찾음
- 윈도우 밖 음소는 1개당 `skip_cost`(0.15)만 부담 — 삽입 비용(0.6)보다 훨씬 싸다
- 마이크 백그라운드 잡음, 호흡, 키보드 소리 등이 음소로 잡혀도 정답 통과
- 윈도우가 정답 길이의 **50% 미만**이면 페널티 (잡음만 있는 케이스 거절)

> `skip_cost`가 0이면 윈도우 밖이 완전히 공짜가 되어, 점수가 발화 길이에 대해
> **단조 비감소**가 된다. 즉 말을 길게 할수록 후보 윈도우만 늘어나 어떤 임계값도
> 결국 넘게 되고, 무관한 발화가 통과한다(측정: 48음소 잡담 → "가요" 0.950).
> 임계값 조정으로는 막을 수 없어 `skip_cost`를 도입했다.
> 회귀 테스트: `TestSubstringFalseAccept`

### 연속 청취 (스트리밍)

VR 흐름은 마이크를 열어두고 정답이 인식되면 즉시 다음으로 넘어갑니다. 이건
녹음 하나를 채점하는 것과 **다른 문제**라 별도 설정을 씁니다
(`Matcher.for_streaming` + [StreamingMatcher](python/runtime/matching/streaming.py)).

```python
sm = StreamingMatcher(Matcher.for_streaming(matrix), [asked_word])
for t, window in rolling_windows(audio, window_s=2.5, hop_s=0.5):
    hit = sm.push(recognizer.recognize(window))
    if hit:
        break   # 녹음 중단, 다음 문제로
```

| | 배치 (웹 UI·CLI) | 스트리밍 (VR) |
|---|---|---|
| `skip_cost` | 0.15 | 0.05 |
| 윈도우 커버리지 | 0.5 | 0.8 |
| 문맥 제한 | 없음 | 최근 `4×정답음소수+3` |
| 연속 확인 | 없음 | 2회 |

**설정을 하나로 통일할 수 없습니다.** 배치 설정을 스트리밍에 쓰면 정답 검출이
1/4로 떨어지고, 스트리밍 설정을 배치에 쓰면 positives가 69.4%→47.2%로
무너집니다(커버리지 0.8이 아빠→"앞" 같은 의도된 관용을 죽임).

핵심 두 가지:

- **음소를 이어붙이지 말 것.** 청크를 독립 인식해 합치면 wav2vec2가 문맥을
  잃어 뭉개진다(4.4초 발화에서 0.5초 청크는 43음소 → 31음소, 단어 파손).
  매 프레임 **최근 창을 통째로 재인식**해야 한다.
- **연속 확인은 여전히 값어치가 있다.** 골든 클립 36개로 스트리밍 세션을
  만들어 스윕한 결과다(0.5초 hop, 각 세션에서 자기 단어와 나머지 17단어를
  모두 시도).

  | 연속 | 검출 | 오발동 | 확정까지 |
  |---|---|---|---|
  | 1회 | 18/36 | 19/612 | 1.5초 |
  | **2회** | **18/36** | **16/612** | **2.1초** |
  | 3회 | 15/36 | 10/612 | 2.5초 |

  2회는 1회보다 무조건 낫다(검출 같고 오발동만 줄어듦). 3회로 올리면
  오발동 1%p를 얻는 대신 검출 8%p를 잃는데, "제대로 말했는데 못 알아듣는"
  쪽이 더 나쁘므로 **기본값은 2회**다.

  단 이건 성인 TTS 기준이다. 아이 발화는 프레임별 점수가 더 흔들려서 긴
  연속이 다시 유리해질 수 있으니, 실제 녹음이 생기면 다시 재야 한다.

대가는 **지연 `연속횟수 × hop`**(기본 1.0초)이고, 정답 직후 오디오가 끊기면
확정되지 않으므로 발화 후 그만큼 더 들어야 합니다.

### 경쟁 단어는 없앴습니다

한때는 정답이 같은 세그먼트의 다른 단어들과 겨뤄서 이겨야 통과했습니다.
골든셋으로 재보니 **정확도가 완전히 동일**했고(positives 69.4%, negatives
거절율도 같음), 실제로 걸러내는 건 단어별 임계값과 `skip_cost`였습니다.
경쟁은 앱이 "화면에 어떤 단어들이 떠 있는지"를 알아야 하게 만들 뿐이라
제거했습니다. 지금은 **물어본 단어만 채점**합니다.
회귀 테스트: [test_streaming.py](python/tests/test_streaming.py)

`Confusion Matrix` ([shared/confusion_matrices/ko_child_v1.json](shared/confusion_matrices/ko_child_v1.json)):
- 격음/평음/경음 자유 교체 (페널티 ~0.2)
- /ㅅ/→/ㅌ/, /ㄹ/→/ㄷ/ 등 발달지연 미숙 패턴 (~0.3)
- 종성 불파음 ↔ 동가 초성 (~0.15~0.2)
- 모음 인접 (~0.3~0.5)
- 종성 탈락 (del cost ~0.3)
- 윈도우 밖 음소 스킵 (`skip_cost` 0.15)

임계값 (자동, 음소 수 기반):
- 1~2 음소: 0.85
- 3 음소: 0.73
- 4 음소: 0.70
- 5~6 음소: 0.65
- 7+ 음소: 0.60

## 테스트

```powershell
pytest python/tests -v                      # 파이썬 (83개)
dotnet test csharp/PhonemeMatching.Tests    # C# 포팅 (20개, Unity 불필요)
```

- 파이썬: 매핑 테이블, 한국어 G2P, 매처 + Confusion Matrix + Substring,
  연속 청취 (실제 ASR 프레임 픽스처)
- C#: 파이썬이 생성한 기준 벡터 340개와 대조. 매처나 matrix를 고쳤으면
  `python -m python.tools.export_parity_vectors`로 벡터를 다시 만들 것
- C# CTC 디코더: 파이썬 `batch_decode`와 43케이스 대조(36개는 실제 ONNX가
  골든 클립에 대해 내놓은 토큰 id). 모델이나 어휘를 바꿨으면
  `python -m python.tools.export_ctc_vocab`로 다시 만들 것

## Phase 진행

- [x] **Phase 0**: 환경 셋업 (venv + g2pkk + jamo)
- [x] **Phase 1**: 빌드 파이프라인 (G2P + targets.json)
- [x] **Phase 2**: 한국어 ASR + 매처 (Substring + Confusion Matrix + 임계값)
  - 골든셋 검증: Positives 69.4% / Negatives 거절율 95.8% (다른 단어 17개 대조)
  - 자세한 분석: [REPORT_PHASE2.md](REPORT_PHASE2.md)
- [x] **Phase 2.5**: 사용자 도구 (웹 UI + CLI + 진단 + 일괄)
- [x] **Phase 3**: Unity 패키지 — C# 매칭 엔진 + 파이썬 대조 테스트
  - [unity/Packages/com.domicube.phoneme-matching/](unity/Packages/com.domicube.phoneme-matching/)
  - 매처·스트리밍·자모/IPA 포팅 완료, 기준 벡터 340개 일치
  - 마이크 캡처 + 리스닝 루프 + stub recognizer까지 연결
- [x] **Phase 3.5**: ONNX export 검증 (`python -m python.tools.export_onnx`)
  - Sentis 미지원 연산자 0종, opset 18 (지원 범위 7~25)
  - PyTorch와 골든셋 36/36 동일한 한글·IPA 출력
- [x] **Phase 4**: Sentis `IPhonemeRecognizer` 구현
  - Sentis 2.6.1 임포트 성공 (동적 축 인식, 미지원 연산자 0종)
  - `CtcDecoder` + `SentisPhonemeRecognizer` — 골든 클립 4개를 Unity에서
    통과시켜 파이썬과 **한글·IPA 완전 일치**, 클립당 26~51ms
  - 데모 씬에서 실제 마이크 발화로 확정 배너까지 확인 (성인 명료 발화)
- [ ] Phase 5: 실제 아동 녹음으로 재튜닝 ← **다음**
- [ ] Phase 6: Fine-tune (필요 시, catastrophic ASR 회수)

### 배포 전제가 바뀌었습니다

**온디바이스 VR이 아니라 PC 테더링**입니다. 그래서:

- **양자화가 불필요**해졌습니다. Phase 3의 원래 목표("기기에 욱여넣기")가 사라졌고,
  오히려 정확도 우선으로 모델을 고를 수 있습니다
- **Sentis 백엔드는 GPUCompute 단일이고 CPU 폴백은 두지 않습니다.**
  아래 실측 참고. CPU 폴백이 필요한 수준의 PC는 어차피 PCVR을 못 돌립니다
- 파이썬은 **빌드 타임 도구로만** 남습니다. 앱에는 파이썬도 서버도 들어가지 않습니다

#### 백엔드 실측 (2.5초 창, 유휴 PC, RTX 5060 Ti / Ryzen 7 7800X3D)

| 실행 환경 | 프레임당 | 비고 |
|---|---|---|
| Sentis `GPUCompute` | **22~32ms** | 첫 추론 1931ms (셰이더 컴파일 + 1.2GB 업로드) |
| Sentis `CPU` | 461~523ms | hop 예산 500ms를 거의 소진. 코어 6개 점유 |
| PyTorch CPU (파이썬) | ~240ms | 참고용 |
| ONNX Runtime CPU (파이썬) | ~258ms | 같은 .onnx 파일 |

Sentis CPU가 느린 건 그래프 융합 부재도, 에디터 오버헤드(Burst 안전검사)도,
스레드 부족도 아니었습니다 — 셋 다 측정으로 배제했습니다. 코어를 6개 쓰면서
ORT보다 코어-시간을 2.7배 쓰는, **커널 효율 차이**입니다.

주의사항 두 가지:
- **VRAM 1.2GB**를 가중치가 차지합니다. 렌더링과 공유하므로 4GB 카드는 빠듯합니다
- **첫 추론 1.9초**는 모든 PC에서 발생합니다. 로딩 중에 `Warmup()`을 부르지
  않으면 아이의 첫 발화에서 멈춥니다
- 아직 미측정: VR 렌더링과 같은 GPU를 쓸 때의 경합. 25ms를 한 번에 던지면
  90fps(프레임 11ms)에서 끊길 수 있어, 필요하면 `ScheduleIterable`로
  레이어를 여러 프레임에 나눠야 합니다

### Unity 셋업 (모델은 저장소에 없습니다)

```powershell
python -m python.tools.export_onnx        # shared/models/wav2vec2_ko.onnx (1.18GB)
copy shared\models\wav2vec2_ko.onnx unity\Assets\Models\
python -m python.tools.export_ctc_vocab   # StreamingAssets 어휘 + 대조 벡터
```

임포트 후 씬의 `Pronunciation Demo` 오브젝트에서 `Model` 필드에
`Assets/Models/wav2vec2_ko.onnx`를 넣고 Play. 비워두면 stub으로 동작하며
배너에 어느 쪽이 도는지 표시됩니다.

### Unity 2022 와 Unity 6 을 모두 지원합니다

패키지는 두 엔진용 인식기를 모두 담고 있고, 설치된 쪽만 컴파일됩니다
(asmdef `versionDefines` + `defineConstraints`). 소비하는 프로젝트는
manifest 에 엔진 한 줄만 자기 에디터에 맞춰 넣습니다 — 2022 는
`com.unity.sentis: 1.2.0-exp.2`, Unity 6 은 `com.unity.ai.inference: 2.6.1`.

**모델은 시간 축을 40000 샘플(2.5초 창)로 고정해서 export 합니다.**
Sentis 1.2 는 동적 축 그래프를 실행하지 못합니다 — 임포트는 되지만 어떤
입력 길이를 줘도 같은 Reshape 에서 죽고, 임포터 최적화를 꺼도 같습니다.
축을 고정하면 그 계산이 상수로 접혀 사라집니다. Unity 6 은 동적도
실행하지만 파일 하나로 양쪽을 덮으려고 통일했고, 부수 효과로 프레임당
30ms → 20ms 로 빨라졌습니다.

Unity 2022.3 + Sentis 1.2 실측: 골든 클립 4개 × 124프레임 토큰 id가
파이썬과 전부 일치, GPU 30ms / CPU 390ms.

### 다음 작업 (Phase 5)

1. **실제 아동 녹음 확보** — 지금까지의 모든 튜닝값은 TTS 성인 발화 기준입니다
2. 확보한 녹음으로 confusion matrix·임계값 재튜닝
3. VR 씬에서 GPU 경합 측정, 필요하면 `ScheduleIterable`로 프레임 분할

**미해결 마찰**: ONNX가 1.18GB라 Sentis 임포트가 느리고 Git LFS가 필요합니다.
FP16(≈600MB)이나 더 작은 모델을 검토할 여지가 있습니다.

## 알려진 한계

| 한계 | 원인 | 영향 |
|---|---|---|
| 짧은 1~2음절 단어 ASR 정확도 낮음 | wav2vec2 학습 데이터(KsponSpeech)가 다어절 위주 | 빵, 책, 가요 등에서 자주 빈 출력 |
| 경음(ㄲ/ㄸ/ㅃ/ㅆ/ㅉ) 인식 약함 | 음향적으로 미세 | 토끼, 빵, 아빠 등 |
| 활음 위주 단어 인식 약함 | ㅛ, ㅑ 같은 활음 손실 | 우유, 와요 등 |
| TTS와 실제 아동 발화의 도메인 갭 | 학습 데이터에 아동 발화 없음 | 실제 아동 정확도는 본 평가보다 낮을 수 있음 |
| 첫 인식 호출 시 1~2분 대기 | wav2vec2 모델이 lazy-load (~1.2GB) | 한 번 워밍업되면 이후 호출은 빠름 |

해결책: Phase 5 fine-tuning (KsponSpeech IPA 라벨 또는 아동 발화 데이터)

## 알려진 이슈

- **eunjeon `pkg_resources` 경고**: `setuptools<70`에서 발생. 무해.
- **Gradio 6.0 API**: `theme`은 `launch()` 인자로만 전달 가능.
- **마이크 녹음 잘림**: `--trailing-ms 1000`으로 늘리거나 `record_live`의 기본값 조정.
