# Audio Recognition — 한국어 발음 판정

아이가 화면에 뜬 단어를 **말했는지 아닌지** 판정합니다. VR 발음 훈련 앱에
넣을 엔진이고, 발음이 부정확한 사용자(아동·발달지연 포함)를 대상으로 합니다.

- Unity 패키지: [`com.domicube.phoneme-matching`](unity/Packages/com.domicube.phoneme-matching/)
- 종합 보고서: [REPORT.md](REPORT.md)
- 저장소: <https://github.com/rnrdus5642/Audio-Recognition>

---

## 목차

1. [왜 STT 를 쓰지 않는가](#1-왜-stt-를-쓰지-않는가)
2. [파이프라인](#2-파이프라인)
3. [매칭 알고리즘](#3-매칭-알고리즘)
4. [배치와 스트리밍](#4-배치와-스트리밍)
5. [저장소 구조](#5-저장소-구조)
6. [파이썬 도구 사용법](#6-파이썬-도구-사용법)
7. [Unity 에서 쓰기](#7-unity-에서-쓰기)
8. [측정 결과](#8-측정-결과)
9. [평가용 음성 데이터](#9-평가용-음성-데이터)
10. [테스트](#10-테스트)
11. [알려진 한계](#11-알려진-한계)
12. [용어](#12-용어)

---

## 1. 왜 STT 를 쓰지 않는가

겉보기엔 음성 인식이지만, 일반 STT 를 그대로 쓰면 두 가지가 무너집니다.

**환각.** STT 는 무슨 소리든 그럴듯한 문장으로 만들어 냅니다. 아이가
어물거리면 "사과했어요" 같은 걸 내놓고, 문자열 비교는 이걸 오답으로
처리합니다. 반대로 전혀 다른 말을 했는데 정답 단어가 섞여 나오기도 합니다.

**부정확한 발음.** 대상 사용자는 아동이고 발달지연도 포함합니다. "사과"가
"타과"로, "토끼"가 "도끼"로 나오는 게 정상입니다. 글자가 틀렸다고 틀린 게
아닙니다.

그래서 이 프로젝트는 **글자가 아니라 음소(IPA)를 비교**하고, **완전 일치가
아니라 부분 일치에 가중치를 매겨** 판정합니다. 아이가 낼 법한 오류는 싸게,
낼 리 없는 오류는 비싸게 계산합니다.

음향 모델도 언어모델이 붙지 않은 CTC 모델(wav2vec2)을 씁니다. 언어모델은
잘못된 발음을 비슷한 단어로 "고쳐주는데", 그게 정확히 이 시스템이 판정해야 할
대상이기 때문입니다.

---

## 2. 파이프라인

**빌드 타임**은 PC 에서 한 번 돌려 데이터를 만들고, **런타임**은 앱 안에서
0.5 초마다 돕니다. 빌드된 앱에는 파이썬도 서버도 들어가지 않습니다.

### 빌드 타임 — 5 단계 (파이썬)

| # | 단계 | 하는 일 |
|---|---|---|
| 1 | `shared/words.csv` | 정답 단어를 글자로 적어둠 (`answer_id,text,language`) |
| 2 | g2pkk + eunjeon(mecab) | 음운 규칙을 적용해 소리 나는 대로 (`먹어요` → `머거요`) |
| 3 | 자모 → IPA + 임계값 | `shared/targets.json` 생성 |
| 4 | `export_onnx.py` | wav2vec2 를 시간축 40000 고정 ONNX 로 |
| 5 | `export_ctc_vocab.py` | 번호 ↔ 글자 대응표와 전처리 설정 추출 |

### 런타임 — 6 단계 (Unity C#)

| # | 단계 | 하는 일 |
|---|---|---|
| 1 | 오디오 2.5초 창 | 최근 2.5 초를 16 kHz 모노로 들고 있다가 40000 샘플을 통째로 넘김 |
| 2 | wav2vec2 추론 | 볼륨을 정규화해 모델에 넣고 124 프레임 × 1205 글자 점수를 받음 |
| 3 | CTC 디코딩 | 프레임마다 최고점을 골라 중복과 blank 를 걷어내고 한글로 되돌림 |
| 4 | 자모 → IPA | 인식된 한글을 정답과 **같은 표**로 음소 기호로 바꿈 |
| 5 | 부분문자열 매칭 | 정답 음소가 가장 잘 맞는 구간을 찾아 confusion matrix 가중치로 채점 |
| 6 | 연속 2회 확인 | 같은 단어가 두 프레임 연속 통과하면 확정하고 세션을 닫음 |

**핵심은 빌드 3 단계와 런타임 4 단계가 같은 매핑 테이블을 지난다는 점입니다.**
정답 측과 사용자 측이 같은 IPA 표기 체계로 떨어지지 않으면 거리 계산이 의미가
없습니다.

### 왜 G2P 를 빌드 타임으로 뺐나

한국어 음운 규칙은 형태소 분석기(mecab)가 필요하고, 사전만 112MB 에 네이티브
바이너리까지 딸려 옵니다. Unity 에 넣을 수 없습니다.

넣을 필요도 없습니다 — 단어→IPA 변환은 아이가 말할 때가 아니라 **단어 목록을
만들 때** 하는 일입니다. 앱에는 결과인 `targets.json` 만 들어갑니다.

규칙을 생략했을 때의 손해는 측정했습니다. 런타임의 자모→IPA 만으로 18단어를
변환해 빌드 결과와 비교하니 **3/18 만 달랐습니다**. 사용자 발화 쪽은 ASR 이
이미 소리 나는 대로 뱉으므로 규칙을 적용할 대상이 아니고, **정답 쪽만
정확하면 됩니다.**

다만 규칙이 음소를 통째로 바꾸는 경우는 손해가 큽니다.

```
같이   규칙 적용 [k,a,tɕʰ,i]  →  1.000
       규칙 없음 [k,a,t̚, i]  →  0.800   (임계 0.70)
```

### 왜 시간축을 고정했나

**Sentis 1.2(Unity 2022)는 동적 축 그래프를 실행하지 못합니다.** 임포트는
되지만 어떤 입력 길이를 줘도 같은 `Reshape` 에서 죽고, 임포터 최적화를 꺼도
같습니다. 축을 40000 샘플(2.5초)로 고정하면 그 계산이 상수로 접혀 사라집니다.

Unity 6 은 동적도 실행하지만 파일 하나로 양쪽을 덮으려고 통일했고, 부수
효과로 프레임당 30ms → 20ms 로 빨라졌습니다.

대신 런타임은 **항상 정확히 40000 샘플**을 줘야 합니다. `AudioWindowBuffer`
가 아직 2.5초가 안 찼어도 앞을 무음으로 채워 반환하는 이유입니다.

---

## 3. 매칭 알고리즘

출발점은 편집 거리입니다 — 두 음소 배열을 같게 만드는 최소 비용, 삽입·삭제·
치환 각 1 점, 전체를 전체와 비교. 이걸 그대로 쓰면 아이 발화에 맞지 않아
**거리 계산에서 세 군데를 바꿨습니다.**

아래 숫자는 정답 `사과` `[s,a,k,w,a]` 에 대한 실측입니다.

### (1) 치환 비용을 음소쌍마다 다르게 — Confusion Matrix

[`shared/confusion_matrices/ko_child_v1.json`](shared/confusion_matrices/ko_child_v1.json)

| 패턴 | 비용 | 예 |
|---|---|---|
| 평음/격음/경음 교체 | ~0.2 | `k` ↔ `kʰ` |
| 발달지연 미숙 (/ㅅ/→/ㄷ/, /ㄹ/→/ㄷ/) | ~0.3 | `s` ↔ `t` |
| 종성 불파음 ↔ 동가 초성 | ~0.15–0.2 | `k̚` ↔ `k` |
| 모음 인접 | ~0.3–0.5 | `a` ↔ `o` |
| 종성 탈락 (삭제) | ~0.3 | 책 → 채 |
| 그 밖의 치환 (기본) | 0.8 | `l` ↔ `t` |

```
사과 → 타과   [t,a,k,w,a]     거리 0.30   점수 0.940
사과 → 차과   [tɕʰ,a,k,w,a]   거리 0.75   점수 0.850
```

> **이 값들은 아직 검증되지 않았습니다.** 문헌의 아동 발달 패턴을 보고 손으로
> 넣었고, 조정 근거는 합성음 골든셋이었습니다. 실제 아동 녹음으로 재튜닝이
> 필요합니다.

### (2) 전체 대신 가장 잘 맞는 구간만 — 부분문자열 매칭

배열 전체를 비교하면 "음... 사과!" 의 앞부분이 삽입 비용으로 잡혀 점수가
무너집니다. **정답이 가장 잘 들어맞는 구간(창)** 을 찾고, 창 밖 음소는 1 개당
`skip_cost` 만 물립니다.

```
사과 → [ɯ,m,h, s,a,k,w,a]   거리 0.45   점수 0.910   창 [3:8]
```

> `skip_cost` 를 0 으로 두면 창 밖이 공짜가 되어 점수가 발화 길이에 대해
> **단조 비감소**가 됩니다. 말을 길게 할수록 후보 창만 늘어나 어떤 임계값도
> 결국 넘습니다 — 실측으로 48음소짜리 무관한 잡담이 "가요" 에 0.950 을
> 받았습니다. 회귀 테스트: `TestSubstringFalseAccept`

### (3) 창이 너무 짧으면 감점 — 커버리지 페널티

(2) 의 부작용입니다. `사` 만 말해도 창을 `[0:2]` 로 잡으면 그 안은 완전
일치라 점수가 높게 나옵니다.

```csharp
if (windowLen < Coverage * targetLen)
    baseScore *= windowLen / (double)targetLen / Coverage;
```

```
사과 → [s,a]   점수 0.544   창 [0:2]   ← 2/5 로 감점
```

### 판정은 별개 단계입니다

```
점수 = 1 − 거리 ÷ 정답 음소 수     (커버리지 감점 적용)
통과 = 점수 ≥ 그 단어의 임계값
```

**임계값은 단어마다 다릅니다.** 빌드 타임에 음소 수로 자동 결정됩니다
([`auto_threshold`](python/build/build_targets.py)).

| 정답 음소 수 | 1–2 | 3 | 4 | 5–6 | 7+ |
|---|---|---|---|---|---|
| 임계값 | 0.85 | 0.73 | 0.70 | 0.65 | **0.75** |

7 음소 이상이 0.60 → 0.75 로 올라간 것은 실측 결과입니다. 142단어를 실제
성인 발화에 걸어보니 긴 단어의 오확정이 14.6% 로 짧은 단어의 3배였습니다 —
음소가 많으면 긴 문장 안에 통과할 구간이 생기기 쉬운데 임계값은 오히려
낮았습니다. 0.75 로 올리자 1.2% 가 됐고 검출은 2%p 만 잃었습니다.

6 음소 이하는 손대지 않았습니다. 짧은 단어는 음소 하나가 틀리면 점수가 크게
떨어져 진짜 발화도 임계값 근처에 있고, 올릴 여유가 없습니다.

---

## 4. 배치와 스트리밍

VR 흐름은 마이크를 열어두고 정답이 들리면 즉시 넘어갑니다. 녹음 하나를
채점하는 것과 **다른 문제**라 설정을 분리했습니다.

| | 배치 (웹 UI·CLI) | 스트리밍 (VR) |
|---|---|---|
| `skip_cost` | 0.15 | 0.05 |
| 창 커버리지 하한 | 0.5 | 0.8 |
| 문맥 제한 | 없음 | 최근 `6×정답음소수+3` |
| 연속 확인 | 없음 | 2회 |

**하나로 통일할 수 없습니다.** 배치 설정을 스트리밍에 쓰면 검출이 1/4 로
떨어지고, 스트리밍 설정을 배치에 쓰면 positives 가 무너집니다 (커버리지 0.8
이 "아빠 → 앞" 같은 의도된 관용을 죽입니다).

### 청크를 이어붙이면 안 됩니다

wav2vec2 는 문맥 모델이라 짧은 조각을 따로 인식해 합치면 뭉개집니다 — 4.4초
발화를 0.5초 청크로 자르니 43음소가 31음소로 줄고 단어가 깨졌습니다. 매
프레임 **최근 2.5초 창을 통째로 재인식**해야 합니다. 비용이 아니라 정확도
문제입니다.

### 연속 확인이 필요한 이유

프레임마다 채점하므로 우연히 한 번 넘길 기회가 계속 생깁니다. 10초 세션에
0.5초 hop 이면 채점이 20번 돕니다. 같은 답을 연속 N회 요구하면 우연한 일치는
다음 프레임에 흩어지고, 진짜 발화는 창에 남아 계속 이깁니다.

대가는 지연 `연속횟수 × hop`(기본 1.0초)입니다. **정답 직후 마이크를 닫으면
확정되지 않습니다.**

연속 횟수를 더 올리면 지연·검출·상한을 함께 잃습니다. 단어가 2.5초 창에
온전히 들어있는 프레임은 5개 남짓이라, 연속 6회는 원리적으로 불가능합니다.

### 문맥 제한은 연속 횟수와 묶여 있습니다

스트리밍은 채점 직전에 사용자 음소 배열을 최근 `6×정답음소수+3` 개로
잘라냅니다. 계속 말하면 배열이 무한정 길어지고 어딘가에서 우연히 맞는 구간이
반드시 생기기 때문입니다.

그 창이 담는 발화 시간이 `연속횟수 × hop` 보다 짧으면, 정답이 확정되기 전에
창 밖으로 밀려나 **영원히 확정되지 않습니다.** 한국어는 초당 10음소 정도이고
`사과`(5음소)면 창이 33음소 ≈ 3.3초라 필요한 1.0초보다 넉넉합니다. 둘 중
하나를 바꾸면 다른 쪽도 확인해야 합니다.

**넓힐수록 통과가 어려워집니다.** 매칭 창 밖 음소는 각각 `skip_cost` 를
부담하므로, 창을 넓히면 우연히 정답을 닮은 파편이 주변 음소값까지 물게
됩니다. 4 → 6 은 세 데이터에서 모두 검출이 오르고 오확정이 줄었습니다.

| | 문맥 4배 | 문맥 6배 | 문맥 8배 |
|---|---|---|---|
| 성인 낭독 | 80% / 4.58% | **84% / 2.13%** | 70% / 1.89% |
| 아동 6~7세 | 48% / 4.39% | **50% / 4.13%** | 50% / 4.13% |
| 아동 8~9세 | 53% / 2.35% | **56% / 2.01%** | 54% / 2.01% |

8배부터는 진짜 발화도 주변 비용을 물기 시작해 검출이 무너집니다.

### 경쟁 단어는 없앴습니다

한때는 정답이 같은 세그먼트의 다른 단어들과 겨뤄 이겨야 통과했습니다.
골든셋으로 재보니 정확도가 완전히 동일했고, 실제로 걸러내는 건 단어별
임계값과 `skip_cost` 였습니다. 경쟁은 앱이 "화면에 어떤 단어들이 떠 있는지"를
알아야 하게 만들 뿐이라 제거했습니다. 지금은 **물어본 단어만 채점**합니다.

`Begin` 에 단어를 여러 개 넘길 수는 있습니다(동의어용). 다만 오확정 확률이
대략 곱해집니다.

```
엄마/아빠                  3/100
엄마/아빠/할아버지          5/100
엄마/아빠/할머니           21/100   ← 할머니 하나가 19
```

개수가 아니라 **어느 단어인지**가 문제입니다.

---

## 5. 저장소 구조

```
AudioProject/
├── python/
│   ├── build/                    # 빌드 타임 (G2P + targets.json)
│   │   ├── g2p/ko/               #   KoreanG2P, 자모↔IPA 매핑
│   │   └── build_targets.py      #   words.csv → targets.json, auto_threshold
│   ├── runtime/                  # 파이썬 기준 구현
│   │   ├── audio.py              #   16kHz mono 로딩
│   │   ├── matching/             #   matcher, confusion_matrix, streaming
│   │   └── recognizer/ko/asr.py  #   wav2vec2 + g2pkk
│   ├── tools/
│   │   ├── web_test.py           # 🌐 웹 UI (탭 4개)
│   │   ├── record_live.py        # 마이크 녹음
│   │   ├── test_real_audio.py    # 파일/일괄 테스트
│   │   ├── test_streaming.py     # 연속 청취 시뮬레이션
│   │   ├── diagnose_audio.py     # 오디오 진단
│   │   ├── generate_golden_audio.py
│   │   ├── evaluate.py           # 골든셋 회귀 검사
│   │   ├── export_onnx.py        # ONNX export + 검증
│   │   ├── export_ctc_vocab.py   # CTC 어휘 + 대조 벡터
│   │   ├── export_parity_vectors.py
│   │   └── sentis_parity/        # Unity 대조 하네스 (일회용)
│   └── tests/                    # 83개
├── shared/
│   ├── words.csv                 # 정답 단어 (UTF-8 BOM)
│   ├── targets.json              # 빌드 산출물
│   ├── confusion_matrices/ko_child_v1.json
│   └── models/                   # ONNX (gitignore)
├── unity/
│   ├── Assets/                   # 데모 프로젝트 (Unity 6)
│   └── Packages/com.domicube.phoneme-matching/   ← 팀에 배포되는 것
│       ├── Runtime/              # 엔진 참조 없는 코어
│       │   ├── Matcher.cs, StreamingMatcher.cs, CtcDecoder.cs
│       │   ├── PronunciationSession.cs, AudioWindowBuffer.cs
│       │   ├── PhonemeData.cs, ConfusionMatrix.cs
│       │   └── Unity/            #   마이크·리스너·Sentis (별도 asmdef)
│       ├── Editor/               # 초기 세팅·데이터·모델·테스트 메뉴
│       ├── Tests/Runtime/        # 파이썬 대조 벡터
│       └── Samples~/             # 한국어 데이터 + 예제
├── csharp/PhonemeMatching.Tests/ # Unity 없이 dotnet test (20개)
├── docs/ADDING_A_LANGUAGE.md
├── REPORT.md                     # 종합 보고서
└── REPORT_PHASE2.md              # 초기 골든셋 분석 (역사)
```

**언어 추가**는 `<lang>/` 폴더 신설 + REGISTRY 두 줄이면 됩니다. 매칭 엔진·
도구·Unity 코드는 건드리지 않습니다. → [docs/ADDING_A_LANGUAGE.md](docs/ADDING_A_LANGUAGE.md)

---

## 6. 파이썬 도구 사용법

### 셋업

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt          # 빌드 + 테스트 (g2pkk, eunjeon, jamo)
pip install -r requirements-phase2.txt   # 모델 + 웹 UI + 녹음
pip install gradio sounddevice
```

단어만 바꿀 거면 `requirements.txt` 만으로 충분합니다 — torch 도 wav2vec2 도
필요 없습니다.

> **주의**: numpy 는 `>=2.0,<2.5` 만 가능하고, `pip install` 이 librosa 를
> 깰 수 있습니다. `.venv` 는 다른 PC 로 복사해도 동작하지 않습니다.

### 웹 UI

```powershell
python -m python.tools.web_test          # http://127.0.0.1:7860
```

| 탭 | 용도 |
|---|---|
| 🎤 발음 테스트 | 단어 입력 후 마이크/파일 채점, 음소 정렬 시각화 |
| 🔁 연속 청취 | VR 흐름 재현. 창 길이·hop·연속 횟수를 바꿔가며 점수 그래프 |
| 🔴 실시간 | 마이크 스트리밍 (⚡ 준비 버튼으로 예열 먼저) |
| 📝 정답 데이터 만들기 | 단어 → IPA 추출 |

첫 인식 시 모델(~1.2GB)이 로드되어 1~2분 걸립니다.

### 단어 추가

```csv
# shared/words.csv  (UTF-8 BOM)
answer_id,text,language
mom,엄마,ko
apple,사과,ko
```

```powershell
python -m python.build.build_targets     # → shared/targets.json
```

Unity 에서는 `Tools → Phoneme Matching → 정답 데이터 다시 만들기` 가 이 빌드를
대신 돌리고 `StreamingAssets` 까지 복사합니다.

**모델이 낼 수 없는 음절이 있습니다.** wav2vec2 의 어휘는 음절 1205개고, 거기
없는 음절은 아무리 정확히 발음해도 전사되지 않습니다. 단어를 넣기 전에
`unity/Assets/StreamingAssets/wav2vec2_ko_vocab.json` 의 `tokens` 에 그 음절이
있는지 보세요.

지금 커리큘럼에서는 `놔`(`놔`·`놔요`) 하나가 여기 걸립니다. AI Hub 아동
전사 276만 발화 전체를 봐도 `놔` 는 64건뿐이라 학습으로 메우기도 어렵습니다.
다른 동사로 바꾸는 것이 맞습니다.

### 그 밖의 CLI

```powershell
python -m python.tools.record_live --target 사과
python -m python.tools.record_live --queue-from-words     # words.csv 전체 순회

python -m python.tools.test_real_audio my.wav --target 사과
python -m python.tools.test_streaming my.wav --target 사과 --verbose
python -m python.tools.diagnose_audio recordings/.../001_사과.wav

python -m python.tools.generate_golden_audio              # 합성음 36클립
python -m python.tools.evaluate                           # 회귀 검사
```

### 모델 준비

```powershell
python -m python.tools.export_onnx        # shared/models/wav2vec2_ko.onnx (1.18GB)
python -m python.tools.export_ctc_vocab   # 어휘 + 대조 벡터
```

`--static-samples 40000` 이 기본입니다. 창 길이를 바꾸려면 `48000`(3.0초)
처럼 함께 바꿔야 합니다.

---

## 7. Unity 에서 쓰기

**Unity 2022.3 과 Unity 6 을 모두 지원합니다.** 자세한 것은
[패키지 README](unity/Packages/com.domicube.phoneme-matching/README.md).

### 설치

```
Window > Package Manager > + > Add package from git URL…
https://github.com/rnrdus5642/Audio-Recognition.git?path=/unity/Packages/com.domicube.phoneme-matching

Tools > Phoneme Matching > 초기 세팅
```

`초기 세팅` 이 없는 것만 차례대로 설치합니다 — 추론 엔진, 데이터 JSON 3개,
음향 모델. 이미 있는 것은 건너뛰므로 두 경우에 같은 버튼을 씁니다.

| 상황 | 실제로 하는 일 |
|---|---|
| 처음 붙이는 프로젝트 | 세 단계 전부 |
| 이미 쓰는 프로젝트를 clone/pull | **모델만** (나머지는 커밋돼 있음) |

모델은 1.18GB 라 커밋되지 않고
[Release `model-v1`](https://github.com/rnrdus5642/Audio-Recognition/releases/tag/model-v1)
에 있습니다. 각자 한 번 받으면 됩니다.

### 코드

```csharp
var matrix  = PhonemeData.LoadMatrix(Read("ko_child_v1.json"));
var catalog = PhonemeData.LoadTargets(Read("targets.json"));
var vocab   = PhonemeData.LoadCtcVocabulary(Read("wav2vec2_ko_vocab.json"));

var model      = Resources.Load<ModelAsset>("Models/wav2vec2_ko");
var recognizer = new SentisPhonemeRecognizer(model, vocab);
recognizer.Warmup(2.5f);            // 로딩 화면에서. 첫 추론이 ~2초

var buffer  = new AudioWindowBuffer(2.5f);
var session = new PronunciationSession(matrix, catalog, recognizer, buffer: buffer);

session.Begin("사과");               // 문제 시작
var frame = session.Push();         // 0.5초마다
if (frame.Confirmed) 다음문제로();
session.End();
```

오디오는 앱 방식에 따라 한 줄만 다릅니다.

```csharp
buffer.Append(raw, clip.frequency);                                  // Microphone
buffer.AppendInterleaved(data, ch, AudioSettings.outputSampleRate);  // OnAudioFilterRead
```

앱에 오디오 캡처가 없으면 `PronunciationListener` 컴포넌트가 마이크까지
맡습니다.

**알아둘 것 셋**

- `Begin` 전에는 아무것도 안 돌고 GPU 도 쓰지 않습니다
- **오답 신호는 없습니다.** 확정되거나 아직 아니거나 둘뿐이라, "틀렸다"를
  말해주려면 앱이 시간을 재야 합니다
- `frame.Best` 는 확정 전에도 채워집니다. **`frame.Confirmed` 를 먼저**

**모델을 인스펙터에 끌어다 놓지 마세요.** 모델은 gitignore 되어 각자
임포트하므로 애셋 GUID 가 사람마다 다릅니다. 한 사람이 씬에 끌어다 놓고
커밋하면 나머지에게는 그 칸이 비어 보입니다. `Resources.Load` 는 GUID 를 쓰지
않아 이 문제가 없습니다.

### 추론 속도 (2.5초 창, RTX 5060 Ti / Ryzen 7 7800X3D)

| 실행 환경 | 프레임당 |
|---|---|
| Sentis `GPUCompute` (Unity 6) | **20~32 ms** |
| Sentis `GPUCompute` (Unity 2022 + Sentis 1.2) | **30 ms** |
| Sentis `CPU` | 461~523 ms |
| PyTorch CPU (참고) | ~240 ms |
| ONNX Runtime CPU (참고) | ~258 ms |

hop 예산이 500ms 이므로 GPU 는 여유롭고 CPU 는 거의 소진입니다. **CPU 폴백은
두지 않습니다** — 그게 필요한 PC 는 어차피 PCVR 을 못 돌립니다.

Sentis CPU 가 느린 이유는 그래프 융합 부재도, 에디터 오버헤드도, 스레드
부족도 아니었습니다(셋 다 측정으로 배제). 코어 6개를 쓰면서 ORT 보다 코어-
시간을 2.7배 쓰는 커널 효율 차이입니다.

주의: **VRAM 1.2GB** 를 가중치가 차지하고, **첫 추론 1.9초**가 모든 PC 에서
발생합니다. VR 렌더링과 GPU 를 공유할 때의 경합은 아직 미측정이며, 필요하면
`ScheduleIterable` 로 레이어를 여러 프레임에 나눠야 합니다.

### 배포 전제

**온디바이스 VR 이 아니라 PC 테더링**입니다. 그래서 양자화가 불필요해졌고,
정확도 우선으로 모델을 고를 수 있습니다. 파이썬은 빌드 타임 도구로만
남습니다.

---

## 8. 측정 결과

> **아동 발화 기준 정확도는 아직 확정되지 않았습니다.** 아래는 측정 조건과
> 함께 읽어야 합니다. 조건을 뗀 숫자 하나는 의미가 없습니다.

### 실제 아동 발화 (AI Hub, 5~7세, 가정 녹음, 자유 발화)

| 항목 | 케이스 | 결과 |
|---|---|---|
| 검출 | 1,393 | **625/1,393 = 44.9%** |
| 오확정 | 10,702 | **223/10,702 = 2.08%** |

음소 수별 검출:

| 음소 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| 검출 | 21% | 40% | 75% | 62% | 64% | 36% |

**짧은 단어가 무너집니다.** `우유` 0/13, `책` 3/57, `나비` 0/3.

그리고 검출 점수(0.66~0.99)와 오확정 점수(0.65~0.97)가 **겹칩니다.** 성인
발화에서는 갈렸는데(검출 0.82 / 오확정 0.62~0.69) 아동에서는 안 갈립니다 —
임계값 하나로 두 분포를 나눌 수 없다는 뜻입니다.

화자 15명, 사실상 6~7세(5세는 자유 발화에 없음).

### 실제 성인 발화 (Zeroth-Korean test split)

| 항목 | 데이터 | 결과 |
|---|---|---|
| ASR 문자 오류율 | 457발화 | 1.7% (빈 출력 0건) |
| 검출 | 단독 발화 10건 | 6/10 |
| 오확정 (스트리밍) | 100발화 × 18단어 | 39/1800 = 2.2% |
| 오확정 (배치) | 457발화 × 18단어 | **0/8226** |

142단어로 확장한 측정:

| 음소 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|
| 검출 | 36% | 78% | 89% | 89% | 95% | 100% | 100% |
| 오확정 | 5.7% | 4.3% | 10.8% | 5.6% | 14.6% | 13.3% | 13.1% |

이 표가 7음소 이상 임계값을 올린 근거입니다.

> 이 음향 모델은 Zeroth-Korean 으로 학습됐습니다(모델 카드 CER 1.78%, 우리
> 측정 1.7% 로 일치). test split 은 held-out 이지만 도메인이 정확히 맞아
> 인식률이 최상으로 나옵니다.

### 합성음 골든셋은 정확도 근거로 쓰지 않습니다

Edge-TTS 2목소리 36클립으로 `positives 69.4% / negatives 95.8%` 를 재던
방식은 폐기했습니다. **목소리를 바꾸면 결과가 뒤집힙니다** — 같은 `빵` 이
SunHi 에서는 빈 출력, InJoon·Hyunsu·SAPI 에서는 0.93 통과였습니다.

골든셋은 **회귀 검사**로만 유지합니다. 매처를 고쳤을 때 값이 움직이는지 보는
용도라면 합성음이 오히려 낫습니다 — 재현되고, 공짜고, 동의 문제가 없습니다.
ONNX↔PyTorch 대조와 Sentis 검증도 이 클립으로 했습니다.

---

## 9. 평가용 음성 데이터

### 한국어 아동 — AI Hub (승인 자동, 휴대폰 인증 필요)

| 데이터셋 | 내용 |
|---|---|
| [한국어 아동 음성 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=540) | 5~12세. Validation 만 27만 발화 / 500시간 |
| [어린이 음성 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=266) | 적응·튜닝용 |
| [어린이 음성 맥락 인식률 향상 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71502) | EBS·KBS 교육방송 |

**형식이 우리와 그대로 맞습니다** — 16kHz mono 16bit, 발화 구간 타임스탬프,
나이·성별·학년·소음환경·녹음장치 라벨.

Validation 화자 분포:

| 나이 | free 화자 / 발화 | formatted 화자 / 발화 |
|---|---|---|
| 5 | 0 / 0 | 30 / 8,103 |
| 6 | 3 / 1,129 | 46 / 18,614 |
| 7 | 12 / 6,764 | 59 / 17,181 |
| 8 | 22 / 13,324 | 60 / 17,366 |
| 9 | 25 / 14,428 | 84 / 38,237 |
| 10~12 | 42 / 22,860 | 323 / 113,011 |

`free` 는 자유 발화(가정·모바일), `formatted` 는 동화 낭독입니다. **5세는
자유 발화에 없습니다.**

**전부 받을 필요 없습니다.** 오디오 56GB 를 다 추론하면 열흘이 걸리는데,
필요한 건 몇 시간 분량입니다. 라벨(537MB)만 먼저 받아 필요한 발화 목록을
뽑고, 그 wav 만 꺼내면 됩니다.

> 두 코퍼스 모두 **정상 발달 아동**입니다. 이 시스템의 대상인 발달지연 아동
> 코퍼스는 개인정보·의료정보라 공개된 것이 사실상 없습니다.

### 한국어 성인 — 승인 없이 즉시

[Zeroth-Korean](https://openslr.org/40/) (CC BY 4.0, 51.6시간, 화자 105명).
[HuggingFace](https://huggingface.co/datasets/Bingsu/zeroth-korean) 에서 test
split(457발화) 만 받으면 60MB 입니다.

---

## 10. 테스트

```powershell
pytest python/tests -v                      # 파이썬 83개
dotnet test csharp/PhonemeMatching.Tests    # C# 20개 (Unity 불필요)
```

Unity PlayMode 테스트 20개는 같은 소스를 컴파일합니다.

**파이썬에서 C# 으로 옮긴 것은 전부 벡터로 대조합니다.** "옮겼다"가 아니라
"같은 값이 나온다"를 확인하는 게 목적입니다.

| 대조 | 케이스 |
|---|---|
| 매처·Confusion Matrix·자모/IPA | 340 |
| CTC 디코더 ↔ 파이썬 `batch_decode` | 43 (36개는 실제 ONNX argmax) |
| ONNX ↔ PyTorch 한글·IPA | 36 |
| Unity 2022 Sentis 1.2 ↔ 파이썬 토큰 id | 4클립 × 124프레임 |

매처나 matrix 를 고쳤으면 `export_parity_vectors`, 모델이나 어휘를 바꿨으면
`export_ctc_vocab` 로 벡터를 다시 만드세요.

Unity 대조는 [python/tools/sentis_parity/](python/tools/sentis_parity/) 에
하네스가 있습니다. **임포트가 됐다는 것과 계산이 맞다는 것은 다른
문제입니다** — Sentis 1.2 가 그 사이에서 죽는 것을 이 검사로 찾았습니다.

---

## 11. 알려진 한계

| 한계 | 원인 | 영향 |
|---|---|---|
| **짧은 단어 검출 실패** | 음소 하나가 틀리면 점수가 크게 떨어짐 | 아동 3음소 21%, 4음소 40%. `빵`·`책`·`우유` 는 사실상 동작 안 함 |
| **`놔`·`놔요` 인식 불가** | 음절 `놔` 가 모델 어휘 1205개에 없음 | 정확히 발음해도 전사될 수 없음. 다른 단어로 바꿔야 함 |
| **아동에서 점수 분포가 겹침** | 발음 편차가 큼 | 임계값만으로 검출·오확정을 나눌 수 없음 |
| Confusion Matrix 미검증 | 문헌 기반 + 합성음 튜닝 | 실제 아동 녹음으로 재튜닝 필요 |
| 경음(ㄲ/ㄸ/ㅃ/ㅆ/ㅉ) 약함 | 음향적으로 미세 | 토끼·빵·아빠 |
| 활음(ㅛ/ㅑ) 손실 | | 우유·와요 |
| 발달지연 아동 데이터 없음 | 공개 코퍼스 부재 | 실제 대상에 대한 근거가 없음 |
| ONNX 1.18GB | FP32 large | 임포트 느림, Release 배포 필요 |
| 첫 추론 1.9초 | 셰이더 컴파일 + 가중치 업로드 | 로딩 중 `Warmup()` 필수 |
| GPU 경합 미측정 | VR 렌더링과 같은 GPU | 90fps(11ms)에서 20ms 를 한 번에 던지면 끊길 수 있음 |

### 알려진 이슈

- **eunjeon `pkg_resources` 경고** — `setuptools<70` 에서 발생. 무해
- **Gradio 6.0** — `theme` 은 `launch()` 인자로만 전달 가능
- **마이크 녹음 잘림** — `--trailing-ms 1000` 으로 늘리거나 `record_live` 기본값 조정
- **Unity 실행 중 `packages-lock.json` 편집 금지** — Unity 가 manifest 를 다시
  써서 항목이 사라질 수 있습니다

---

## 12. 용어

| 용어 | 뜻 |
|---|---|
| **IPA** | 국제음성기호. `사과` = `[s,a,k,w,a]` |
| **G2P** | Grapheme-to-Phoneme. 글자를 소리로 (`학교` → `학꾜`) |
| **CTC** | 정렬 없이 학습하는 방식. 출력이 늘어져 나오고 blank 토큰을 씀 |
| **blank** | "지금은 아무 글자도 아님" 토큰. 이 모델에서 1204번 |
| **greedy 디코딩** | 프레임마다 최고점 하나만 집기. 빔 서치·언어모델 없음 |
| **창(window)** | 한 번에 인식하는 오디오 길이. 2.5초 = 40000 샘플 |
| **hop** | 채점 주기. 0.5초 |
| **프레임** | 창 하나를 채점한 결과. 모델 출력에서는 20ms 단위 124개 |
| **연속 확인(streak)** | 같은 단어가 몇 프레임 연속 통과해야 확정할지. 기본 2 |
| **`skip_cost`** | 매칭 창 밖 음소 1개당 비용 |
| **커버리지** | 매칭 창이 정답 길이의 몇 배 이상이어야 감점을 피하는지 |
| **문맥 제한** | 채점 직전에 사용자 음소 배열을 최근 몇 개로 자를지 |
| **검출(detection)** | 말한 단어를 확정하는 비율 |
| **오확정(false accept)** | 말하지 않았는데 확정되는 비율 |
| **배치 / 스트리밍** | 녹음 하나를 통째로 채점 / 마이크를 열어두고 프레임마다 채점 |
| **positives / negatives** | 자기 단어와 맞춰본 케이스 / 다른 단어와 맞춰본 케이스 |

---

## 다음 작업

1. **아동 녹음으로 파라미터 재튜닝** — 진행 중. 화자 단위 3분할(튜닝 free
   6~7세 / 검증 free 8~9세 / 평가 formatted 5세), 오확정 예산 10초 세션당 1%
   아래에서 검출 최대화
2. Confusion Matrix 를 실제 아동 발화로 재튜닝
3. 커리큘럼 단어 선정 — 짧은 단어를 피하면 지금 상태로도 크게 개선됨
4. VR 씬에서 GPU 경합 측정
5. (검토) FP16 ≈ 600MB 또는 더 작은 모델
