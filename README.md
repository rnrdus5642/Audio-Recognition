# Audio Recognition — 한국어 발음 판정

악기놀이 발표용 12단어 설정은 `./start_music_demo.ps1`로 실행합니다.
9단계 선택, 마이크 체험, 녹음 파일 테스트를 지원합니다.
[최신 적용 현황·9단계 정답표·오확정률·남은 한계](docs/MUSIC_DEMO_STATUS_2026-09-10.md)를 확인하세요.

Unity 패키지 0.8.5에는 변경된 12단어·9단계와 실제 사용자 음성에 따른 `잡아` 보조 판정이 내장되어 있습니다. 제공 녹음은 잡아 4/5 통과, 작아 2/2 거절이며, 다른 단어 설정은 0.8.4와 같습니다. [실제 음성 검증과 남은 한계](docs/GRAB_REAL_VOICE_2026-09-10.md)를 참고하세요.
[9월 10일 수정·검증 결과](docs/MUSIC_WORD_FIX_2026-09-10.md)를 확인하세요.
`Tools > Phoneme Matching > 악기놀이 발표 설정`으로 준비하고,
[새 프로젝트·기존 게임 적용 안내](unity/Packages/com.domicube.phoneme-matching/Documentation~/music-presentation.md)를 따라 연결하세요.

현재 개발 소스(2026-09-10)는 아동 모델·단어별 임계값·Unity Session API에
악기놀이 발표 프로필, 실시간 웹 테스트, 모델 오류를 고려한 발음 안내를 포함합니다.
웹과 Unity의 연속 확정 기본값은 **2회**입니다.
초기 음성 버퍼도 기본 2.5초·앞쪽 무음 패딩으로 맞췄으며, 여러 정답은 합격 후보를
우선 선택합니다. Unity 초기 세팅의 기존 JSON 보존과 테스트 창의 모델별 데이터 연결도 반영했습니다.
변경 사항은 [CHANGELOG](CHANGELOG.md), 확정 결과 필드는
[발음 설명 API](unity/Packages/com.domicube.phoneme-matching/Documentation~/pronunciation-feedback.md)를 참고하세요.
설치·Windows 실행·웹 대조 결과와 남은 제약은
[2026-09-06 검증 기록](unity/Packages/com.domicube.phoneme-matching/Documentation~/validation-2026-09-06.md)에 정리했습니다.

아이가 화면에 뜬 단어를 **말했는지 아닌지** 판정합니다. VR 발음 훈련 앱에
넣을 엔진이고, 발음이 부정확한 사용자(아동·발달지연 포함)를 대상으로 합니다.

- Unity 패키지: [`com.domicube.phoneme-matching`](unity/Packages/com.domicube.phoneme-matching/)
- 초기 설계·실험 기록: [REPORT.md](REPORT.md) (2026-08-11 시점)
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
| 2 | Python 한국어 음운 규칙 | 소리 나는 대로 변환 (`먹어요` → `머거요`), Unity에도 같은 규칙 구현 |
| 3 | 자모 → IPA + 임계값 | `shared/targets.json` 생성 |
| 4 | `export_onnx.py` | wav2vec2 를 시간축 40000 고정 ONNX 로 |
| 5 | `export_ctc_vocab.py` | 번호 ↔ 글자 대응표와 전처리 설정 추출 |

### 런타임 — 6 단계 (Unity C#)

| # | 단계 | 하는 일 |
|---|---|---|
| 1 | 오디오 2.5초 창 | 최근 2.5 초를 16 kHz 모노로 들고 있다가 40000 샘플을 통째로 넘김 |
| 2 | wav2vec2 추론 | 볼륨을 정규화해 모델에 넣고 124 프레임 × 1205 글자 점수를 받음 |
| 3 | CTC 디코딩 | 프레임마다 최고점을 골라 중복과 blank 를 걷어내고 한글로 되돌림 |
| 4 | 음운 규칙 → 자모 → IPA | 인식 한글에 `Korean.Rules.Apply` 후 정답과 같은 IPA 매핑 적용 |
| 5 | 부분문자열 매칭 | 정답 음소가 가장 잘 맞는 구간을 찾아 confusion matrix 가중치로 채점 |
| 6 | 연속 2회 확인 | 같은 단어가 두 프레임 연속 통과하면 확정하고 세션을 닫음 |

**핵심은 빌드 3 단계와 런타임 4 단계가 같은 매핑 테이블을 지난다는 점입니다.**
정답 측과 사용자 측이 같은 IPA 표기 체계로 떨어지지 않으면 거리 계산이 의미가
없습니다.

### 한국어 음운 규칙

현재는 정답 빌드뿐 아니라 인식된 한글에도 음운 규칙을 적용합니다.
Python의 `python/build/g2p/ko/rules.py`와 Unity의 `Korean.Rules`가 같은 규칙을
구현하며, 그 결과를 자모 → IPA로 바꿉니다. 아동 모델도 철자형 전사로 학습했으므로
이 경로를 유지해야 합니다.

런타임에는 mecab이 필요 없습니다. g2pkk는 비교용
`KoreanG2P.apply_rules_g2pkk()`에서만 쓰며 Windows 어댑터는 mecab-ko를 사용합니다.

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
사과 → 다과   [t,a,k,w,a]     거리 0.30   점수 0.940
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

| | 배치 (CLI) | 스트리밍 (웹 UI·VR) |
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
걸러질 수 있습니다. 다만 실제 발화가 창에 남아 있어도 다음 프레임에서 같은
전사·점수가 나온다는 보장은 없습니다. 짧은 발화가 한 번만 통과해 미확정되는
문제는 현재도 남아 있습니다([재현 기록](docs/SHORT_WORD_CONFIRMATION_2026-09-06.md)).

첫 통과 뒤 추가 확인 간격은 `(연속횟수 - 1) × hop`입니다. 기본 2회·0.5초

**보정된 예외 하나.** 정답 데이터가 그 단어에 `allow_exact_single_frame`
을 켜 뒀고 연속 확인이 2회이며 지금 설정이 그 단어의 보정 스냅샷(매트릭스
이름·스트리밍 프로파일·임계값·음소열)과 같을 때는, 점수가 **정확히 1.0**
인 프레임 하나로 확정합니다. 점수 1.0 은 설정된 정답 음소열과 그대로
일치했다는 뜻입니다 — 정확히는 **그 프레임에서 채점에 쓰인 구간**(문맥
제한으로 잘라낸 최근 음소열)이 그렇다는 뜻이지, 전사 전체가 정답뿐이었다는
뜻도 발음이 옳았다는 인증도 아닙니다. 짧은 단어가 그런
프레임을 한 번만 얻는 경우를 위한 것입니다 — 발화는 다음 hop 에도 2.5초 창
안에 있지만 창 안 위치가 바뀌면 인식 결과가 달라집니다. 어떤 단어가 이
예외를 받을지는 오확정 예산으로 정합니다 — 예산을 늘리지 않고 일반
임계값을 조여서 값을 치릅니다. 29단어 중 28단어가 켜져 있고 와요는
제외입니다. 근거·트레이드오프·한계는
[docs/DEMO_RELIABILITY_2026-09-06.md](docs/DEMO_RELIABILITY_2026-09-06.md).
주기에서는 한 번 더 채점합니다. 실제 지연에는 음성 창과 추론·큐 시간도 포함됩니다.

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

`Begin` 에 단어를 여러 개 넘길 수는 있습니다(동의어용). 각 후보가 자기 임계값을
통과하는지 먼저 검사하고, 통과한 후보 중 점수가 가장 높은 단어를 선택합니다.
전부 불합격이면 최고 점수를 표시용으로 반환합니다. 선택된 **같은 통과 단어**가
기본 2프레임 연속 유지돼야 확정되며, 웹 Python과 Unity C#에 같은 규칙을 적용합니다.
합격 기준·점수 계산·호출 방식·피드백 JSON 형식은 그대로입니다.

다만 오확정 확률이 대략 곱해집니다. 아래는 **이전 후보 선택 규칙에서의 측정 예시**로,
현재 규칙의 정확도 수치로 사용하면 안 됩니다.

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
│   │   └── recognizer/ko/asr.py  #   wav2vec2 + 한국어 음운 규칙
│   ├── tools/
│   │   ├── web_test.py           # 🌐 실시간 마이크 전용 UI
│   │   ├── _web_feedback.py      # 한글 설명·원본 IPA 비교
│   │   ├── _web_syllable_feedback.py # 기존 모델 오류표 기반 판단 보류
│   │   ├── record_live.py        # 마이크 녹음
│   │   ├── test_real_audio.py    # 파일/일괄 테스트
│   │   ├── test_streaming.py     # 연속 청취 시뮬레이션
│   │   ├── diagnose_audio.py     # 오디오 진단
│   │   ├── generate_golden_audio.py
│   │   ├── evaluate.py           # 골든셋 회귀 검사
│   │   ├── export_onnx.py        # ONNX export + 검증
│   │   ├── export_ctc_vocab.py   # CTC 어휘 + 대조 벡터
│   │   ├── export_parity_vectors.py
│   │   ├── sentis_parity/        # Unity 대조 하네스 (일회용)
│   │   ├── aihub/                # AI Hub 아동 코퍼스 정리
│   │   ├── child_finetune/       # 아동 모델 fine-tune
│   │   │   ├── manifest.py       #   학습셋 선정 (676h → 200h)
│   │   │   └── train.py          #   CTC 학습 · --resume
│   │   └── child_tuning/         # 아동 매칭 파라미터
│   │       ├── frames.py         #   프레임 캐시 (--isolated)
│   │       ├── build_reference.py#   기준 데이터 두 파일
│   │       ├── derive_thresholds.py  # 단어별 임계값 + 상한 경고
│   │       ├── fit_streaming.py  #   프로필 탐색
│   │       └── compare.py        #   모델·설정 비교
│   └── tests/                    # 201개 (2026-09-06)
├── shared/
│   ├── words.csv                 # 정답 단어 (UTF-8 BOM)
│   ├── targets.json              # 빌드 산출물 (성인)
│   ├── targets_child.json        # 빌드 산출물 (아동)
│   ├── thresholds_child.json     # 단어별 임계값 - 빌드 입력
│   ├── reference/                # 임계값을 재는 기준 (325KB)
│   │   ├── reference_frames.json.gz  # 아이 4,998발화의 음소열
│   │   └── reference_errors.json     # 이 모델이 뭘 뭘로 틀리는지
│   ├── confusion_matrices/
│   │   ├── ko_child_v1.json      # 성인 모델용
│   │   └── ko_child_v2.json      # 아동 모델용 (프로필만 다름)
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
├── csharp/PhonemeMatching.Tests/ # Unity 없이 dotnet test
├── docs/ADDING_A_LANGUAGE.md
├── REPORT.md                     # 초기 설계·실험 기록 (2026-08-11)
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

pip install -r requirements.txt          # 빌드 + 테스트 (g2pkk 비교용 mecab-ko 포함)
pip install -r requirements-phase2.txt   # 모델 + 웹 UI + 녹음
pip install gradio sounddevice
```

단어만 바꿀 거면 `requirements.txt` 만으로 충분합니다 — torch 도 wav2vec2 도
필요 없습니다.

> **주의**: numpy 는 `>=2.0,<2.5` 만 가능하고, `pip install` 이 librosa 를
> 깰 수 있습니다. `.venv` 는 다른 PC 로 복사해도 동작하지 않습니다.

### 웹 UI

실시간 화면 하나만 제공합니다. 정답 입력 → 준비 → 마이크 청취 → 동일 후보
연속 2회 통과 → 확정/자동 중단 흐름입니다. 창 기본 2.5초, 입력 전달 주기 0.5초입니다.
Unity와 같이 처음부터 2.5초(40,000샘플) 입력을 유지하며, 아직 녹음되지 않은
앞부분은 무음으로 채웁니다. 2.5초를 기다리지 않고 첫 입력부터 판정하며,
추가한 무음은 경과 시간이나 연속 판정 횟수에 포함하지 않습니다.

**채점 한 번은 새 샘플 8,000개(0.5초)이지, 콜백 한 번이 아닙니다.**
브라우저가 청크를 어떻게 쪼개 보내든 같은 PCM 이면 같은 창이 같은 순서로
생기고 같은 샘플에서 확정됩니다 — 예전에는 1초를 8,000샘플씩 두 번 보내면
2회, 4,000샘플씩 네 번 보내면 4회 채점돼 확정 시점이 달라졌습니다.
한 청크에 hop 이 여러 개 들어 있으면 각각의 경계 시점 창을 순서대로
채점하고 확정된 프레임에서 멈춥니다. hop 이 덜 찼으면 아무것도 하지
않습니다(모델 호출 없음). 표에 찍히는 시각은 그 창이 끝난 오디오 위치이지
콜백이 도착한 시각이 아닙니다. 구현은 `python/runtime/audio.py` 의
`HopFramer` 이고 라이브 콜백이 `HopFramer.feed` 로 씁니다.

인식이나 채점이 도중에 실패하면 그 녹음은 **버립니다**: 프레임·연속 횟수·
로그를 비우고 빨간 실패 배너를 띄운 뒤 녹음을 멈춥니다. 실패한 프레임
앞뒤를 이어 붙여 "연속 2회" 로 세지 않기 위해서이고, 그 뒤에 도착하는
오디오는 모델에 닿지 않습니다. 마이크를 다시 누르거나 초기화하면
처음부터입니다.

웹에서 창 길이를 바꾸면 다음 입력부터 버퍼·연속 판정을 새로 시작합니다.
현재 배포된 Unity ONNX 모델과 비교할 때는 기본 2.5초를 유지하세요.
악기놀이 프로필에는 같은 실시간 판정 경로로 녹음 파일을 재생하는 기능이 있습니다.
별도 녹음 저장·정답 빌드 화면은 없으며 해당 CLI 도구는 유지합니다.
즉석 정답의 단어별 임계값 측정과 아동용 streaming profile은 유지합니다.
정답 카탈로그 없이도 실행할 수 있습니다.

```powershell
python -m python.tools.web_test          # http://127.0.0.1:7860
```

학습한 아동 체크포인트로 테스트하려면 모델과 아동 판정 설정을 함께 지정합니다.

```powershell
python -m python.tools.web_test --model python/tools/child_finetune/runs/child/best --matrix shared/confusion_matrices/ko_child_v2.json
```

체크포인트는 Git에 포함되지 않으므로 별도로 보관해야 합니다.

아동 확정 정책(보정된 정확일치 1프레임 예외)까지 그대로 쓰려면 카탈로그도
함께 지정해야 합니다. `--matrix` 나 `--targets` 가 빠지면 정책이 상속되지
않습니다. `--model` 은 다릅니다 — 런타임은 **음향 모델의 신원을 확인하지
않으므로**, 매트릭스와 카탈로그만 맞으면 다른 모델에서도 예외가 적용됩니다.
보정은 아동 파인튜닝 모델에서 잰 값이니 모델을 맞추는 것은 호출자의
책임이고, 아래 세 인자를 모두 준 조합만 검증된 구성입니다.

```powershell
python -m python.tools.web_test --model python/tools/child_finetune/runs/child/best --matrix shared/confusion_matrices/ko_child_v2.json --targets shared/targets_child.json
```

`--targets` 만 주고 기본 매트릭스(ko_child_v1)로 두면 카탈로그가 다른
매트릭스용이라 상속되지 않습니다. 설정을 바꾸면 UI를 다시 시작하세요.
확정 시 쉬운 한글 설명·음절별 참고 안내·공통 JSON을 제공합니다. 기존 모델 오류표가
보류하는 차이는 주 안내에서 제외하고 접힌 원본 비교에 남깁니다.
0.8.1부터 웹·C# 모두 같은 보류 규칙을 적용합니다. `KoreanFeedback`은 표시용이며
원본 비교는 `RawKoreanFeedback`과 `Alignment`에 남습니다. 모델 자료가 없거나 부족한
차이도 내부적으로 보류합니다. 0.8.3부터 보류된 음절은 화면에 `문제 없음`으로 표시하며,
남은 지적이 없으면 `발음에 문제 없어요.`라고 안내합니다. 내부 상태와 원본 차이는 유지합니다.
[변경 내용과 실제 파일 검증](docs/MODEL_AWARE_FEEDBACK_2026-09-09.md).
웹과 Unity가 하나의 C# 실행 모듈로 연결된 상태는 아닙니다.

첫 인식 시 모델(~1.2GB)이 로드되어 1~2분 걸립니다.

### 단어 추가

```csv
# shared/words.csv  (UTF-8 BOM)
answer_id,text,language,action_group
mom,엄마,ko
apple,사과,ko
lift_plain,들어,ko,lift
lift_gerund,들기,ko,lift
shake_plain,흔들어,ko,shake
```

`action_group` 은 선택 항목입니다. 비워 두면 예전과 같습니다. 같은 그룹의
두 줄은 **같은 동작을 부르는 두 표현**이라 서로 경쟁하지 않습니다. 빌드는
그룹이 있는 모든 답을 `contrastive_alternatives` 로 넣습니다. 같은 그룹의
표현은 비교 기준을 보완하고, 다른 그룹의 표현만 통과 취소에 쓰입니다. 런타임은
통과한 답이 있을 때만 그 목록을 보고, **다른 동작이 발화 전체에 더 잘 맞으면
통과를 취소**합니다(점수·정렬·임계값은 그대로, 취소 사유만 따로 실립니다).
다른 동작을 대신 정답으로 바꾸지는 않습니다. 들기 `t ɯ l k i` 가 흔들기
`h ɯ n t ɯ l k i` 의 뒷부분이라 창이 어디서든 시작할 수 있는 부분열 채점이
흔들기 발화를 들기로 0.997 에 받아주던 문제를 막습니다.

확정은 `Passed`·`Confirmed`·확정 이벤트로 판단하세요. 보호 규칙이 취소한
경우에도 원래 점수는 유지되므로 `Score >= Threshold` 만으로 판단하면 안 됩니다.
아동 악기 표현 10종의 등록·실측 결과·Unity 적용 방법은
[2026-09-07 검증 기록](docs/MUSIC_ACTION_RELIABILITY_2026-09-07.md)을 참고하세요.
현재 해당 아동 카탈로그는 35개이며, 성인 카탈로그 29개는 이번에 재생성하지 않았습니다.

```powershell
python -m python.build.build_targets     # → shared/targets.json
```

Unity 에서는 `Tools → Phoneme Matching → 정답 데이터 다시 만들기` 가 이 빌드를
대신 돌리고 `StreamingAssets` 까지 복사합니다.

**모델이 낼 수 없는 음절이 있습니다.** 출력은 1205항목(한글 1202개 + 특수
항목 3개)이고, 목록에 없는 음절은 그 철자로 정확히 전사되지 않습니다. 단어를 넣기 전에
확인하세요. 손으로 볼 필요는 없습니다.

```powershell
python -m python.tools.vocab_coverage        # 배포 어휘 vs 커리큘럼 전부
python -m python.tools.vocab_coverage --strict --allow 놔
```

지금 커리큘럼에서는 `놔`(`놔`·`놔요`) 하나가 여기 걸립니다. 배포 중인 두 모델
로는 정확히 전사할 수 없습니다. 다만 매칭은 IPA 와 혼동행렬로 하므로 "항상
확정 실패"와 같은 말은 아니고, 실제 확정률은 그 단어 녹음으로 재야 합니다.

`놔`만 빠진 것은 아닙니다. 분할에 배정된 아동 전사에서는 누락 철자 523종,
그중 기존 글자와 같은 **단독 IPA**를 내는 110종을 확인했습니다. 남은 413종은
391가지 IPA 조합이지만, 문맥을 포함한 최소 추가 출력 수를 뜻하지는 않습니다.
**어휘·발음 통합은 문서 정리 후 보류**했으며 추가 학습·배포는 하지 않았습니다.
[분석 결과와 남은 판단](docs/VOCABULARY_IPA_COVERAGE_2026-09-06.md)을 참고하세요.

> **2026-09-06 정정.** 이 문단에 오래 적혀 있던 "코퍼스에 64건뿐"은 틀린
> 수치였습니다. 인덱스 2,765,817행을 훑고 **분할에 배정된 전사**를 기준으로
> 다시 세니, `놔` 가 있으면서 다른 어휘 밖 음절은 없고 1~15초인 발화가
> train 235 · dev 11 · valid 74 · test 63 건입니다. train 쪽은 화자 179명
> 1,420초이고 383개 파일 전부 16kHz 모노로 열립니다 (헤더만 확인). 따라서
> **"학습으로 메우기 어렵다"와 "다른 동사로 바꾸는 것이 맞다"는 근거가
> 확인되지 않았습니다.** 반대로 학습이 된다는 근거도 없습니다 — 분할에
> 배정된 전사 중 `놔`·`놔요` 를 단독으로 말한 발화는 0건이라, 코퍼스만으로는
> 단어 단위 정확도를 잴 수단이 없습니다. 자세한 것은
> [python/tools/child_finetune/validation-vocab-2026-09-06.md](python/tools/child_finetune/validation-vocab-2026-09-06.md).

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

```powershell
python -m python.tools.vocab_coverage                     # 어휘 밖 음절 점검
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

var buffer  = new AudioWindowBuffer(2.5f, 0.5f);   // 창 2.5초 · hop 0.5초
var session = new PronunciationSession(matrix, catalog, recognizer, buffer: buffer);

session.Begin("사과");                          // 문제 시작
while (session.TryPush(out var frame))          // 새 샘플 0.5초당 한 번
{
    if (frame.Confirmed) { 다음문제로(); break; }
}
session.End();
```

오디오는 앱 방식에 따라 한 줄만 다릅니다.

```csharp
buffer.Append(raw, clip.frequency);                                  // Microphone
buffer.AppendInterleaved(data, ch, AudioSettings.outputSampleRate);  // OnAudioFilterRead
```

웹과 같은 규칙입니다. `AudioWindowBuffer` 에 hop 을 주면 새 샘플 8,000개마다
프레임을 하나씩 FIFO 로 쌓고 `PronunciationSession.TryPush` 가 순서대로
꺼내 채점합니다. 덜 찼으면 추론 없이 `false` 입니다. 패키지 안에서는
`PronunciationListener` 와 에디터 발음 테스트 창이 `TryPush` 를 쓰고,
저장소의 `PronunciationTestBench` 는 `MicrophoneRollingBuffer.TryReadFrame`
으로 직접 프레임을 꺼내 자기 `StreamingMatcher` 로 채점합니다.

**기존 코드는 그대로 돕니다.** 창을 직접 넘기는 `Push(float[])` 와 인자
없는 `Push()` 는 수동 API 로 남겨 두었습니다. 다만 `Push()` 를 타이머로
돌리는 통합은 여전히 "부른 횟수 = 채점 횟수" 이므로, 청크 크기와 무관한
동일성을 원하면 `TryPush` 로 옮겨야 합니다. hop 을 웹과 맞추려면 기본
0.5초를 그대로 쓰세요.

보장 범위는 **동일한 정규화 16 kHz 모노 PCM** 입니다. 다른 샘플레이트를
청크로 나눠 넣을 때의 리샘플링은 이번에 손대지 않았고, 그 경우까지
동일하다고 주장하지 않습니다.

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

주의: **VRAM 1.2GB** 를 가중치가 차지하고, 측정 PC의 **첫 추론은 약 1.9초**였습니다.
VR 렌더링과 GPU 를 공유할 때의 경합은 아직 미측정이며, 필요하면
`ScheduleIterable` 로 레이어를 여러 프레임에 나눠야 합니다.

### 배포 전제

**온디바이스 VR 이 아니라 PC 테더링**입니다. 그래서 양자화가 불필요해졌고,
정확도 우선으로 모델을 고를 수 있습니다. 파이썬은 빌드 타임 도구로만
남습니다.

---

## 8. 측정 결과

> **아동 발화 기준 정확도는 아직 확정되지 않았습니다.** 아래는 측정 조건과
> 함께 읽어야 합니다. 조건을 뗀 숫자 하나는 의미가 없습니다.

### 실제 아동 발화 (AI Hub, 5~7세)

아동 전용 모델과 설정이 따로 있습니다 — 성인 모델은 손대지 않았습니다.
자세한 것은 [child_finetune](python/tools/child_finetune/README.md) 를 보세요.

측정 조건이 둘로 나뉩니다. **단독 단어**가 제품 조건입니다(화면에 단어를
띄우고 아이가 그 단어만 말함). **덧붙임**은 같은 단어를 문장 안에서 말한
경우로, 아이가 `"사과요"` 처럼 말을 더할 때에 해당합니다.

| | 단독 단어 | 덧붙임 | 오확정 |
|---|---|---|---|
| 성인 모델 + 성인 설정 | 37.3% | 48.5% | 3.50% |
| 아동 모델 + 성인 설정 | 91.3% | 91.3% | 6.47% |
| **아동 모델 + 아동 설정** | **88.1%** | **94.8%** | **0.70%** |

가운데 줄은 검출이 높아 보이지만 오확정이 예산(1%)의 6배라 쓸 수 있는
동작점이 아닙니다.

test 분할, 처음 보는 아이 52명, 단독 발화 126건 기준입니다. 표본이 작아
단독 수치의 95% 구간은 ±5.7pp 입니다.

**단독 근거가 있는 단어는 9개뿐입니다** (엄마·할머니·사과·우유·빵·와요·
먹어요·친구·책). 나머지 20개는 코퍼스에 단독 발화가 없어 문장에서만
쟀습니다. `실로폰` 은 276만 발화 어디에도 없습니다.

문장으로 맞춘 값이 단독 조건에서 더 나빴던 적이 있습니다(88.8% 대 90.0%).
두 조건은 따로 재야 합니다 — `skip_cost` 는 창 밖 음소에 매기는 벌점인데
단독 단어는 창 밖이 비어 있어 아무 일도 하지 않습니다.

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

새로 받은 소스에서는 Git에 저장된 배포 카탈로그를 테스트 입력으로 복사합니다.
`shared/targets*.json`은 생성 파일이므로 Git에서 제외됩니다.

```powershell
Copy-Item unity/Assets/StreamingAssets/targets.json shared/targets.json
Copy-Item unity/Assets/StreamingAssets/targets_child.json shared/targets_child.json
python -m python.build.build_music_presentation  # 발표용 카탈로그 동기화
python -m pytest python/tests -v
dotnet test csharp/PhonemeMatching.Tests    # Unity 불필요
```

이번 커밋 대상 파일만 별도 경로에 꺼내 위 절차로 확인한 결과는 **Python 550개·C# 216개 통과**입니다.
별도로 진행하던 모델 학습 실험 변경은 이 커밋에 포함하지 않았습니다.

2026-09-10의 0.8.5 배포 검증에서는 당시 전체 작업 트리의 Python 621개,
C# 216개와 Unity 2022.3.62f2 EditMode 42개가 통과했습니다.
실제 사용자 녹음 7개를 28개 시작 조건·139개 창으로 재생해 Unity와 Python 결과를 확인했습니다.
[최신 현황](docs/MUSIC_DEMO_STATUS_2026-09-10.md)과 [실제 음성 검증](docs/GRAB_REAL_VOICE_2026-09-10.md)을 참고하세요.

아래는 2026-09-06의 이전 검증 기록입니다. 당시 Python CPU/GPU 환경에서 각각 201개가 통과했습니다.
Unity 2022.3.62f2와 6000.3.18f1에서는 각각 Editor 27개, PlayMode 47개가 통과했습니다.
Editor는 기존 JSON 보존·모델별 설정을, PlayMode는 공통 코어와 Listener 이벤트 순서·초기화를 검사합니다.
공통 JSON 결과, 24개 한글 설명 사례, 전체 한글 11,172음절, Session의 확정 스냅샷을
검증합니다. 실제 아동 정확도 평가는 이 자동 테스트와 별개입니다.

수정본의 Git 의존성 설치와 두 Unity 버전의 Windows x64 Mono 빌드·실제 GPU 추론도
확인했습니다. 웹 버퍼 수정 후에는 58개 합성 음성/199개 창을 앞선 Unity 실행 기록과
대조해 인식·점수·확정 시점 차이가 없음을 확인했습니다. 마이크 실기기·IL2CPP는 미검증입니다.
설치 방식, 모델, 경고, 실행 기록 재사용 범위는
[상세 검증 기록](unity/Packages/com.domicube.phoneme-matching/Documentation~/validation-2026-09-06.md)을 참고하세요.

**파이썬에서 C# 으로 옮긴 것은 전부 벡터로 대조합니다.** "옮겼다"가 아니라
"같은 값이 나온다"를 확인하는 게 목적입니다.

| 대조 | 케이스 |
|---|---|
| 매처·Confusion Matrix·자모/IPA | 340 |
| CTC 디코더 ↔ 파이썬 `batch_decode` | 43 (36개는 실제 ONNX argmax) |
| ONNX ↔ PyTorch 한글·IPA | 36 |
| Unity 2022 Sentis 1.2 ↔ 파이썬 토큰 id | 4클립 × 124프레임 |

위 36클립·4클립 수치는 기존 성인 모델 기준 벡터입니다. 최신 아동 모델의 58클립
연속 입력 대조와는 다른 검사이며, 합성 음성을 실제 사용자 정확도로 일반화하지 않습니다.

매처나 matrix 를 고쳤으면 `export_parity_vectors`, 모델이나 어휘를 바꿨으면
`export_ctc_vocab` 로 벡터를 다시 만드세요.

Unity 대조는 [python/tools/sentis_parity/](python/tools/sentis_parity/) 에
하네스가 있습니다. **임포트가 됐다는 것과 계산이 맞다는 것은 다른
문제입니다** — Sentis 1.2 가 그 사이에서 죽는 것을 이 검사로 찾았습니다.

---

## 11. 알려진 한계

| 한계 | 원인 | 영향 |
|---|---|---|
| **짧은 단어 검출 실패** | 음소 하나가 틀리면 점수가 크게 떨어짐 | 성인 모델에서 아동 3음소 21%. 아동 모델 + 단어별 임계값으로 대부분 해소됐고(`빵` 4%→98%, `책` 10%→92%), 성인 모델 경로에는 그대로 남음 |
| **`놔`·`놔요` 정확한 전사 불가** | 음절 `놔` 가 배포 어휘 1205개에 없음 | 정확히 발음해도 그 글자로는 전사되지 않음. 확정 자체는 IPA 매칭이 결정하므로 실측 필요. 어휘를 1206으로 넓힌 **학습 전** 후보가 `runs/child-vocab-0906-c4e7/init` 에 있고, 배포본은 그대로 1205 |
| **아동에서 점수 분포가 겹침** | 발음 편차가 큼 | 임계값만으로 검출·오확정을 나눌 수 없음 |
| Confusion Matrix 미검증 | 문헌 기반 + 합성음 튜닝 | 아동 정렬 935건으로 재학습해도 1쌍만 바뀌어, 손으로 넣은 값이 그대로 남아 있음 |
| 경음(ㄲ/ㄸ/ㅃ/ㅆ/ㅉ) 약함 | 음향적으로 미세 | 토끼·빵·아빠 |
| 활음(ㅛ/ㅑ) 손실 | | 우유·와요 |
| 발달지연 아동 데이터 없음 | 공개 코퍼스 부재 | 실제 대상에 대한 근거가 없음 |
| ONNX 1.18GB | FP32 large | 임포트 느림, Release 배포 필요 |
| 첫 추론 1.9초 | 셰이더 컴파일 + 가중치 업로드 | 로딩 중 `Warmup()` 필수 |
| GPU 경합 미측정 | VR 렌더링과 같은 GPU | 90fps(11ms)에서 20ms 를 한 번에 던지면 끊길 수 있음 |

### 알려진 이슈

- **악기놀이 0.8.5의 남은 실패** — 실제 잡아 녹음 47번은 실패하며 45번은 시작 시점에 따른
  변동이 있습니다. 잡아먹/잡아당 포함 문장을 오답으로 검사하면 앞부분에서 먼저 확정하는
  문제가 각각 21/30, 25/30 남아 있습니다. [단계별 오확정률과 측정 범위](docs/MUSIC_DEMO_STATUS_2026-09-10.md).

- **짧은 발화의 연속 2회 확정 누락 — 미해결** — 아동 모델에서도 한 번만
  통과하는 사례가 있습니다. 샘플 기준 hop 수정 전후를 비교하면 0.5초 입력은
  동일했지만, 더 짧거나 불규칙한 입력을 가정하면 기존에 통과하던 `책`이
  수정 후 실패했습니다. 입력 간격 문제의 해결과 짧은 단어 검출 개선은
  별개이며, 일부 조건의 기존 통과 사례 손실을 확인했습니다.
  [실측 결과·Unity 검증 범위·남은 작업](docs/SHORT_WORD_CONFIRMATION_2026-09-06.md)을 참고하세요.
- **추론 엔진 설치 전 사용** — 패키지와 `LessonExample`은 0.8.0에서 엔진 없이도
  컴파일되도록 수정했습니다. 실제 음성 인식에는 추론 엔진과 모델 설치가 필요합니다.
- **Unity 6 Editor 추론 종료 경고** — Inference Engine 2.6.1의 정적 GPU 버퍼에서
  Persistent 할당 경고를 확인했습니다. 검사한 Windows Player에는 같은 경고가 없었습니다.
- **Windows MeCab** — g2pkk 비교 경로는 `mecab-ko` 어댑터를 사용하며 실행 중 자동 설치하지 않습니다. 기본 인식·판정 경로는 mecab-free 한국어 규칙을 씁니다.
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

### 막고 있는 것

1. **출력 어휘·IPA 통합 — 보류** — `놔` 한 글자뿐 아니라 아동 전사의 누락
   철자와 동일 발음 문제까지 분석했습니다. 523종 중 110종은 기존 글자와
   단독 IPA가 같지만 문맥에서도 같다는 보장은 없습니다. 사용자 요청으로
   문서만 정리했으며, 새 어휘·라벨 방식·추가 학습 범위는 미정입니다.
   `놔`를 추가한 1206 출력 후보는 미학습 상태로 보존하고 배포본은 유지합니다.
   [어휘·IPA 분석 기록](docs/VOCABULARY_IPA_COVERAGE_2026-09-06.md)
2. **아이 1명이 29단어를 한 번씩** — 근거 발화가 0~2건인 단어가 9개입니다
   (`쳐`·`실로폰`·`흔들어요` 등). 구조적으로 불가능한 단어는 1명으로
   드러납니다 — `놔` 가 그렇게 발견됐습니다. 5분이면 됩니다
3. **Unity 실기기 확인** — Editor 스크립트는 Unity 안에서만 컴파일됩니다

### 근거가 없는 것

4. **발달지연 아동 데이터** — 실제 사용자인데 공개 코퍼스가 없어 측정한 적이
   없습니다. 10~15명이면 지금 값이 통하는지 처음으로 알 수 있습니다
5. **단어별 검출률** — 29단어 중 9개만 실측했습니다. 나머지는 문장에서만
   쟀거나 근거가 없습니다. 아이 25명이면 ±6pp 로 잽니다

### 개선 여지

6. **Confusion Matrix 실측** — 61쌍 중 대부분이 문헌 기반입니다. 아동 정렬
   935건으로 재학습해도 1쌍만 바뀌었습니다. 프레임 캐시를 늘리면 같이
   좋아집니다
7. **연속 확인 1회 vs 2회** — 1회가 검출 6.3pp 높은데, 오확정을 코퍼스
   발화로만 재서 헤드셋 잡음이 빠져 있습니다. 잡음 녹음이 생기면 정할 값
8. VR 씬에서 GPU 경합 측정
9. (검토) FP16 ≈ 600MB 또는 더 작은 모델

### 손대지 않은 것

`learn_matrix.py`·`budget_curve.py`·`sensitivity.py` 는 아직 옛 파일명
(`tune_frames.json`)을 읽어 실행되지 않습니다. `autotune.py` 는 새 캐시로
옮겼지만 음소 길이별 임계값과 문장 조건을 써서 지금 설정을 재현하지
못합니다 — 아동 설정을 다시 맞출 때는 `fit_streaming.py` 를 쓰세요.
