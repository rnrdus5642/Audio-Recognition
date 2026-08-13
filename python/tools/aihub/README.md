# AI Hub 아동 음성 정리

[AI Hub 한국어 아동 음성 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=540)
를 받은 그대로 넣으면 학습·평가에 바로 쓸 수 있는 형태로 정리합니다.

휴대폰 인증만 되어 있으면 자동 승인입니다. 전체 5,000시간이고 Validation 이
그중 약 10%(522시간)입니다.

---

## 순서

```bash
python python/tools/aihub/index.py       # 라벨 → index.csv
python python/tools/aihub/organize.py    # 오디오 → 나이별 폴더
python python/tools/aihub/splits.py      # 화자 분리
python python/tools/aihub/verify.py      # 검사
```

경로는 [config.py](config.py) 에 있습니다. 환경변수로도 바꿀 수 있습니다.

```
AIHUB_RAW   받은 폴더        기본 C:\Users\user\Desktop\AIHUB
AIHUB_OUT   정리 결과        기본 C:\Users\user\Desktop\AIHUB정리
```

받은 폴더는 **AI Hub 가 준 구조 그대로 두면 됩니다.** 아카이브를 하위
경로 어디서든 찾아냅니다.

먼저 뭐가 잡히는지만 보려면:

```bash
python python/tools/aihub/archive.py
```

---

## 결과

```
AIHUB정리/
  index.csv          전 발화 메타 (전사 포함)
  splits.json        화자별 train / valid / test
  audio/
    05세/formatted/*.wav
    06세/free/*.wav
    07세/...
```

`index.csv` 한 줄이 발화 하나입니다.

| 열 | 내용 |
|---|---|
| `subset` `style` | Training/Validation · 낭독(formatted)/자유발화(free) |
| `wav` | 파일명 — `audio/` 안에서 이 이름으로 찾습니다 |
| `age` `gender` `school_year` | 화자 속성 |
| `speaker` | **해시된** 화자 ID |
| `seconds` | 길이 |
| `speech_start` `speech_end` | 발화 구간 — 학습 때 앞뒤 무음 자르는 데 씁니다 |
| `noise` `device` `environ` `snr` `quality` | 녹음 조건 |
| `text` | 전사 |

---

## 알아둘 것

### 파트 파일을 합치지 않습니다

AI Hub 는 tar 을 1GB 파트로 쪼개 줍니다. 합치면 순간 점유가 두 배가 되는데
(490GB 받으면 980GB 필요), 여기서는 파트를 이어붙인 **스트림으로 바로
읽습니다.** 합칠 필요도, 중간 파일을 지울 필요도 없습니다.

라벨도 풀지 않고 스트림에서 바로 읽습니다. 예전에 라벨 27만 개를 파일로
풀어 하나씩 열었을 때 22분이 걸렸는데, 순차 read 는 그보다 훨씬 빠릅니다.

### 화자 ID 는 해시됩니다

원본은 아이를 이름이나 이니셜로 적어둡니다 (`CHOIYEJUN`, `KDH`). 저장소에
들어가는 파일에 그대로 두면 아동 개인정보를 공개하는 게 됩니다. `index.csv`
와 `splits.json` 은 `sha256(대문자ID)[:12]` 만 씁니다.

분할에 필요한 건 "같은 아이가 양쪽에 걸치지 않는 것"뿐이라 해시로 충분합니다.

### 분할은 화자 단위입니다

같은 아이가 학습과 평가에 함께 들어가면 모델이 말이 아니라 목소리를
알아본 것이고, 그 뒤 측정한 숫자는 전부 부풀려집니다.

해시로 배정하므로 **데이터를 더 받아도 기존 화자의 소속은 안 바뀝니다.**
다시 돌려도 같은 결과가 나옵니다.

```
5~7세   train 70 / valid 15 / test 15
8~9세   train 만
```

8~9세를 학습에만 쓰는 이유는 제품 사용자가 5~7세이기 때문입니다. 그래도
학습에 넣는 건, 측정해보니 **나이 차이보다 아이별 개인차가 1.5배 컸기**
때문입니다 — 나이가 좀 많아도 "아이 목소리"를 가르치는 값이 있습니다.

10세 이상은 색인에만 남고 오디오는 꺼내지 않습니다. [config.py](config.py)
의 `AGES` 로 바꿀 수 있습니다.

### 평가 화자가 적으면 경고합니다

`verify.py` 는 valid/test 의 나이별 화자가 40명 미만이면 문제로 잡습니다.
예전에 화자 15명으로 튜닝했을 때 3음소 임계값이 0.925 같은 극단값으로
튀었고, 118명으로 늘리자 0.875 로 내려왔습니다. 화자가 적으면 그 아이들
발음에 맞춘 값이 나옵니다.

### 다시 돌려도 됩니다

`organize.py` 는 이미 있는 파일을 건너뜁니다. 중간에 끊겨도 이어서 하면
됩니다.

---

## 용량

| | |
|---|---|
| 받는 총량 | 약 550GB |
| 5~9세만 추출 | 약 200GB |
| 5~7세만 추출 (`AGES` 축소 시) | 약 70GB |

추출이 끝나면 원본 아카이브는 지워도 됩니다. `index.csv` 에 메타가 다 있어서
다시 받을 일은 오디오가 더 필요할 때뿐입니다.

---

## 파일

| | |
|---|---|
| [config.py](config.py) | 경로·나이 |
| [archive.py](archive.py) | 아카이브 탐색 + 파트 스트리밍 |
| [index.py](index.py) | 라벨 → `index.csv` |
| [organize.py](organize.py) | 오디오 → 나이별 폴더 |
| [splits.py](splits.py) | 화자 분리 |
| [verify.py](verify.py) | 검사 |
| [rescue_splits.py](rescue_splits.py) | 옛 분할을 캐시에서 복원 (1회용, 완료) |
| [splits_v1.json](splits_v1.json) | 2026-08-11 튜닝에 쓴 분할 — 기록용 |

`splits_v1.json` 은 예전 숫자(아동 검출 48.7%, 5세 28.2%)가 어떤 화자에서
나왔는지의 기록입니다. 그 분할은 `tune` 화자가 15명뿐이라 새 작업에는 쓰지
않습니다. `splits.py` 가 만드는 분할을 쓰세요.

---

## 그 다음

정리가 끝나면 [child_tuning](../child_tuning/README.md) 으로 이어집니다.

```
1. 프레임 캐시 → 새 기준선     현재 모델로 추론
2. fine-tune
3. 프레임 캐시 → 비교          새 모델로 추론
4. matrix · 하이퍼파라미터 재튜닝
```

1번의 표본 설계(분할별 몇 발화, 어떤 단어)는 분할 크기가 나온 뒤에
정합니다. 지금 정하면 추측이 됩니다.
