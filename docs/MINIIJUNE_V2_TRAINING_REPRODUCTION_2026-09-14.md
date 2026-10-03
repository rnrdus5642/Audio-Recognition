**Miniijune V2 학습량과 재현 가능성 — 2026-09-14**

**공개된 구조와 학습 설정을 바탕으로 비슷한 모델을 직접 학습하는 것은 가능하다. 다만 정확한 학습 자료·라벨 생성 코드·시작 가중치가 확인되지 않아 동일 모델과 동일 성적의 재현을 보장할 수는 없다.** 현재 우리 데이터 처리 및 GPU 학습 기반을 활용하되 자모 라벨, 토크나이저, CTC 출력층과 평가 방법을 새로 연결해야 한다. 이번 조사는 학습을 실행하지 않았다.

대상은 [Miniijune V2](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2)의 revision `448e988f77d2e95cc0636bf0eb98132b4a544849`다. 모델 카드뿐 아니라 공개 `training_args.bin`, `config.json`, 토크나이저·전처리 파일, 저장소 파일 목록과 이력을 확인했다. 저장된 Python 객체는 실행하지 않고 pickle 연산을 정적으로 해석해 설정을 확인했다. 제작자의 로컬 사용자명·경로는 이 문서와 공개용 집계에 싣지 않는다.

**얼마나 학습했는가**

| 항목 | 확인한 내용 | 근거 수준 |
|---|---|---|
| 기반 | `facebook/wav2vec2-xls-r-300m` | 모델 카드 표기. 실제 초기화 코드는 미공개 |
| 아동 학습 반복 | `num_train_epochs=4`, `max_steps=-1` | 저장된 설정 |
| 마지막 공개 학습 로그 | 30,000 optimizer step, 3.8408 epoch | 카드의 실제 기록 |
| 업데이트당 배치 | GPU당 4개, 누적 8회, 단일 GPU·프로세스 | 저장된 설정. 통상 유효 배치 32개 |
| 학습 자료 규모 | 약 25만 발화로 추정 | `30,000 × 32 ÷ 3.8408 ≈ 249,948`. 발화 목록이 없어 정확한 수·고유 녹음 수는 미확인 |
| 누적 발화 처리 | 30,000 step까지 약 96만 회 | 유효 배치로 계산한 반복 포함 근사치. 서로 다른 녹음 96만 개가 아님 |
| 전체 음성 시간 | 미공개 | 평균 발화 길이와 필터링 목록 없음 |
| 제작자의 GPU 종류·VRAM | 미공개 | 저장 설정으로는 CUDA GPU 1개임을 확인 |
| V2 실제 소요 시간 | 미공개 | 초기 음절 모델의 32시간 기록을 V2 시간으로 사용할 수 없음 |

설정 4 epoch는 계획된 반복 횟수다. 공개 로그는 3.8408 epoch까지이고 전체 `trainer_state.json`·종료 로그가 없어 실제 종료 시점과 추가 학습 이력은 확정하지 않는다. epoch 경계의 누적 배치 처리와 반올림 때문에 위 역산은 규모 추정으로만 사용한다. 카드에 `None dataset`이라고 적힌 것은 데이터셋 이름이 자동 문서에 채워지지 않았다는 뜻으로 읽어야 하며, 데이터 없이 학습했다는 근거가 아니다. [V2 모델 카드](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2)

V2의 저장된 출력 폴더에는 `011.한국어 아동 음성 데이터`라는 명칭이 포함되어 있다. 같은 제작자의 초기 모델에는 AI Hub 아동 자료, G2PK로 문장 전사를 발음 표기로 변환, 학습 250,000개·검증 5,000개·테스트 5,000개가 명시돼 있다. 이를 합치면 V2도 해당 아동 데이터 계열을 사용했다는 근거가 강해진다. 그러나 **동일한 25만 파일·동일한 연령·동일한 분할·동일한 라벨 규칙을 썼는지는 확인되지 않는다.** [V2 저장 학습 설정](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2/blob/448e988f77d2e95cc0636bf0eb98132b4a544849/training_args.bin), [초기 모델 설명](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation)

**학습 진행에 따른 성적**

| 업데이트 수 | epoch | 제작자 CER |
|---|---|---|
| 2,000 | 0.2561 | 21.36% |
| 8,000 | 1.0242 | 10.62% |
| 16,000 | 2.0484 | 8.72% |
| 24,000 | 3.0726 | 8.00% |
| 30,000 | 3.8408 | 7.87% |

후반 24,000→30,000 step의 개선은 0.13%p여서 해당 평가 지표의 개선 폭은 작아지고 있었다. 이것으로 모든 아동·모든 단어에서 학습이 충분했다고 단정할 수 없다. 검증 손실은 8,000 step에서 0.7361, 30,000 step에서 0.7929로 오히려 높아져 CER와 같은 방향으로 움직이지 않았다. 저장 설정은 **CER가 낮은 체크포인트 선택**이다. 두 지표만으로 과적합 여부를 확정하지 않는다. CER의 구체적인 계산 코드·평가 자료는 공개되지 않았고, 이 수치는 우리 [64개 모델 실험](WAV2VEC2_KOREAN_MODEL_SELECTION_2026-09-13.md)의 기준 소리 오류율과 직접 비교할 수 없다. [제작자 학습 결과](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2#training-results)

저장소 루트와 `final/`의 가중치 LFS SHA-256이 같고, 설정·토크나이저·학습 인자도 같았다. `final/`이 별도의 추가 학습 모델이라는 근거는 없다. 공개 루트 가중치 SHA-256 `eb6d669fa610871da49c4cd2b05745e4042cbbd079a5a65a81c5d56ddc1ce76d`는 앞선 실제 추론에 사용한 파일의 검증값과 일치한다.

**재현할 수 있는 설정**

| 항목 | V2 저장값 |
|---|---|
| 입력 | 16 kHz, 정규화 적용, attention mask 사용 |
| 출력 | 자음 19개·모음 21개와 제어 기호를 사용하는 자모 CTC |
| 학습률 | `3e-5` |
| 옵티마이저 | `adamw_torch_fused` |
| Adam 계수 | beta1 `0.9`, beta2 `0.999`, epsilon `1e-8` |
| Weight decay | `0.0` |
| Gradient clipping | 최대 norm `1.0` |
| 학습률 변화 | cosine, warmup `2,000` step |
| 배치 | 학습 `4`, 평가 `1`, gradient accumulation `8` |
| 정밀도 | FP16 true, BF16 false |
| Seed | `42`, 별도 data seed 미설정 |
| 학습 반복 | `4` epoch |
| 평가·저장 | `2,000` step마다, CER 최소값 선택, 체크포인트 최대 `2`개 유지 |
| 길이별 묶음 | `group_by_length=false` |
| Gradient checkpointing | false |
| CTC 손실 | reduction `mean`, zero_infinity true |
| SpecAugment 시간 마스크 | probability `0.05`, length `10` frame, minimum `2` masks |
| SpecAugment 특징 마스크 | probability `0.0` |
| Dropout | attention/hidden/feature projection `0.1`, activation/final `0.0`, layerdrop `0.1` |
| 제작 환경 표기 | Transformers `4.57.0`, PyTorch `2.9.1+cu128`, Datasets `4.2.0`, Tokenizers `0.22.1` |

위 표는 저장된 학습·모델·전처리 설정이며 원본 학습 코드의 모든 동작을 증명하지 않는다. 예를 들어 일부 층을 고정했는지는 이 설정만으로 알 수 없다. [학습 인자](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2/blob/448e988f77d2e95cc0636bf0eb98132b4a544849/training_args.bin), [모델 설정](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2/blob/448e988f77d2e95cc0636bf0eb98132b4a544849/config.json), [전처리 설정](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2/blob/448e988f77d2e95cc0636bf0eb98132b4a544849/preprocessor_config.json)

V1과 V2는 `config.json`, `vocab.json`, `tokenizer_config.json`, `preprocessor_config.json`이 내용상 같았다. 저장 학습 인자에서 경로·저장소 이름을 제외한 차이는 **학습률 `1e-5→3e-5`, clipping `1.5→1.0`**였다. V1의 카드 CER 10.84%와 V2의 7.87% 차이를 이 두 변경의 인과적 효과라고 단정하지 않는다. 동일한 자료·초기화·학습 코드로 통제된 비교인지 알 수 없기 때문이다. `resume_from_checkpoint=None`도 코드에서 V1 가중치를 직접 불러오지 않았다는 증거는 아니다. [V1](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based), [V2](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2)

**정확한 재현을 막는 정보**

- V2의 사용 파일·화자·연령·학습/검증/테스트 분할과 음성 총시간.
- 오디오 길이 제한, 잘못된 전사와 발음 사례의 필터링 방법, 음량·잡음·속도 변형 여부.
- G2PK 옵션, 자모 분해·초성 ㅇ·겹받침·발음 변이·띄어쓰기 처리 방법과 실제 라벨 예시.
- Meta 원본에서 새로 시작했는지, 다른 체크포인트를 이어 학습했는지와 층 동결 여부.
- 학습 스크립트, 전체 진행 상태, 실제 GPU 종류와 소요 시간.

공개 저장소 파일 목록과 관련 검색에서는 위 정보를 채워 줄 학습 코드나 분할 목록을 찾지 못했다. 같은 제작자의 `semi-supervised_V2`도 확인했으나 이름만으로 의사 라벨 생성 모델·신뢰도 필터·혼합 비율을 알 수 없었다. [V2 파일 목록](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2/tree/448e988f77d2e95cc0636bf0eb98132b4a544849), [semi V2](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-semi-supervised_V2)

**우리 환경에서 진행할 수 있는 범위**

현재 CSV를 직접 집계하면 학습 목록은 **123,943개·884명·199.76시간**, 개발용 검증 목록은 **787개·50명·1.43시간**이다. 기존 음절 모델용으로 선별된 목록이므로 자모 모델 학습용 확정 목록으로 그대로 사용하지 않는다. 이전 음절 어휘에 없는 글자 때문에 제외한 발화를 다시 검토하고, 화자 분리·파일 존재·라벨 변환을 새로 검사해야 한다. 이번에는 목록을 집계했으며 전체 오디오를 재검사하지 않았다. [현재 학습 코드](../python/tools/child_finetune/train.py), [자료 선별 코드](../python/tools/child_finetune/manifest.py)

PC의 GPU는 RTX 5060 Ti 16GB다. 기존 동급 모델에서 약 200시간 자료를 8 epoch 학습한 약 7.05시간의 실행 기록이 있다. 이는 로컬 학습을 시도할 기반이 있다는 근거다. **V2 방식도 7시간이나 32시간에 끝난다는 예측은 아니다.** 자모 라벨 길이, 길이별 배치, feature encoder 동결, 초기화·수렴 속도가 다르므로 실제 소요 시간과 메모리는 예비 실행으로 측정해야 한다. [기존 학습 기록](../python/tools/child_finetune/README.md)

권장하는 재현 순서는 다음과 같다. 이는 아직 실행하지 않은 제안이다.

1. Meta XLS-R 300M 가중치와 한국어 자모·발음 라벨 규칙을 고정하고, 우리 자료의 학습·개발·최종 평가 화자를 먼저 분리한다.
2. G2PK 기반 발음 표기와 자모 변환으로 기본 인식 학습용 라벨을 만들되, 실제 발음이 원문과 다른 사례를 따로 검토한다. 발음 오류 안내 평가는 사람이 실제로 들은 소리의 라벨을 사용한다.
3. 위 V2 설정을 기본 비교 조건으로 삼아 토크나이저·CTC 출력층·학습 루프를 연결한다. feature encoder 동결 등 미공개 사항은 우리의 선택으로 명시한다. 길이 제한·작은 배치·누적·gradient checkpointing 변경도 재현과 다른 조건으로 기록한다.
4. 데이터 준비·유한한 손실·자모 출력 학습·VRAM·처리 속도를 작은 예비 실행으로 확인한다. 낮은 학습률이 무작위 새 출력층에서도 충분히 수렴하는지는 확인해야 하므로 짧은 시험의 낮은 정확도만으로 기반 모델을 탈락시키지 않는다.
5. 본 학습은 여러 epoch의 학습 곡선을 보고 결정한다. 우리 12.4만 발화에서는 유효 배치 32·4 epoch가 약 15,500 업데이트여서 V2 추정 31,200 업데이트와 다르다. 2,000 warmup을 유지할지 비율을 맞출지도 명시적으로 정한다.
6. 최종 평가는 정답 통과·오확정·실제 발음 보존을 분리한다. Miniijune의 AI Hub 학습 중복을 확인할 수 없으므로, 해당 자료만으로 공정한 두 모델의 미지 아동 성능 비교를 주장하지 않는다. 별도로 확보해 학습·튜닝에 쓰지 않은 아동 녹음을 비교에 사용한다.

V2의 출력층은 44칸인데 `vocab.json`에는 ID 41 대응이 비어 있고 EOS 44는 출력 범위 밖이다. 기존 V2를 사용할 때는 번호를 바꾸면 안 되지만, **Meta에서 새 출력층을 만들 때 이 불일치를 복제할 이유는 없다.** 자모와 blank·공백·unknown의 연속된 토큰 번호를 정의하고, 필요한 모든 라벨이 unknown 없이 표현되는지 확인한다. [토큰표](https://huggingface.co/Miniijune/wav2vec2-xls-r-300m-Korean-children-pronunciation-jamo-based-V2/blob/448e988f77d2e95cc0636bf0eb98132b4a544849/vocab.json)

Meta 원본은 이미 다국어 음향 표현을 사전학습한 모델이다. 위 계획은 그 표현을 가져와 우리 라벨로 CTC를 미세조정하는 작업이며, 대규모 음향 사전학습을 다시 하는 작업량이 아니다. [Meta 설명](https://huggingface.co/facebook/wav2vec2-xls-r-300m), [공식 XLS-R 미세조정 예제](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2)

설정 추출값, 버전별 학습 곡선, 파일 검증값과 로컬 목록 집계는 [조사 데이터](data/miniijune-training-audit-2026-09-14.json)에 보존했다. 제작자의 로컬 경로와 아동 녹음은 포함하지 않는다.
