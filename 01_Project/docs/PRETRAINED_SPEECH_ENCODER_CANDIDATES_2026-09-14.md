# 한국어 아동 자모 인식을 직접 학습하기 위한 출발 모델 검토

확인일: 2026-09-14. 공개 모델 카드, 설정, 가중치 목록, 논문과 공식 학습 예제를 확인한 사전 조사다. 아래 판단은 실제 비교 실험을 시작하기 전의 기록이다.

**후속 실험 완료:** 11개 인코더의 자모 출력층 학습, 주요 6개의 짧은 아동 적응 학습, Windows Unity GPU 실행 결과는 [실측 보고서](PRETRAINED_ENCODER_WINDOWS_UNITY_RESULTS_2026-09-14.md)에 정리했다. 현재 권고는 Unity 6에서는 한국어 HuBERT XLarge, Unity 2022.3 / Sentis 1.2에서는 한국어 HuBERT Large다. 아래의 미검증 표시는 사전 조사 당시 상태이며 최신 성능·호환성 판단에는 후속 보고서를 사용한다.

**판단:** XLS-R 300M을 기준 모델로 유지하고, OmniASR W2V 300M, W2V-BERT 2.0, 한국어 HuBERT-Large를 우선 비교한다. WavLM-Large는 발음 구별 성능을 확인할 추가 후보로 둔다. 성능을 우선하므로 1B급도 후보에서 제외하지 않는다. 아래 우선순위는 실험 설계에 관한 판단이며, 한국어 아동 인식 성능 순위가 아니다.

기존 [64개 체크포인트 비교](WAV2VEC2_KOREAN_MODEL_RESULTS_2026-09-13.md)는 공개된 인식 모델의 현재 성능을 비교한 실험이다. 같은 아동 자료로 여러 사전학습 인코더를 학습했을 때 어느 모델이 더 좋아지는지까지 검증한 것은 아니다. [Miniijune V2 학습 이력 조사](MINIIJUNE_V2_TRAINING_REPRODUCTION_2026-09-14.md)와도 구분한다.

**실제로 사용할 수 있는 후보**

| 역할 | 공개 체크포인트 | 규모·사전학습 | 비교할 이유 | 적용 시 확인할 부분 |
|---|---|---|---|---|
| 기준 | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | 약 300M, 43.6만 시간·128개 언어 | 기존 Wav2Vec2 학습·추론 코드와 가장 가까움 | 새 자모 토크나이저와 CTC 출력층, 학습 목록 재생성 필요 |
| 우선 | [Meta omniASR_W2V_300M](https://github.com/facebookresearch/omnilingual-asr#models) | 약 317M, 다국어 음성 약 430만 시간 | XLS-R과 비슷한 크기의 최신 사전학습 인코더 | 공식 구현은 fairseq2. 가중치 변환 또는 별도 학습 경로, 배포 변환 검증 필요 |
| 우선 | [facebook/w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0) | 공식 표기 600M, 약 450만 시간·143개 이상 언어. 언어 목록에 한국어 포함 | Conformer 구조와 훨씬 큰 사전학습 자료라는 다른 출발점 | 파형을 그대로 받는 기존 경로와 달리 음향 특징 전처리 필요. Unity용 내보내기 재검증 필요 |
| 우선 | [team-lucid/hubert-large-korean](https://huggingface.co/team-lucid/hubert-large-korean) | 카드 약 317M, 공개 인코더 가중치 약 315M. 한국어 약 4,000시간 | 한국어에 집중한 사전학습이 다국어 사전학습보다 유리한지 확인 가능 | 자모 출력층 추가 필요. 우리 아동 자료에서의 성능은 미확인 |
| 추가 | [microsoft/wavlm-large](https://huggingface.co/microsoft/wavlm-large) | 약 317M, 영어 약 9.4만 시간 | 음성 혼합·잡음 처리를 포함한 학습 방식이 발음 구별과 실제 녹음에 도움이 되는지 확인 | 영어 사전학습의 한국어 전이 성능은 직접 검증해야 함 |
| 추가 | [facebook/hubert-large-ll60k](https://huggingface.co/facebook/hubert-large-ll60k) | 약 300M, 영어 Libri-Light 약 6만 시간 | 한국어 HuBERT와 비교하여 사전학습 자료 차이의 영향을 볼 수 있음 | 같은 구조라도 사전학습 설정까지 모두 같은 대조 실험은 아님 |

위 모델들은 기본적으로 음성의 특징을 추출하는 인코더다. 이미 한국어 자모를 출력하는 완성 모델이라는 뜻이 아니다. 그 위에 우리가 정한 자모 목록을 출력하는 CTC 층을 붙이고, 라벨이 있는 아동 음성으로 학습하는 방향이다. 따라서 기존 음절 출력 목록에 `놔`가 없었던 문제를 그대로 물려받을 필요가 없다. 다만 출력할 수 있다는 것과 정확히 알아듣는다는 것은 별개다.

**Omnilingual을 새로 포함할 근거**

Meta는 SSL 인코더인 `omniASR_W2V_300M`, `omniASR_W2V_1B` 등의 가중치와, 인코더에서 CTC 모델을 학습하는 공식 예제를 공개했다. 사용할 출발점은 W2V 계열이다. 기존 다국어 CTC나 LLM-ASR 체크포인트와 구분해야 한다. 2025년 12월의 v2 표기는 공개 안내에서 CTC·LLM 계열에 붙으며, 여기서 조사한 SSL 인코더 이름에는 붙지 않는다. [공식 저장소](https://github.com/facebookresearch/omnilingual-asr), [인코더부터 CTC 학습하는 예제](https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/configs/ctc-from-encoder-recommendation.yaml).

원 논문 Table 24에서는 인코더들을 같은 AllASR 자료로 CTC 학습해 비교했다. 300M급의 FLEURS-102 평균 CER은 XLS-R 14.6%, OmniASR W2V 12.0%였지만, MLS는 각각 3.7%, 4.1%로 XLS-R이 더 좋았다. 따라서 비교에 포함할 근거는 있으나, 모든 환경에서 개선되는 결과도 한국어 아동 전용 결과도 아니다. SSL 자료는 상세 설명상 언어가 식별된 384만 시간과 미식별 46만 시간이며 내부 자료도 포함한다. 한국어의 정확한 사전학습 시간이나 우리 파일·화자와의 겹침은 확인되지 않았다. [논문 §3.3.4, §5.9](https://arxiv.org/html/2511.09690v1).

공식 아키텍처는 16kHz 파형 → Wav2Vec2 인코더 → CTC 선형 출력층이다. 기존 접근과 가깝지만, 현재 프로젝트의 Hugging Face 체크포인트 로더에 모델 이름만 넣으면 그대로 동작한다고 검증한 상태는 아니다. 공식 학습 설정의 대규모 배치나 학습 자원은 우리 환경에 맞게 바꿔야 한다. [아키텍처](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/README.md), [학습 안내](https://github.com/facebookresearch/omnilingual-asr/blob/main/workflows/recipes/wav2vec2/asr/README.md).

**한국어 HuBERT의 공개 상태**

Large는 실제로 공개되어 있고, 설정은 `HubertModel`, 24층, hidden size 1024다. HF 메타데이터상 safetensors의 파라미터는 315,438,720개이며 확인한 리비전은 `62054d17ebf51437336a371a2cdd80d5da2b44ac`이다. 카드의 약 317M이라는 표기는 이 수치와 구분한다. 카드 예제의 출력 차원 768은 설정의 1024와 맞지 않아 그대로 복사하지 않는다. [모델 카드](https://huggingface.co/team-lucid/hubert-large-korean), [설정](https://huggingface.co/team-lucid/hubert-large-korean/blob/main/config.json), [공개 메타데이터](https://huggingface.co/api/models/team-lucid/hubert-large-korean).

카드에 적힌 자료는 AI Hub 자유대화 음성(일반남여), 다화자 음성합성 데이터, 방송 콘텐츠 대화체 음성인식 데이터다. Miniijune 계열에서 확인한 한국어 아동 음성 자료와 같은 데이터셋 이름은 아니지만, 공개 설명만으로 모든 파일·화자의 독립성을 보장할 수는 없다. 카드의 학습 표에는 최대 40만 스텝이 제시되어 있으며 실제 완료 학습 로그를 이번에 검증한 것은 아니다. [학습 자료·설정](https://huggingface.co/team-lucid/hubert-large-korean#training-details).

한국어 XLarge도 공개되어 있다. 설정은 48층·1280차원, safetensors 파라미터 962,497,408개, 리비전은 `9aeb683b37383026a0ff74d3be90a215be9fc5b6`이다. 다만 카드 제목과 학습 표가 Base/Large 내용을 반복하므로 XLarge 고유의 학습 이력은 불충분하다. [XLarge 카드](https://huggingface.co/team-lucid/hubert-xlarge-korean), [설정](https://huggingface.co/team-lucid/hubert-xlarge-korean/blob/main/config.json), [메타데이터](https://huggingface.co/api/models/team-lucid/hubert-xlarge-korean).

**WavLM·HuBERT의 발음 평가 근거가 의미하는 범위**

2026년 한국어 유아 발음 평가 논문에서는 HuBERT와 WavLM의 특징으로 자음·모음의 정오를 분류했다. 인코더를 고정하고 로지스틱 회귀 분류기를 학습했으며, 분류 자료는 34명이고 화자 기준 학습 27명·평가 7명으로 나뉘었다. 자모를 시간 순서대로 전사하는 CTC 실험이나 어느 소리가 틀렸는지 위치를 찾는 실험은 아니다. 영어 사전학습 모델도 후보로 확인할 이유는 되지만, 우리 제품에서의 인식률·오확정률을 입증하지는 않는다. [논문 방법 및 결과](https://arxiv.org/html/2606.10213v1).

**큰 모델과 나머지 후보**

- 1B급 비교에는 [XLS-R 1B](https://huggingface.co/facebook/wav2vec2-xls-r-1b), [OmniASR W2V 1B](https://github.com/facebookresearch/omnilingual-asr#models), 한국어 HuBERT XLarge를 고려한다. 크기를 키웠을 때 개선되는지 확인하는 실험이며, 1B가 반드시 더 정확하다는 전제는 두지 않는다.
- [MMS 1B](https://huggingface.co/facebook/mms-1b)는 약 50만 시간의 다국어 사전학습 후보지만 현재 카드의 라이선스가 CC-BY-NC-4.0이다. ASR 완료 모델인 `mms-1b-all`과 구분한다.
- [XEUS](https://huggingface.co/espnet/xeus)는 577M E-Branchformer 인코더로, 100만 시간 이상 다국어 음성을 학습했다. 공식 카드의 현재 라이선스는 CC-BY-NC-SA-4.0이며 ESPnet 경로의 추가 구현도 필요해 제품 출발점으로는 우선순위를 낮춘다.
- XLS-R·한국어 HuBERT 카드는 Apache-2.0, W2V-BERT 2.0 카드는 MIT, [Omnilingual 저장소](https://github.com/facebookresearch/omnilingual-asr#license)는 코드·모델을 Apache-2.0으로 안내한다. 모델 라이선스와 우리가 추가로 쓰는 음성 자료의 이용 조건은 별개다.
- WavLM은 HF 카드가 연결하는 [UniSpeech LICENSE](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)는 CC-BY-SA-3.0인 반면, 공식 WavLM 코드가 속한 [UniLM LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE)는 MIT다. 코드 라이선스만 보고 해당 가중치도 MIT라고 단정하지 않는다.

현재 확인한 GPU는 RTX 5060 Ti 16GB다. 600M·1B 전체 미세조정의 최대 메모리와 학습 시간은 아직 측정하지 않았다. 짧은 발화 배치, 혼합 정밀도, gradient checkpointing, optimizer 메모리 절감이나 더 큰 학습 장비를 검토해야 한다. 공개된 추론 메모리나 기존 300M 학습 시간을 새 모델의 학습 요구량으로 대신 쓰면 안 된다.

**우리 프로젝트에서 비교할 방법**

1. 학습·개발·최종 평가 화자를 먼저 분리한다. 기존 음절 모델에서 표현할 수 없다는 이유로 제외했던 자료를 새 자모 목록 기준으로 재검토한다. 최종 평가용 새 아동 녹음은 임계값 조정에도 사용하지 않는다.
2. 목표로 제시한 단어와 아동이 실제로 낸 소리를 구분한다. 예를 들어 `잡아`를 요구했는데 `작아`처럼 `자까`로 발음한 경우, 목표 발음 `자바`와 실제 발음 `자까`를 별도 필드로 둔다. 목표 문장을 G2P로 변환하는 것만으로 실제 발음 오류의 라벨이 생기지는 않는다.
3. 정상적인 같은 발음은 일관되게 정규화한다. 사용자가 예로 든 `샅`과 `삳` 같은 비교는 뒤 음절과 연음 여부를 포함한 실제 발음 문맥에 따라 처리한다. 모든 문맥에서 두 철자를 일괄 치환하지 않는다.
4. 동일한 학습 자료·화자 분할·자모 정의를 사용한다. 모델에 필요한 음향 전처리와 적절한 학습률 범위는 각각 적용하고, 비교 가능한 조정 예산을 배정한다. 새 출력층 학습 초기의 몇 번 결과만으로 탈락시키지 않는다.
5. 자모 오류율, 목표 단어 정답 통과율, 다른 단어의 오확정률을 함께 측정한다. `놔/와`, `잡아/작아`, 평음·경음·격음, 받침·모음 혼동을 분리해서 본다. 기존 `흔들어` 안의 `들어` 부분 일치 문제는 인코더 오류와 별도로 집계한다.
6. 발음 오류 안내는 사람의 실제 발음 라벨을 기준으로 잘못 지적한 비율과 오류를 놓친 비율을 평가한다. 정답 단어를 많이 통과시키는 설정만으로 발음 안내까지 좋아졌다고 판단하지 않는다.
7. 유력 모델은 더 충분히 학습하고 다른 seed에서도 확인한 뒤 결정한다. Python 인식 성능과 별도로 ONNX 변환 후 출력 일치, Unity 실행 지연·메모리를 측정한다. GPU 학습 가능성과 Unity 배포 가능성은 별도의 검증 항목이다.

W2V-BERT는 `SeamlessM4TFeatureExtractor`의 80개 mel bin·stride 2 전처리와 `Wav2Vec2BertForCTC`를 사용한다. HuBERT는 `HubertForCTC`, WavLM은 해당 CTC 클래스로 모델 어댑터를 분리하는 방향이다. 현재 `python/tools/child_finetune/train.py`의 Wav2Vec2 전용 학습 경로를 모델 이름만 바꾸어 그대로 쓰는 작업으로 보지 않는다. [W2V-BERT 전처리 설정](https://huggingface.co/facebook/w2v-bert-2.0/blob/main/preprocessor_config.json), [Transformers 문서](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert).

이번 작업은 조사 문서 추가다. 학습 실행, 사용 중인 모델 교체, Unity 패키지 변경은 수행하지 않았다.
