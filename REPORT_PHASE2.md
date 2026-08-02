# Phase 2 정확도 평가 보고서

> 한국어 단일 언어, 사전학습 모델 그대로 사용, CPU 추론.
> Fine-tuning 없이 confusion matrix + 임계값 튜닝까지만 진행.

> **⚠ 사후 업데이트 (Phase 2.5)**: 이 보고서의 수치는 **exact mode** 매칭 기준입니다.
> 이후 사용자 피드백("마이크 잡음이 음소로 잡혀서 매칭 실패")에 따라
> **substring (부분 문자열) 매칭**으로 기본 모드를 변경했습니다.
>
> | 모드 | Positives | Negatives 거절율 |
> |---|---|---|
> | Exact (당시 보고된 결과) | 25/36 = 69.4% | 137/144 = 95.1% |
> | **Substring (현재 기본값)** | 25/36 = 69.4% | 114/144 = 79.2% |
>
> 트레이드오프: TTS 골든셋은 잡음이 없어서 negatives만 떨어지지만,
> 실제 녹음에서는 잡음 무시 효과가 훨씬 크므로 substring을 기본화.
> Exact 모드는 `Matcher(matrix, mode="exact")`로 여전히 사용 가능.

> **⚠ 2차 업데이트**: 위 substring 수치(79.2%)는 `skip_cost = 0`, 즉 윈도우 밖
> 음소가 완전히 공짜이던 시점의 값이다. 이 설정은 점수를 발화 길이에 대해
> 단조 비감소로 만들어, 무관한 긴 발화가 통과하는 결함이 있었다
> (48음소 잡담 → "가요" 0.950 통과). `skip_cost = 0.15` 도입 후:
>
> | 지표 | skip 0 | skip 0.15 |
> |---|---|---|
> | Positives | 25/36 = 69.4% | 25/36 = 69.4% |
> | Negatives 거절율 | 114/144 = 79.2% | **132/144 = 91.7%** |
> | 무관 발화 거절 | 0/4 | **4/4** |
> | 잡음 ±2음소 내성 | 100% | 68% |
>
> 잡음 내성 하락이 대가다. 임계값 완화로 되살리려 하면 무관 발화가 다시
> 통과하므로(2차원 스윕 확인), 임계값은 건드리지 않았다.

## 결과 요약

| 지표 | 값 | 목표 | 평가 |
|---|---|---|---|
| Positive accuracy (자기 세그먼트 내 정답 매칭) | **25/36 = 69.4%** | 70%+ | ⚠️ 근접 |
| Negative rejection rate (다른 세그먼트 거절) | **137/144 = 95.1%** | 90%+ | ✅ 달성 |
| 평균 positive 점수 | 0.759 | | |
| 평균 negative 점수 | 0.455 | | |
| Positive ↔ Negative 점수 격차 | +0.303 | | 양호 |
| 전체 36 발화 추론 시간 (CPU) | 56.2초 | | |

## 평가 설정

| 항목 | 값 |
|---|---|
| ASR 모델 | `kresnik/wav2vec2-large-xlsr-korean` (300M params, FP32 CPU) |
| G2P | g2pkk → 자모 → IPA (빌드와 동일) |
| Confusion matrix | `ko_child_v1` (격음·경음·발달지연 패턴 + 종성 불파음 ↔ 초성 동가) |
| 정답 단어 셋 | 18개, 5 세그먼트 (가족/동물/음식/행동/학교) |
| 골든 오디오 | 18단어 × 2 voices (Edge-TTS: SunHi 여, InJoon 남) = 36 클립 |
| 평가 케이스 수 | Positives 36, Negatives 144 (각 클립을 4개의 다른 세그먼트 정답들과 매칭) |

## 실패 분석 (11개 positive 실패)

### Category A: ASR Catastrophic Failure (5건)
모델이 음향을 거의 인식 못한 경우. 매처로 회복 불가.

| 단어 | ASR 출력 | IPA | 점수 |
|---|---|---|---|
| 아빠 (남성) | "아" | `[a]` | 0.320 |
| 우유 (여) | "오" | `[o]` | 0.302 |
| 우유 (남) | "오" | `[o]` | 0.302 |
| 빵 (여) | "" | `[]` | 0.100 |
| 와요 (여) | "" | `[]` | 0.100 |

**원인**: ASR 모델이 짧은 1~2음절 발화에서 음향 일부만 캡처하거나 무음 처리. Wav2Vec2 large 모델이 KsponSpeech(긴 발화 위주)로 학습된 영향.

### Category B: Wrong Pick (5건)
ASR이 부분/오인식한 결과가 같은 세그먼트의 다른 정답과 더 가까움.

| 단어 | ASR 출력 | 매처 픽 | 점수 |
|---|---|---|---|
| 토끼 (여) | "히" `[h, i]` | butterfly (나비) | 0.550 |
| 토끼 (남) | "히" `[h, i]` | butterfly (나비) | 0.550 |
| 나비 (여) | "자기" `[tɕ, a, k, i]` | rabbit (토끼) | 0.625 |
| 바나나 (여) | "바" `[p, a]` | bread (빵) | 0.833 |
| 바나나 (남) | "바" `[p, a]` | bread (빵) | 0.833 |

**원인**: ASR 음향 오인식 결과가 의도와 다른 단어에 더 가까움. 특히 바나나→빵은 ASR이 첫 음절만 캡처했을 때 빵(3음소)이 바나나(6음소)보다 정확히 부합.

### Category C: Threshold Miss (1건)
정답을 픽했지만 점수가 임계값에 살짝 못 미침.

| 단어 | ASR 출력 | 점수 | 임계값 |
|---|---|---|---|
| 가요 (여) | "사" `[s, a]` | 0.550 | 0.70 |

**원인**: ASR이 단어를 완전히 잘못 인식 ("가요" → "사"). 매처는 정답 픽했으나 음소 일치도 낮음.

## False Accept 분석 (7건, 6 케이스)

다른 세그먼트의 정답으로 잘못 통과한 케이스:

| 케이스 | 사용자 IPA | 잘못 통과한 곳 |
|---|---|---|
| dad_f (아빠 → "앞") | `[a, p̚]` | 1건 |
| puppy_f (강아지 → "가아") | `[k, a, a]` | 1건 |
| puppy_m (강아지 → "간아지") | `[k, a, n, a, tɕ, i]` | 1건 |
| butterfly_f (나비 → "자기") | `[tɕ, a, k, i]` | 2건 |
| 추가 2건 (threshold 0.75→0.73 완화 영향) | | 2건 |

대부분 ASR 오인식 결과가 우연히 다른 단어와 일치. UX 원칙(False Reject 회피 우선)에 따라 임계값을 살짝 완화한 트레이드오프.

## 튜닝 이력

| 라운드 | Positives | Negatives | 주요 변경 |
|---|---|---|---|
| 1. 베이스라인 | 66.7% | 77.1% | 초기 confusion matrix + 임계값 (단어 길이별) |
| 2. Matrix 확장 | 66.7% | 96.5% | 종성 불파음 ↔ 초성 동가, l↔m·l↔n, 모음 a↔o 등 추가. 짧은 단어 임계값 상향 (0.70→0.85). 길이 페널티 도입 |
| 3. 길이 페널티 강화 | 66.7% | 96.5% | 사용자 음소 << 정답일 때 페널티 강화 (mult 0.5+ratio → 0.2+ratio) |
| 4. 3음소 임계값 완화 | **69.4%** | **95.1%** | 0.75 → 0.73 (아빠/앞 케이스 회수) |

## 핵심 인사이트

1. **ASR이 병목**: 11건 실패 중 10건이 ASR의 음향 인식 한계. 매처/임계값 튜닝으로는 회복 불가.
2. **Matcher는 잘 작동**: 정확한 ASR 출력 (24-25 케이스) 모두 정답 매칭. 잘못된 ASR 출력에서도 음향적으로 가까운 후보를 합리적으로 선택.
3. **Negative rejection 95%+ 달성**: 길이 페널티 + Confusion matrix + 단어별 적응형 임계값의 조합으로 거짓 통과를 효과적으로 차단.
4. **UX 트레이드오프**: False Reject < False Accept 원칙에 따라 3음소 임계값을 0.73으로 완화 → +1 positive, +2 false accept.

## TTS 골든셋의 한계

Edge-TTS는 성인 명료 발화를 생성합니다. 실제 타겟 사용자(아동, 발달지연)는 다음 특성이 있어 본 평가가 over-optimistic할 수 있음:

- 발화 시간 더 짧음 → ASR 추가 손실 가능성
- 종성 탈락 빈발 → 현재 confusion matrix가 일부 흡수
- /ㅅ/, /ㄹ/ 미숙 → matrix 페널티 0.3 적용됨
- 1-2음절 단어 비중 높음 → 가장 어려운 영역

실제 아동 발화 정확도는 본 평가보다 낮을 가능성이 큼.

## 다음 단계 권장

### 즉시 적용 가능
- **추가 ASR 모델 비교**: `Bingsu/wav2vec2-large-xls-r-300m-korean`, MMS 등을 같은 골든셋으로 평가해서 짧은 단어 정확도 더 나은 모델 탐색
- **VAD 후처리 강화**: 무음 / 너무 짧은 발화는 모델에 보내지 않고 "다시 말해줘" UX
- **단어 셋 설계 시 음향 분리**: 같은 세그먼트 내 단어들이 너무 비슷하지 않게 (예: 나비/토끼는 [n,a,p,i] vs [tʰ,o,k͈,i]로 음향이 가까움)

### Phase 5 후보 (별도 결정 필요)
- **Korean wav2vec2 + IPA fine-tuning**: KsponSpeech 텍스트를 g2pkk → IPA 라벨로 변환 후 CTC 학습. Phase 2의 catastrophic 케이스 대부분 회수 가능 추정.
- **아동 발화 데이터 수집**: AI Hub 아동 코퍼스 또는 자체 수집. Confusion matrix 가중치를 실데이터로 재학습.
- **이중 모델 앙상블**: 한국어 ASR + Allosaurus(IPA 직접 출력) 점수 가중 평균. 한쪽이 catastrophic fail해도 다른 쪽이 보완.

## 재현 방법

```powershell
# 셋업 (Phase 1)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-phase2.txt

# 빌드 (정답 텍스트 → IPA targets)
python -m python.build.build_targets

# 골든 오디오 생성 (Edge-TTS, ~1분)
python -m python.tools.generate_golden_audio

# 평가 (첫 회 ASR 추론 ~1분, 이후 캐시로 즉시)
python -m python.tools.evaluate
python -m python.tools.evaluate --refresh-cache  # 캐시 무시
```

모든 산출물:
- `shared/targets.json` — 빌드된 정답 IPA
- `python/tests/fixtures/golden_set.json` — 골든 케이스 메타
- `python/tests/fixtures/audio/*.wav` — TTS 오디오 (gitignored)
- `python/tests/fixtures/recognizer_cache.json` — ASR 캐시
- `python/tests/fixtures/eval_results.json` — 상세 평가 결과
