# Korean data (ko_child_v1)

선택한 모델에 맞는 파일을 프로젝트의 `Assets/StreamingAssets/`로 복사합니다.
패키지의 모델 가져오기 메뉴는 필요한 누락 파일을 설치하고 기존 사용자 설정은 보존합니다.

| 파일 | 내용 | 바꾸려면 |
|---|---|---|
| `ko_child_v1.json` | confusion matrix — 어떤 발음 실수를 얼마나 봐줄지 | 값만 고치면 됨 |
| `targets.json` | 정답 단어 18개의 IPA와 임계값 | 파이썬 빌드 필요 (아래) |
| `wav2vec2_ko_vocab.json` | CTC 어휘 1205개 | 모델 바꿀 때만 |
| `ko_child_v2.json` | 기존 아동 모델의 판정 비용 | 새 모델에도 초기값으로 사용 |
| `targets_child.json` | 아동 모델용 정답 목록과 임계값 | 실제 게임 단어와 판정 기준 확인 |
| `hubert_ko_child_jamo_vocab.json` | 원문 자모 HuBERT의 69개 위치별 자모 | raw 16 kHz, NFC 조합 유지 |

원문 자모 모델의 웹 v4 모드는 `ko_child_v2.json`과 `hubert_ko_child_jamo_vocab.json`을
사용합니다. 정답 후보는 `TargetWords`에 직접 지정하며 `targets_child.json`은 필요 없습니다.
기본 임계값은 0.85입니다. 기존 모델/기존 판정 모드만 정답 JSON의 IPA와 임계값을 사용합니다.

## 기존 판정 모드의 정답 단어를 바꾸려면

`targets.json`은 빌드 도구가 한국어 음운 규칙과 IPA 매핑을 적용해 만들며,
**정답 측과 사용자 측이 같은 IPA 표기 체계를
쓰는 것이 이 시스템이 성립하는 전제**입니다. 직접 편집하면 그 대칭이
깨집니다.

저장소(https://github.com/rnrdus5642/Audio-Recognition)의 `01_Project` 폴더에서:

```powershell
# shared/words.csv 편집 후
python -m python.build.build_targets      # -> shared/targets.json
```

## 음향 모델

ONNX 파일(1.18GB)은 용량 때문에 패키지에 들어있지 않습니다. 같은 저장소에서:

```powershell
python -m python.tools.export_onnx        # -> shared/models/wav2vec2_ko.onnx
```

만들어진 `.onnx` 를 `Assets/` 아래 아무 곳에나 두면 Sentis 가 임포트하고,
그 `ModelAsset` 을 `SentisPhonemeRecognizer` 에 넘기면 됩니다.
