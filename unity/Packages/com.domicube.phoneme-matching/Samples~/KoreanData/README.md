# Korean data (ko_child_v1)

이 세 파일을 프로젝트의 `Assets/StreamingAssets/` 로 복사하세요. 런타임이
읽는 곳이 거기입니다.

| 파일 | 내용 | 바꾸려면 |
|---|---|---|
| `ko_child_v1.json` | confusion matrix — 어떤 발음 실수를 얼마나 봐줄지 | 값만 고치면 됨 |
| `targets.json` | 정답 단어 18개의 IPA와 임계값 | 파이썬 빌드 필요 (아래) |
| `wav2vec2_ko_vocab.json` | CTC 어휘 1205개 | 모델 바꿀 때만 |

## 정답 단어를 바꾸려면

`targets.json` 은 손으로 쓰는 파일이 아닙니다. 한국어 음운 규칙(g2pkk, mecab
필요)을 적용해 만들어지고, **정답 측과 사용자 측이 같은 IPA 표기 체계를
쓰는 것이 이 시스템이 성립하는 전제**입니다. 직접 편집하면 그 대칭이
깨집니다.

저장소(https://github.com/rnrdus5642/Audio-Recognition)에서:

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
