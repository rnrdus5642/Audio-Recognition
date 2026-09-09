# Korean data

일반 단어용 파일은 `Tools > Phoneme Matching > 초기 세팅`으로
`Assets/StreamingAssets/`에 설치합니다. 이미 있는 파일은 보존합니다.
악기놀이 발표에는 이 샘플을 복사할 필요가 없습니다.
`Tools > Phoneme Matching > 악기놀이 발표 설정`과 `MusicPresentation` API가
패키지 Resources의 최신 데이터를 읽습니다. 이 폴더의 사본을 수정해도 내장 발표 설정은 바뀌지 않습니다.

| 파일 | 내용 | 바꾸려면 |
|---|---|---|
| `ko_child_v1.json` | confusion matrix — 어떤 발음 실수를 얼마나 봐줄지 | 값만 고치면 됨 |
| `targets.json` | 정답 단어 18개의 IPA와 임계값 | 파이썬 빌드 필요 (아래) |
| `ko_child_v2.json` | 아동 모델용 confusion matrix | 모델별 혼동 데이터로 빌드 |
| `targets_child.json` | 아동 모델용 정답·임계값 | 파이썬 빌드 필요 |
| `targets_music_demo.json` | 악기놀이 12개 정답·9단계 참고 사본 | shared/music_presentation.json 수정 후 발표 빌드 |
| `wav2vec2_ko_vocab.json` | CTC 어휘 1205개 | 모델 바꿀 때만 |

## 정답 단어를 바꾸려면

`targets.json`은 프로젝트의 한국어 음운 규칙(mecab 불필요)을 적용해 만들어지고,
**정답 측과 사용자 측이 같은 IPA 표기 체계를
쓰는 것이 이 시스템이 성립하는 전제**입니다. 직접 편집하면 그 대칭이
깨집니다.

저장소(https://github.com/rnrdus5642/Audio-Recognition)에서:

```powershell
# shared/words.csv 편집 후
python -m python.build.build_targets      # -> shared/targets.json
python -m python.build.build_music_presentation # 발표 설정·Resources 동기화
```

## 음향 모델

ONNX 파일(1.18GB)은 용량 때문에 패키지에 들어있지 않습니다. 설정 메뉴에서 내려받거나,
`아동 음향 모델 파일에서 가져오기…`로 기존 파일을 가져오세요. 직접 기본 모델을 빌드하려면:

```powershell
python -m python.tools.export_onnx        # -> shared/models/wav2vec2_ko.onnx
```

만들어진 `.onnx` 를 `Assets/` 아래 아무 곳에나 두면 Sentis 가 임포트하고,
그 `ModelAsset` 을 `SentisPhonemeRecognizer` 에 넘기면 됩니다.
