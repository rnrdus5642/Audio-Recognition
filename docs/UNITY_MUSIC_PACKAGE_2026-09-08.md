# Unity 악기놀이 패키지 0.8.0 검증

> 이 문서는 당시 설정과 검증의 기록입니다. 현재 정답표·0.8.5 설정·오확정률은 [최신 적용 현황](MUSIC_DEMO_STATUS_2026-09-10.md)을 참고하세요.

2026-09-08, 로컬 배포 파일 `com.domicube.phoneme-matching-0.8.0.tgz`를 생성했다.
당시에는 Git 커밋·태그·Release를 게시하지 않았다.

- 파일 크기: 182,259 bytes
- SHA-256: `ba2ec931b749a4e172fd509866ef5f50f83c63e3189abf4b7c9d8f614d16c4b9`
- 내용: 152개 파일, Unity GUID 77개 중복 없음. ONNX·사용자 녹음은 포함하지 않는다.
- 설치와 기존 게임 연결: [패키지 사용 안내](../unity/Packages/com.domicube.phoneme-matching/Documentation~/music-presentation.md)

## 검증 결과

모든 Unity 검사는 소스 폴더 참조 대신 배포 `.tgz`를 설치한 독립 프로젝트에서 실행했다.
최종 파일과 동일한 SHA-256을 가진 사본으로 검사했으며, 압축 내부 파일 152개의 내용도
작업 소스와 바이트 단위로 대조했다.

| 환경 | 결과 | 범위 |
|---|---|---|
| Unity 2022.3.62f2 / Sentis 1.2.0-exp.2 | 40/40 통과 | Editor 검사 39개 + 실제 아동 ONNX 연결·GPU 준비 검사 1개 |
| Unity 6000.3.18f1 / Inference Engine 2.6.1 | 39/39 통과 | 설치·컴파일·패키지 Resources·9단계·기존 설정 보존 |
| 새 Unity 2022.3 프로젝트 / 추론 엔진 없음 | 39/39 통과 | 설치 메뉴 컴파일, Resources 로딩, LessonExample 임포트 포함 |
| .NET 코어 회귀 | 180/180 통과 | 기존 판정·세션·단어별 정책 |
| Python 발표 프로필 검사 | 25/25 통과 | 정답 사본 4곳·아동 행렬·어휘 동기화 포함 |
| git diff --check | 통과 | 공백 오류 없음 |

실제 모델 검사는 기존 `wav2vec2_ko_child.onnx`를 사용했다. 설정 메뉴 실행 전후
모델 수정 시각이 같고, 다운로드 없이 아동 모델·MusicDemo 설정이 테스트 창에 연결되는 것을 확인했다.
`MusicPresentationLesson.Prepare()`가 GPU 워밍업을 마치고 1단계 정답을 설정했으며,
마이크는 자동 시작하지 않았다. 같은 Listener에 9단계를 적용해 `두드려 / 때려`로 바뀌는 것도 확인했다.

설치 중 발견한 무엔진 컴파일 오류는 수정한 뒤 최종 배포본으로 다시 검사했다.
기존 `LessonExample`의 무엔진 참조 오류도 수정했다.

원시 검사 기록은 `.venv/unity-music-package-0908-logs/`의
`unity2022-model.xml`, `unity6-final.xml`, `noengine-final.xml`에 있다.
실제 모델 검사 스크립트는 해당 임시 소비 프로젝트의 `Assets/MusicSmoke/`에 있다.

## 정책과 검증 범위

발표 프로필 `music_demo_2026_09_07`의 수치는 변경하지 않았다.
이번 변경은 Unity 패키지 설치·업데이트·단계 연결을 위한 작업이다.
새 아동 발화나 실제 마이크 음성의 인식률을 다시 측정한 결과가 아니다.
기존 측정 결과와 남은 혼동은 [단어별 조정 기록](MUSIC_WORD_TUNING_2026-09-07.md)을 참조한다.

- 정답 JSON SHA-256: `c2659b42cd14119c6563c8cb438a55d1b0fe62cb728b89fd43a822e965abed4a`
- 아동 행렬 SHA-256: `b3b8c217a577d1dbf40673c47bdfb892d27b8a64593957c44072d34b75261bb6`
- 어휘 SHA-256: `7d6b76bdf58f1d48b6cf2341ac7708bed4647345a1dd4643cdc2a7247dd86344`

패키지 재생성:

```powershell
python -m python.build.build_music_presentation
python -m python.build.package_unity --output-dir C:/Users/user/Desktop
```

새 파일을 추가하는 개발자는 Unity `.meta`를 함께 생성해야 한다.
패키지 빌드는 빠진 `.meta`나 예기치 않은 음성·모델 파일이 있으면 중지한다.
