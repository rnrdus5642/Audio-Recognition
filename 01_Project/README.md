# 아동 음성 인식과 정답 판정

2026-11-02 기준. Git 루트는 상위 [Audio-Recognition](../README.md)이며 코드 작업 공간은 이 폴더입니다.
현재 공개 배포물은 **3~6세 HuBERT 정서법 자모 모델 / 본학습 3에포크 / 후처리 v5 / 패키지 0.9.0-preview.1**입니다.

## 현재 문서

- [모델 전체 분석과 파이프라인](docs/CHILD_ORTHOGRAPHIC_MODEL_2026-11-02.md)
- [한글 보고서 · 9쪽](docs/reports/model-20261102/model-report-ko.hwp)
- [최종 모델·패키지 공개 Release](https://github.com/rnrdus5642/Audio-Recognition/releases/tag/model-v4-child-jamo-epoch3-sentis1)
- [학습 데이터와 보관 범위](../02_Datasets/README.md)
- [모델·실험 보관 위치](../04_Models_and_Experiments/README.md)
- [로컬 실행 방법](../05_Run/README.md)
- [변경 기록](CHANGELOG.md), [이전 모델·실험 상태 기록](docs/CHILD_JAMO_STATUS.md)

## 입력과 판정

16kHz 모노 음성 → HuBERT XLarge → 69종 자모 CTC → 한글 조합 → 문장 내 후보 구간 탐색 →
규칙 기반 IPA·혼동 비용표의 유사도 → 확정 조건 검사 → 게임 이벤트 순서입니다.
정답 후보는 신경망 입력에 넣지 않고 한 번 받아쓴 결과를 공유합니다.

학습은 AIHub 108·540의 640,268발화, 678.06시간, 화자 ID 1,136개를 사용했습니다.
원문 철자를 유지하므로 실제 발음의 정밀 전사나 진단 결과로 사용하지 않습니다.
연령별 동일 비중 평균 CER은 개발 검증 6.05%, 별도 테스트 9.19%이며 게임 정답률과 다릅니다.
3세 테스트는 화자 ID 7개·약 54분으로 제한적입니다.

후처리는 `이건 사과인 것 같아요`에서 `사과` 구간을 비교합니다. 임의로 단어 내부를 잘라
`사과주스`를 `사과`로 만들지는 않습니다. 다만 ASR의 띄어쓰기나 중간 출력에 따른 오통과 가능성은 남습니다.
기본 입력은 2.5초 창·0.5초 간격이며, 유사도 0.85 이상인 같은 후보를 두 번 확인합니다.
모델이 잡음이나 다른 말 때문에 정답을 잘못 받아쓰는 문제는 후처리만으로 복원하지 못합니다.

## Unity와 실행 위치

현재 배포 패키지 작업 위치는 `unity-distribution/com.domicube.phoneme-matching/`입니다.
`unity/Packages/com.domicube.phoneme-matching/`는 별도의 이전 실험 작업 공간입니다.
새 PC 설치에는 위 공개 Release의 0.9.0-preview.1 `.tgz`를 사용하세요.

`Tools > 오디오 정답 처리 기능`에서 초기 세팅·음성 테스트·정답 단어 설정을 사용합니다.
모델은 문서/오디오정답처리/Models에 공용 설치합니다. Music·Dinosaurs·Flower는 같은 모델을 쓰며
프로젝트별 정답 목록은 각각 82·76·41개입니다. Windows 앱은 시작 씬에서 모델 설치를 확인합니다.
Unity 2022.3 / Sentis 1.2.0-exp.2 / GPUCompute 기준이며 CPU 자동 전환은 제공하지 않습니다.

이 PC의 최종 웹 검사는 8782, 이전 글자 모델 비교는 7860입니다. 웹 실행용 Python 환경·모델 번들은
로컬 자료입니다. 게임의 정답 단어 편집에는 Python이나 별도 IPA 신경망 모델이 필요하지 않습니다.

## 과거 기록

기존 README의 wav2vec2·1,205글자 어휘·이전 GPU/CPU 실측·초기 발음 피드백 API 설명은
[이전 구현 안내](README_LEGACY.md)에 보존했습니다. 그 수치와 설치 경로를 현재 모델에 적용하지 않습니다.
날짜가 붙은 실험 문서는 당시 결과를 유지합니다. 최종 모델의 수치·제약·검증 범위는 위 보고서를 따릅니다.
