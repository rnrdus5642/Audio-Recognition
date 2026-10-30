# 오디오 정답 처리 패키지 배포 원본

현재 배포 원본은 `com.domicube.phoneme-matching/`이며 버전은 **0.8.9-preview.5**입니다.
Music·Dinosaurs·Flower에 배포한 패키지의 소스, README, CHANGELOG, 테스트를 함께 보관합니다.
`../unity/Packages/com.domicube.phoneme-matching`는 이전 실험 작업 공간이며 현재 배포 원본이 아닙니다.

## 설치와 실행

- Unity 2022.3 / Sentis 1.2.0-exp.2, Windows PC 및 PC 연결 VR을 대상으로 합니다.
- `Tools > 오디오 정답 처리 기능`의 초기 세팅·음성 테스트·정답 단어 설정을 사용합니다.
- 모델은 현재 Windows 사용자의 `문서/오디오정답처리/Models`에서 콘텐츠들이 공유합니다.
- 초기 세팅은 중앙 진행창을 표시하고 빌드 시작 씬을 구성합니다. 빌드 앱은 모델이 없으면 첫 실행에서 자동 설치합니다.
- 정상 설치 기록이 있는 앱 재실행은 파일 정보만 확인합니다. 초기 세팅 메뉴를 다시 누르면 전체 SHA-256을 검사합니다.
- 정답 목록은 각 Unity 프로젝트에 보관합니다. 추론 모델·녹음·학습 데이터는 이 패키지에 포함하지 않습니다.

자세한 동작은 [패키지 README](com.domicube.phoneme-matching/README.md), 변경 사항은 [CHANGELOG](com.domicube.phoneme-matching/CHANGELOG.md)를 확인하세요.

## 패키지 다시 만들기

저장소 루트에서 Python 3.9 이상으로 실행합니다. 패키지 생성에는 표준 라이브러리만 필요합니다.

```powershell
python 01_Project/python/build/package_unity.py --output-dir 04_Models_and_Experiments/package-build
```

기본 입력은 이 디렉터리의 배포 원본입니다. 다른 패키지를 명시적으로 만들 때만 `--package-dir <경로>`를 사용합니다.
생성된 `.tgz`를 Unity Package Manager의 **Add package from tarball**로 설치할 수 있습니다.

2026-10-30 배포물:

- 파일: `com.domicube.phoneme-matching-0.8.9-preview.5.tgz`
- 파일 수: 208개 / 크기: 2,477,194 bytes
- SHA-256: `c098e8f4acbffaa5e2123f733bd0703cd0c01b4edafa56c0968a379cd33d2041`
- 모델: [model-v3-child-jamo-sentis1 공개 Release](https://github.com/rnrdus5642/Audio-Recognition/releases/tag/model-v3-child-jamo-sentis1)

## 검증 범위

세 Unity 프로젝트에서 관련 Editor 테스트가 각각 17개 통과했습니다. Flower에서는 Windows 개발 빌드와 반복 빌드, 최초 다운로드·해시 검증·게임 진입, 취소·재시도·종료, 기존 설치 재사용을 실행 확인했습니다. 정상 설치 재확인 측정은 15.5ms이며 모델 메모리 로드나 GPU 준비 시간은 포함하지 않습니다.

Music·Dinosaurs의 전체 Player 빌드, 다른 PC, 일반 Release 빌드와 HMD·실제 마이크는 이 검증에 포함하지 않았습니다. Flower Player 검증은 이후 게임 변경 병합 이전 결과입니다. 빌드 결과는 성공이지만 Unity 보고서의 오류 카운터 1건 원인은 특정하지 못했습니다.

세 프로젝트의 `Docs/Integration/PlayerModelSetup-20261030`에 검증 자료가 있습니다. 배포 패키지와 문서가 포함된 커밋은 Music `442f678`, Dinosaurs `c6adf0c`, Flower `c7bdc76`입니다. 이 패키지는 배포된 2에포크 시점 모델을 사용하며, 이어서 진행하는 3에포크 학습 결과로 자동 교체하지 않습니다.
