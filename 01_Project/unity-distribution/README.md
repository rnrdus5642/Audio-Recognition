# 오디오 정답 처리 패키지 배포 원본

현재 배포 원본은 `com.domicube.phoneme-matching/`이며 로컬 적용 버전은 **0.9.1-preview.2**입니다.
최종 3에포크 모델과 웹 후처리 v5를 포함하는 패키지의 소스, README, CHANGELOG, 테스트를 함께 보관합니다. 기존 Music·Dinosaurs·Flower 정답 설정을 읽을 수 있습니다.
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

2026-11-12 로컬 배포물은 `04_Models_and_Experiments/runtime-async-20261112/final-package/com.domicube.phoneme-matching-0.9.1-preview.2.tgz`입니다. SHA-256: `58e72d660665c9490348343a29d5892954b3c212e0167e618539d582e97faeff`. Music·Dinosaurs·Flower에 같은 파일을 적용했습니다. GitHub Release를 새로 게시한 것은 아닙니다. 이전 0.9.0 배포물은 기존 `orthographic-jamo-20261027/releases/Unity-final-v5-20261031`에 보존합니다.

0.9.1은 실시간 음성 추론을 프레임별로 분산하고 GPU 결과를 비동기로 가져옵니다. 시작 시 로딩·Warmup과 모델·후처리·정답 목록은 유지합니다. PNG 저장 기능은 변경하지 않습니다. 측정 조건과 한계는 [비동기 추론 검증](../docs/UNITY_ASR_ASYNC_2026-11-12.md)을 참고하세요. 아래 0.9.0 검증은 당시 기록이며 0.9.1 전체 Player/HMD 검증으로 해석하지 않습니다.

패키지를 추가한 뒤 **초기 세팅**을 실행하면 문서 공용 폴더에 최종 모델을 설치합니다. 모델 파일 이름을 버전별로 나누어 기존 2에포크 앱과 충돌하지 않습니다. 구형 v4 정답 JSON은 그대로 사용합니다. 프로젝트별 설치와 기존 Player 재빌드는 별도 작업입니다.

이번 변경은 Unity 2022.3.62f2에서 후처리·프로필 테스트 22개를 통과했습니다. 웹과 C#의 11,804개 사례·98,494회 검사에서 후보별 점수, 선택 범위, 거절 이유, 정답 확정이 일치합니다. 모델 변환 검사는 3·4·5·6세 각 1개 음성, 무음, 잡음의 6개 입력에서 PyTorch·ONNX·Sentis GPU의 최종 CTC 출력이 일치합니다. Sentis의 모든 프레임 로짓이 동일하다는 뜻은 아닙니다. 실제 마이크·HMD와 세 게임의 전체 Player 빌드는 이번 패키지 검증에 포함하지 않습니다.

생성한 `.tgz`를 별도 검증 프로젝트에 설치한 뒤 Editor 테스트 40개도 통과했습니다. 설치된 패키지의 실제 GPU 인식기로 검사한 8개 입력은 참조 텍스트와 일치했으며, `동생도 사과를 따요`와 `이건 사과인 것 같아요`의 저장된 2.5초 음성 창을 두 번 입력하면 v5에서 `사과`로 확정됩니다. 이는 저장 음성 창을 이용한 연결 검사이며 실제 마이크의 실시간 발화 검사가 아닙니다.

[공개 Release](https://github.com/rnrdus5642/Audio-Recognition/releases/tag/model-v4-child-jamo-epoch3-sentis1)에 모델과 `.tgz`를 게시했습니다. 인증 없는 패키지 다운로드의 해시 일치, 설치기의 실제 공개 모델 다운로드·연결·SHA-256 검증·문서 폴더 설치를 확인했습니다. 설치 후 `RequireInstalled` 확인은 6.0ms였으며 모델 메모리 로드·GPU 준비 시간은 제외됩니다. 이전 모델 파일은 보존합니다.

`.tgz`: 208개 파일 / 2,802,956 bytes / SHA-256 `a83c22a51126e20d576634388b083c47fce0694c548fc56603892deeb8f9239d`.

2026-10-30 이전 배포물:

- 파일: `com.domicube.phoneme-matching-0.8.9-preview.5.tgz`
- 파일 수: 208개 / 크기: 2,477,194 bytes
- SHA-256: `c098e8f4acbffaa5e2123f733bd0703cd0c01b4edafa56c0968a379cd33d2041`
- 모델: [model-v3-child-jamo-sentis1 공개 Release](https://github.com/rnrdus5642/Audio-Recognition/releases/tag/model-v3-child-jamo-sentis1)

## 이전 0.8.9-preview.5 검증 범위

세 Unity 프로젝트에서 관련 Editor 테스트가 각각 17개 통과했습니다. Flower에서는 Windows 개발 빌드와 반복 빌드, 최초 다운로드·해시 검증·게임 진입, 취소·재시도·종료, 기존 설치 재사용을 실행 확인했습니다. 정상 설치 재확인 측정은 15.5ms이며 모델 메모리 로드나 GPU 준비 시간은 포함하지 않습니다.

Music·Dinosaurs의 전체 Player 빌드, 다른 PC, 일반 Release 빌드와 HMD·실제 마이크는 이 검증에 포함하지 않았습니다. Flower Player 검증은 이후 게임 변경 병합 이전 결과입니다. 빌드 결과는 성공이지만 Unity 보고서의 오류 카운터 1건 원인은 특정하지 못했습니다.

세 프로젝트의 `Docs/Integration/PlayerModelSetup-20261030`에 이전 검증 자료가 있습니다. 당시 배포 패키지와 문서가 포함된 커밋은 Music `442f678`, Dinosaurs `c6adf0c`, Flower `c7bdc76`입니다. 이전 패키지는 2에포크 모델을 사용합니다. 새 0.9.0-preview.1과 Player 재빌드를 적용해야 최종 모델·v5 후처리로 바뀝니다.
