# Git 저장소 범위 정리

2026-10-03. Git 기준 폴더를 `Audio-Recognition/01_Project`에서 **상위 `Audio-Recognition`**으로 바꿨다.
소스·가상환경·모델·데이터·녹음의 실제 위치는 그대로다. `01_Project`에서도 같은 Git 저장소를 사용한다.

## 보존한 이력과 현재 작업

기존 원격 `https://github.com/rnrdus5642/Audio-Recognition.git`과 `main → origin/main` 연결을 유지한다.
기존 브랜치와 태그의 이력을 보존하며, `main`의 기존 커밋 뒤에 현재 작업 보존과 폴더 구조 전환 커밋을 추가했다.

전환 전 상태는 수정 78개, 삭제 72개, 미등록 작업 파일 242개였다. 복구용 가상환경의 미등록 파일은 별도였다.
현재 파일 553개를 `7d54244`에 보존했다. 기존 삭제 72개도 그대로 반영했으며 코드 내용을 되돌리지 않았다.
이 커밋의 부모는 전환 전 `366a046`이다. 원본 파일의 SHA256을 대조하고 소스 ZIP·diff·Git bundle도 남겼다.
코드·실험 문서의 최신 상태를 로컬 Git에 기록하는 작업이며, 기존 모델을 배포하거나 실험 결론을 바꾸는 작업은 아니다.

이 단계에서는 로컬 커밋으로 보존했다. 원격 게시 여부는 `git status -sb`와
`git log --oneline origin/main..main`으로 확인한다. 기존 `origin/main` 이후 미게시 커밋 2개도 유지한다.

## 관리 범위

| 위치 | Git에 포함 | 로컬에만 보존 |
|---|---|---|
| 최상위 | README·안내문·제외 규칙 | — |
| `01_Project` | 코드·설정·문서·`docs/data` 결과·테스트·실행 도구 | 모델·실험 캐시·가상환경·청취 답변·마이크 녹음·PC별 경로 대응표 |
| `02_Datasets` | README | 원본 TAR·라벨·색인·추출 음성 |
| `03_Recordings` | README | 실제 녹음과 개별 정리 자료 |
| `04_Models_and_Experiments` | README | PC별 `.lnk` 바로가기 |
| `05_Run` | README·상대 경로 PowerShell 스크립트·URL | PC별 `.lnk` 바로가기 |
| `99_Organization` | README | 원본 파일 백업·이동/삭제 목록·답변 사본·검증 기록 |

복구용 `.venv-gpu-repaired`와 `.venv-gpu-repaired311`도 `.venv-*/` 규칙으로 제외한다.
현재 모델·기반 인코더·비교용 최종본은 가중치 파일과 기존 모델 폴더 규칙으로 제외한다.
실험 폴더 `.venv` 안에 남긴 세부 기록도 로컬 자료다. Git에는 `docs`의 정리본과 결과 자료를 보존한다.
Git clone은 모델·음성·청취 답변·가상환경까지 포함한 전체 작업 폴더의 백업을 대체하지 않는다.

## 경로 사용

Git 상태 확인·커밋은 최상위 `Audio-Recognition`을 기준으로 한다.
Python·Unity의 소스 작업과 기존 명령은 계속 `01_Project`에서 수행한다.
`05_Run`의 새 스크립트는 자신의 위치에서 `01_Project`를 찾아 기존 실행 도구를 호출한다.
기존 바로가기도 계속 사용한다.

새 구조의 Unity Git 패키지 경로는 `/01_Project/unity/Packages/com.domicube.phoneme-matching`이다.
이전 태그와 과거 커밋의 경로는 `/unity/Packages/com.domicube.phoneme-matching`으로 유지된다.
새 URL은 이 구조의 커밋이 GitHub에 반영된 뒤 사용한다. Unity의 소스 폴더 찾기는 두 구조를 모두 인식한다.

경로 대응 Python 테스트 5개, 소스 폴더 탐색 테스트 9개를 통과했다. 후자는 실제 C# 경로 코드와
테스트를 .NET에서 실행하고 Unity 화면 API만 대체한 검증이며, Unity Editor 전체 빌드 검증은 아니다.
상대 경로 실행 스크립트 3개는 기존 서버를 정확히 인식했고, 마이크·청취 페이지는 계속 정상 응답했다.

## 로컬 복구 자료

`99_Organization/git-workspace-20261003/`에 다음을 보존한다.

- `history-before.bundle`: 전환 전 브랜치·태그와 커밋 이력
- `working-source-before.zip`, `working-changes.patch`: 원래 작업 파일과 변경 내용
- `before.json`, `index-before`, `config-before`: 해시·원래 상태·Git 설정
- `snapshot.json`: 현재 작업 보존 커밋과 내용 검증
- `git-directory-move.json`: 실제 Git 메타데이터 이동 기록
- `validation.json`: 최종 경로·이력·제외 규칙·실행 확인

이 자료는 Git에서 제외한다. 보존된 이동·정리 스크립트는 당시 실행 기록이며 다시 실행하지 않는다.
