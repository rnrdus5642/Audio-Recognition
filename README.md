# Audio-Recognition

한국어 아동 발음 인식 프로젝트다. 현재 HuBERT 자모 모델의 상태와 실험 결과는
[작업 현황](01_Project/docs/CHILD_JAMO_STATUS.md)에서 확인한다.

Git 저장소의 기준 폴더는 **이 `Audio-Recognition` 폴더**다.
코드와 문서의 실제 작업 위치는 `01_Project`이며, 기존 파일 위치와 실행 경로를 유지한다.

| 폴더 | 내용 | Git에 보존하는 것 |
|---|---|---|
| `01_Project` | Python·Unity 코드, 문서, 로컬 환경·모델 | 코드·설정·실험 정리본·평가 결과 |
| `02_Datasets` | AIHub 원본 TAR, 인덱스, 추출 음성 | [폴더 안내](02_Datasets/README.md) |
| `03_Recordings` | 기존 아동·악기놀이 녹음 | [폴더 안내](03_Recordings/README.md) |
| `04_Models_and_Experiments` | 모델과 실험 기록을 여는 로컬 바로가기 | [보존 모델과 기록 안내](04_Models_and_Experiments/README.md) |
| `05_Run` | 실행·페이지 열기 | [사용법](05_Run/README.md), 상대 경로 실행 스크립트, URL 파일 |
| `99_Organization` | 이동·삭제·Git 정리의 로컬 검증과 백업 | [정리 기록 안내](99_Organization/README.md) |

## 사용

기존 `05_Run`의 바로가기를 사용하거나 이 폴더에서 다음 명령을 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\05_Run\start_microphone.ps1
powershell -ExecutionPolicy Bypass -File .\05_Run\start_phonetic_review.ps1
powershell -ExecutionPolicy Bypass -File .\05_Run\start_review.ps1
```

마이크는 <http://127.0.0.1:8776/>, 현재 청취는 <http://127.0.0.1:8778/>, 이전 청취 결과는
<http://127.0.0.1:8777/>에서 연다. 코드 개발·개별 도구 실행은 `01_Project`에서 한다.

## Git과 로컬 자료

기존 원격 저장소는 `https://github.com/rnrdus5642/Audio-Recognition.git`이다.
이 폴더 또는 `01_Project`에서 실행한 Git 명령은 같은 저장소를 사용한다.
대용량 모델·원본 데이터·녹음·청취 답변·가상환경·개인 PC 경로 설정·백업은 로컬에 보관한다.
Git clone만으로 이 자료가 복구되지는 않으므로 전체 작업 폴더의 별도 백업이 필요하다.

[Git 범위와 정리 기록](01_Project/docs/GIT_WORKSPACE_2026-10-03.md),
[프로젝트 상세 안내](01_Project/README.md),
[과거 실험 정리본 목록](01_Project/docs/EXPERIMENT_ARTIFACT_CLEANUP_2026-10-03.md).
