# 로컬 정리 기록과 백업

Git에는 이 안내문만 포함한다. 이 폴더의 실제 백업·개별 파일 목록·청취 답변 사본·
머신별 상태와 작업 스크립트는 로컬에서 보존하며 다시 실행할 필요가 없다.

| 로컬 폴더 | 기록 |
|---|---|
| `model-analysis-20261102/` | 최종 모델의 실제 학습 목록·시간·화자·성능 재집계 |
| `model-report-20261102/` | HWP 원본 작성·9쪽 렌더링·누락 검사·문서 갱신 전 사본 |
| `standalone-20261003/` | 바탕화면 폴더만으로 독립 실행하도록 전환한 검증 |
| `legacy-model-cleanup-20261003/` | 비교용 아동 최종본을 제외한 글자 단위 모델 정리 |
| `experiment-cleanup-20261003/` | 완료 실험·과거 후보의 가중치와 캐시 정리 |
| `git-workspace-20261003/` | Git 범위 전환 전 이력 bundle, 원래 소스 ZIP·diff, 검증 |
| `folder-layout-20261026/` | 수집 참고자료 이동, 모델·녹음 바로가기 재분류, 변경 전 안내문·경로 설정과 검증 |
| `c-drive-usage-20261026/` | C 드라이브 용량 조사 결과. D 드라이브는 조사하지 않음 |
| `dataset-inventory-20261026/` | 전체 데이터의 연령·화자·시간 및 실제 학습 목록 집계 |
| `dataset-cleanup-20261026/` | 보관 음성 추출·원본 대조, U4 조사, 원본 음성 TAR 8개 삭제 및 보존 검증 |
| `aihub108-label-audit-20261026/` | 추가 라벨 ZIP 6개 복사·해시·전체 CRC 검사, 연령 분포·어휘 점검 |
| `orthographic-dataset-prep-20261026/` | 원문 라벨 자모 학습용 목록·화자 분할 준비, 구간 충돌·동일 무음 조사, 원문·목록 검증 |
| `aihub108-audio-organization-20261027/` | 원천 ZIP 9개 이동·보존, 3~6세 WAV 567,999개 추출·CRC·저장 후 해시 확인, 원문 라벨 연결·중복 검토·검증 |

`folder-layout-20261026/plan.json`과 `moves.jsonl`에 이전·현재 경로를 기록했다.
`inventory-before.json`에는 이동 파일과 주요 모델·학습 목록의 SHA-256·크기·수정 시간이,
`validation.json`에는 정리 후 검증 결과가 있다. `backups/`에는 수정 전 안내문과 경로 설정을,
`retired-shortcuts/`에는 중복이던 모델 바로가기를 보관한다.
위 폴더 이동 작업에서는 자료를 삭제하지 않았으며, 모델·학습 목록·직접 녹음의 실제 저장 위치는 유지했다.
되돌릴 때는 이동 기록과 백업을 함께 사용한다. 이동 스크립트는 기존 대상 덮어쓰기를 거부하므로 재실행하지 않는다.

`dataset-cleanup-20261026/`에서는 4세·U4 음성 24,049개를 추가로 추출하고,
기존 음성을 포함한 678,737개를 원본 TAR과 대조했다. `preparation.json`에 전체 음성 검증,
`preparation-validation.json`에 글자·자모 모델의 기존 목록 대조 결과를 기록했다.
그 뒤 사용자가 자모 모델·결과만 보존하고 원본 의존 기능은 유지하지 않도록 요청했다.
`deletion.json`에 원본 음성 TAR 8개, 570.01GB 삭제 결과를 기록했다.
`post-deletion-validation.json`에는 보관 WAV 전수 확인, 주요 파일 해시 일치와
모델·실험 기록·녹음·참고자료 46,843개 항목의 크기·수정 시간 보존 결과가 있다.
원본 라벨 TAR 8개는 유지했다. U4 871개는 정확한 연령이 확인되지 않았고 3세 대상과
관련될 가능성이 있어 129.64MB를 별도 보관했다. 근거는 `u4-investigation.json`에 있다.
`plan.json`과 `preparation*.json`은 삭제 전 기록이며 최종 상태는
[데이터셋 안내](../02_Datasets/README.md)의 용량 정리 결과와 삭제 후 검증 기록을 따른다.
삭제 및 원본 대조 스크립트는 재실행하지 않는다. 보관 WAV만으로 원본 TAR 전체를 복원할 수는 없다.

공유할 정리본은 `01_Project/docs/`에 보존한다.
[폴더 통합](../01_Project/docs/WORKSPACE_LOCATION_2026-10-02.md),
[실험 정리](../01_Project/docs/EXPERIMENT_ARTIFACT_CLEANUP_2026-10-03.md),
[Git 관리 범위](../01_Project/docs/GIT_WORKSPACE_2026-10-03.md).

최종 모델의 공유 문서는 [기술 보고서](../01_Project/docs/CHILD_ORTHOGRAPHIC_MODEL_2026-11-02.md)와
[HWP·도식](../01_Project/docs/reports/model-20261102/)에 보존한다.
이 폴더의 중간 HWPML·검토용 PDF·페이지 이미지·집계 도구는 로컬 작성·검증 자료다.
