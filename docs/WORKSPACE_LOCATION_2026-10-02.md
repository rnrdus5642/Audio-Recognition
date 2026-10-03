# 로컬 작업 폴더 정리

갱신: 2026-10-03. **이전 경로 연결 없이 바탕화면 폴더만 사용한다.**

실제 프로젝트: `C:\Users\user\Desktop\Audio-Recognition\01_Project`.
AIHub 원본·색인·추출 음성은 상위 `02_Datasets`, 관련 아동·악기놀이 녹음은
`03_Recordings`에 모았다. 모델·실험 결과의 상대 경로는 유지했으며 상위
`04_Models_and_Experiments`에 찾기 쉬운 바로가기를 만들었다.

10월 2일에는 열려 있던 작업창과 기존 절대 경로를 위해 정션·하드링크를 남겼다.
10월 3일에는 코드와 Python 가상환경의 경로를 현재 위치로 전환하고,
`Documents\GitHub\Audio-Recognition` 및 바탕화면의 옛 데이터·녹음 연결을 제거했다.
`05_Run`의 실행·열기 바로가기를 사용한다. 별도로 열어 두었던 옛 Codex 터미널은
폴더 잠금 해제를 위해 종료했고, 원본 대화 파일은 Codex 저장소에 보존되어 있다.

폴더 이동과 독립 실행 전환 시에는 모델·라벨·매니페스트·원본 전사·청취 답변의 내용을 변경하지 않았다.
폴더 이동 직후, 경로 지원 코드 수정 전에 모든 파일의 상대 경로·크기·수정 시간이
일치했고 Git 상태와 HEAD도 같았다. 중요 파일 39개의 해시 또는 NTFS 파일 식별자를
대조했다. 이 중 대형 원본 TAR은 전체 내용을 다시 해시하지 않고 파일 식별자·크기·
수정 시간을 확인했다.

이후 10월 3일에는 별도 요청에 따라 [이전 글자 단위 모델을 정리](LEGACY_MODEL_CLEANUP_2026-10-03.md)했다.
이전 아동 최종본 한 개만 비교용으로 남겼으며, 글자 단위 중간 가중치·다른 음절 모델·ONNX 사본과
악기놀이 검증 캐시를 삭제했다. `04_Models_and_Experiments/비교용 이전 아동 모델.lnk`로 보존본을 연다.
`05_Run`은 자모 마이크와 청취 페이지용이며, 구형 실시간 인식 7860 실행 항목은 제거했다.
이 삭제 기록은 `99_Organization/legacy-model-cleanup-20261003/`에 별도로 보존한다.

같은 날 후속 요청으로 [완료 실험과 과거 후보의 산출물 약 326.88GiB](EXPERIMENT_ARTIFACT_CLEANUP_2026-10-03.md)를
추가로 제거했다. 결과 문서·설정·평가 기록과 현재 모델·기반 인코더·비교용 아동 최종본은 보존했다.
`04_Models_and_Experiments`의 과거 후보·비교 실험 바로가기는 남겨 둔 **실험 기록**을 연다.
이 단계의 삭제 목록과 검증은 `99_Organization/experiment-cleanup-20261003/`에 있다.

`.workspace-location.json`은 과거 기록을 읽는 데 쓰는 이전/현재 경로 대응표다.
`python/workspace.py`의 `physical_path()`가 저장된 옛 경로를 현재 위치로 해석한다.
코드의 `ROOT`, 데이터·녹음 기본 위치와 실제 파일 접근은 모두 새 위치를 사용한다.
과거 CSV의 `AIHUB정리/audio`도 실제 `02_Datasets/audio`로 해석한다.
`historical_path()`는 변경하지 않은 학습 기록의 **식별 문자열 비교**에만 사용하며,
그 경로에 폴더가 존재할 필요는 없다. 모델·학습 목록·감사 기록을 다시 쓰지 않는다.
TAR 크기·수정 시간, 목록 해시, 인코더 해시와 라벨 계약 검증은 계속 수행한다.

검증 기록은 상위 `99_Organization`의 다음 파일에 있다.

- `inventory-before.json`, `inventory-after.json`: 원본 파일 목록 요약과 주요 파일 확인
- `runtime-path-check.json`: 기존 train/valid/test 계약 일치, 수정된 train의 출처 검증
- `execution-layout.json`, `move-log.jsonl`: 실제 이동 방법 및 경로
- `answer-backups/`: 이동 전 청취 답변 DB 백업

독립 실행 전환 기록과 변경 전 백업은
`99_Organization/standalone-20261003/`에 있다.
옛 경로가 없는 상태에서 서버 재시작, 청취 음성 70개, 기존 녹음의 실제 추론 재현,
학습 계약 일치 및 가상환경 실행 파일을 확인했다.

새 작업창은 위의 실제 프로젝트 폴더를 연다. 예전 보고서에 나타나는 옛 경로는
당시 기록이며, 현재 작업 위치로 사용하지 않는다.
