# 보존 모델과 실험 기록

이 폴더의 Windows 바로가기는 현재 PC에서 실제 모델·기록 위치를 연다.
바로가기 자체와 가중치는 Git에서 제외한다. 아래 경로는 상위 `Audio-Recognition` 기준이다.

| 용도 | 로컬 위치 |
|---|---|
| 현재 HuBERT 자모 모델과 재개 상태 | `01_Project/.venv/hubert-child-20260914/main/` |
| 현재 모델의 기반 인코더 | `01_Project/.venv/encoder-screen-0914/models/hubertko1b/` |
| 비교용 이전 아동 최종본 | `01_Project/python/tools/child_finetune/runs/child/best/` |
| 청취 결과와 직접 녹음 | `01_Project/.venv/child-jamo-web-20260916/` |

과거 후보와 완료 실험은 가중치·캐시를 제거하고 결과 문서·설정·평가 기록을 남겼다.
[정리본 목록과 보존 범위](../01_Project/docs/EXPERIMENT_ARTIFACT_CLEANUP_2026-10-03.md),
[현재 모델 상태](../01_Project/docs/CHILD_JAMO_STATUS.md)를 참고한다.
Git에는 `01_Project/docs`의 정리본·결과와 실험 소스를 보존한다.
`.venv` 안의 세부 기록과 가중치는 별도의 로컬 자료다.
