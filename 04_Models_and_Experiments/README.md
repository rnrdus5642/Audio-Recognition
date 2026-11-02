# 현재 모델과 보존 실험

2026-11-02 기준. **현재 공개 배포 모델은 원문 철자 기반 HuBERT 69자모 모델의 최종 3에포크 선택본**입니다.
학습·최종 테스트는 완료됐으며, 과거 글자 모델이나 발음형 자모 모델과 구분합니다.
[전체 모델 보고서](../01_Project/docs/CHILD_ORTHOGRAPHIC_MODEL_2026-11-02.md)와 [공개 Release](https://github.com/rnrdus5642/Audio-Recognition/releases/tag/model-v4-child-jamo-epoch3-sentis1)를 참고하세요.

## 최종 모델

아래 경로는 이 폴더의 `orthographic-jamo-20261027/`를 기준으로 합니다. 파일과 가중치는 로컬 보관 자료입니다.

| 위치 | 내용 |
|---|---|
| `main/contract.json` | 확정 학습·평가 설정 |
| `main/progress.json` | 완료 상태 `complete`, 총 업데이트 29,430 |
| `main/selection.json` | `candidate_age3` 최종 선택과 개발 검증 비교 |
| `main/evaluations/final-test.json` | 최종 테스트 CER: 연령 동일 비중 평균 9.19% |
| `main/checkpoints/` | 재개 상태·선택 후보. 최종 파일 `ckpt-000029430-089b46152acd.pt` |
| `releases/epoch3-final-web-20261031/` | 최종 모델 웹 테스트 번들 |
| `releases/Unity-final-v5-20261031/` | Sentis 1.2 변환본·배포 패키지·동등성/설치 검증 |

학습 678.06시간·화자 ID 1,136개의 본학습 3에포크이며, 개발 검증 평균 CER은 6.05%입니다.
위 평균은 연령별 동일 비중입니다. CER은 게임 정답 통과율이나 빈 출력 비율과 다릅니다.
`candidate_age3`는 전체 3~6세에 공통으로 사용하는 모델입니다.

Unity 실행 파일은 문서/오디오정답처리/Models의
`hubert_ko_child_jamo_epoch3_sentis1.sentis`를 공유합니다. 모델 약 3.85GB는 Git clone이나 게임 빌드에
포함되지 않으며 공개 Release에서 설치합니다. 후처리는 `orthographic-word-context-v5`입니다.

## 과거 모델과 바로가기

아래 폴더 이름의 “현재”는 2026-10-26 정리 당시 이름입니다. 이름만 보고 최종 배포본으로 선택하지 마세요.

| 기존 폴더 | 보존 용도 |
|---|---|
| `01_현재_글자모델/` | 이전 글자 모델·학습 기록·목록 바로가기. 7860 웹의 비교용 |
| `02_보존_자모모델/` | 이전 발음형 HuBERT 자모 모델·기반 인코더·실험 기록 |
| `03_과거실험/` | 이전 후보 평가·인코더/Unity 실험·정리 문서 |

이전 글자 모델은 `01_Project/python/tools/child_finetune/runs/child/best/`,
이전 발음형 자모 모델은 `01_Project/.venv/hubert-child-20260914/main/`에 보관했습니다.
이 경로는 저장소 루트 기준입니다. 과거 글자 모델의 개발 CER 8.71%는 당시 별도 검증셋 결과입니다.

2026-10-26에는 필요한 WAV·라벨·기존 모델·결과를 보존하고 원본 음성 TAR 8개를 삭제했습니다.
해당 TAR을 직접 읽는 과거 재학습·재생 명령은 그대로 실행할 수 없습니다.
[데이터 보존 범위](../02_Datasets/README.md#용량-정리-결과),
[이전 실험 정리](../01_Project/docs/EXPERIMENT_ARTIFACT_CLEANUP_2026-10-03.md),
[실행·재개 기록](../05_Run/README.md)을 참고하세요.
