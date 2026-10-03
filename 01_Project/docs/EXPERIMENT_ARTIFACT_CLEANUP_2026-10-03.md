# 완료 실험·과거 후보 모델 정리

정리일: 2026-10-03 KST. 사용자 요청에 따라 완료한 실험과 과거 후보의 대용량 산출물을 제거하고,
결과 문서·설정·개별 예측·평가 기록·선정 목록·소스를 보존했다.
이는 [글자 단위 모델 정리](LEGACY_MODEL_CLEANUP_2026-10-03.md) 이후의 별도 단계다.

## 삭제 범위

**총 121,853개 파일, 350,982,254,508바이트 = 326.88GiB**.
Windows 탐색기에서 GB로 표시하는 이진 단위 기준이다. 십진 단위로는 350.98GB다.
삭제 대상에는 외부 링크가 남는 하드링크 사본이 없어 고유 파일 크기 합계도 동일했다.

정리 후 바탕화면 `Audio-Recognition` 전체는 **약 678.77GiB**, 그 안의 `01_Project`는 **약 30.94GiB**다.
남은 용량의 대부분은 보존한 `02_Datasets`의 원본 TAR와 기존 추출 음성이다.
정확한 집계 시점·바이트 수는 아래 `completion.json`에 기록했다.

| 대상 | 제거한 산출물 | 크기(GiB) |
|---|---|---:|
| 초기 인코더 변환 비교 | ONNX·Sentis 모델, 외부 가중치와 시험용 바이너리 | 76.9325 |
| 초기 인코더 특징 캐시 | 79,189개 NPY 배열 | 37.1192 |
| Unity 비교 실험 | 두 프로젝트의 Library와 Windows 실행 빌드 | 2.7604 |
| 사용하지 않는 사전학습 인코더 10종 | HuBERT XLarge 기반 인코더를 제외한 가중치 | 20.9925 |
| 공개 자모·음소 후보 | 한국어 후보와 SLPLab 검증 폴더의 가중치 57개 | 74.1728 |
| 초기 인코더 학습 실험 | 본체 시험 학습·중단 실험·CTC 출력층 가중치 77개 | 12.8136 |
| 추가 학습 실험 5종 | 12개 모델의 최종 가중치와 학습 재개 파일 24개 | 100.1466 |
| 초기 비교용 음성 사본 | 원본 TAR에서 잘라 만든 WAV 7,199개 | 1.9401 |

추가 학습 5종은 `.venv/hubert-child-20260914/`의 다음 폴더다.

- `context-variation-20260927`
- `context-span-20260927`
- `robustness-extended-20260927`
- `robustness-pilot-20260927`
- `label-comparison-20260927`

이 폴더들의 계획·조건·선정 목록·학습 결과·평가 예측은 그대로 남아 있다.
특히 `context-variation`의 `selection.json`, 계획과 기초 평가 기록은 이후 전층 실험 코드가 참조하므로 보존했다.
아직 실행하지 않은 전층 실험의 계획과 소스도 보존했다.

## 계속 보존하는 파일

아래 경로는 `01_Project` 기준이다.

| 용도 | 위치 |
|---|---|
| 현재 HuBERT 자모 모델 | `.venv/hubert-child-20260914/main/best.pt` |
| 현재 모델의 학습 재개 상태 | `.venv/hubert-child-20260914/main/last.pt` |
| 현재 모델이 사용하는 기반 인코더와 설정 | `.venv/encoder-screen-0914/models/hubertko1b/` |
| 이전 글자 단위 아동 최종본 한 개 | `python/tools/child_finetune/runs/child/best/` |
| 현재 학습 목록·라벨·출처·TAR 색인 | `.venv/hubert-child-20260914/`의 목록·계약·catalog 및 학습 설정 |
| 청취 음성·답변·직접 녹음 | `.venv/child-jamo-web-20260916/` 및 상위 `03_Recordings/` |
| 원본 음성·라벨·기존 추출 자료 | 상위 `02_Datasets/` 전체 |

`python/tools/child_finetune/data/dev.csv`도 현행 학습 자료의 화자 제외 기준에 쓰여 보존했다.
초기 실험 음성 사본을 제외한 데이터·사용자 녹음은 제거하지 않았다.
7,199개 실험 음성의 원본 TAR·선정 정보·원본 파일명·SHA256·자른 음성의 SHA256과 추출 코드는 남겨 두었다.

## 결과 문서 찾아보기

| 실험 | 정리본 |
|---|---|
| 현재 모델과 다음 작업 | [현재 상태](CHILD_JAMO_STATUS.md) |
| 공개 한국어 자모·음소 후보 실측 | [후보 선정](WAV2VEC2_KOREAN_MODEL_SELECTION_2026-09-13.md), [전체 결과](WAV2VEC2_KOREAN_MODEL_RESULTS_2026-09-13.md) |
| SLPLab 음소 후보 | [음소 검증](WAV2VEC2_PHONE_AUDIO_VALIDATION_2026-09-13.md) |
| 초기 인코더·Windows·Unity 비교 | [실측 결과](PRETRAINED_ENCODER_WINDOWS_UNITY_RESULTS_2026-09-14.md) |
| 강건성 추가 학습 | [파일럿 결과](CHILD_JAMO_ROBUSTNESS_PILOT_2026-09-27.md), `docs/data/child-jamo-robustness-extended-20260927.json` |
| 청취 라벨 비교 학습 | [라벨 비교 결과](CHILD_JAMO_LABEL_COMPARISON_2026-09-27.md) |
| 무음·음량 변형·구간 비교 | [종합 판단](CHILD_JAMO_CONTEXT_CONCLUSION_2026-09-27.md), [변형 비교](CHILD_JAMO_CONTEXT_VARIATION_2026-09-27.md), [구간 비교](CHILD_JAMO_CONTEXT_SPAN_2026-09-27.md) |
| 데이터·청취 분석 | [데이터 검토 결과](CHILD_JAMO_DATASET_REVIEW_2026-09-29.md), [발음 청취 검토](CHILD_JAMO_PHONETIC_REVIEW_2026-09-29.md) |

`docs/data`의 원본 결과와 실험 폴더의 JSON·JSONL·설정·로그도 보존했다.
보고서의 이전 후보 추천과 체크포인트 경로는 **당시 판단과 실험의 기록**이다.
현재 사용할 모델은 위 보존 목록과 [현재 상태](CHILD_JAMO_STATUS.md)를 따른다.

## 다시 실험할 때

기존 수치와 개별 예측은 남은 기록으로 검토할 수 있다. 삭제한 모델의 새 추론과 학습 재개에는
공개 가중치 재다운로드 또는 시험 모델 재학습이 필요하다. 다운로드 출처·revision·검증 해시는 보존했다.
과거 `result.json`의 완료 표시가 가중치의 현재 존재를 뜻하지는 않는다.
기존 결과 폴더에서 재개하는 대신 새 실험 폴더에 가중치·캐시를 준비해야 한다.
특히 이전 구간 비교의 대조군 가중치는 삭제되어 그 실험을 그대로 재개할 수 없다.
현재 모델의 마이크 추론과 현재 `main/last.pt`에서의 학습 재개에 필요한 파일은 보존했다.

## 삭제와 검증 기록

보존 파일 20,358개의 크기·수정 시간·파일 식별자가 일치했고 20,357개 파일의 SHA256을 대조했다.
원본 데이터 654,718개와 녹음 4,525개의 목록·크기·수정 시간·파일 식별자도 정리 전후 동일했다.
대형 원본 TAR 전체의 내용을 다시 해시한 검증은 아니다.
두 보존 모델을 새 Python 프로세스에서 오프라인으로 로딩해 실제 음성 추론을 수행했다.
현재 HuBERT의 출력 텍스트·자모·토큰 ID는 저장된 녹음의 기존 결과와 일치했고,
기존 학습 데이터 계약 및 수정 학습 목록의 출처 검증도 통과했다.
8776 마이크와 8777·8778 청취 페이지가 정상 응답했고 청취 음성 70개의 응답 바이트도 파일과 일치했다.
답변은 이전 청취 20/20개, 현재 청취 0/15개로 보존되었다. Git HEAD는 변경하지 않았다.

상위 `99_Organization/experiment-cleanup-20261003/`에 다음을 둔다.

- `plan.json`: 정확한 범위·용량·보존 대상과 원본 데이터의 메타데이터 지문
- `deleted-file-inventory.jsonl.gz`, `delete-files.csv`: 삭제한 개별 파일 목록
- `removed-groups.jsonl`: 실제 삭제 완료 기록
- `protected-files-before.jsonl.gz`: 보존 파일의 식별자·크기·수정 시간·해시
- `backups/`: 이번에 갱신한 안내 문서와 바로가기의 변경 전 사본
- `validation.json`: 파일 보존·실제 모델 추론·학습 데이터 계약·청취 페이지 검증
- `completion.json`: 최종 용량과 Git 상태 확인

정리 스크립트는 이미 처리한 파일 목록을 위한 기록이다. 다시 실행할 필요가 없다.
