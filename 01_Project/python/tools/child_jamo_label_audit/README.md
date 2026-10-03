# 실제 학습 음성과 발음 정답의 대조

현재 청취: <http://127.0.0.1:8777/>. 한 페이지에서 20문항을 진행한다.
2026-09-29에 20문항과 10번 추가 원본 비교 분석을 완료했다. 대기 중인 청취 요청은 없다.
[결과와 다음 단계](../../../docs/CHILD_JAMO_DATASET_REVIEW_2026-09-29.md).
서버를 다시 열 때는 저장소 루트에서 다음 명령을 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\start_child_jamo_label_audit.ps1
```

서버는 Python 표준 라이브러리만 사용하며 모델을 로드하지 않는다.
기존 마이크 테스트(8776)와 이전 청취 답변 DB는 별도다.

## 데이터와 선정

[검토 계획·확인된 근거](../../../docs/CHILD_JAMO_DATASET_AUDIT_2026-09-28.md).
현재 모델 본학습에 사용한 **원본 150,253개**에서 10개는 연령·발화 유형별로
해시 순서로 추출하고, 10개는 짧은 단어의 발음 대조를 위해 추출했다.
선정 코드·시드·전체 행·제외 목록은 보존한다. 모델 출력과 오디오는 선정 후 확인했다.

작업 디렉터리: `.venv/child-jamo-web-20260916/dataset-audit-20260928/`.

| 파일 | 용도 |
|---|---|
| `selection.json` | 음성·모델 출력을 열기 전에 고정한 20문항 |
| `previous-reviews.json` | 기존 청취 답변의 읽기 전용 스냅샷 |
| `source-json/` | 해당 원본 JSON의 바이트 그대로인 사본 |
| `evidence.json` | 원본 TAR 위치, 해시, 학습 행, 발음 규칙 결과, 선택지 역할 |
| `audio/` | 정확한 학습 PCM과 자르기 전 PCM; FLOAT WAV로 손실 없이 저장 |
| `review.json` | 서버 문항 정의. 모델 출력은 포함하지 않음 |
| `baseline-predictions.json` | 표본 고정 후 현재 모델로 추론한 결과. 청취 페이지에서 접근 불가 |
| `answers.sqlite3` | **실제 사용자 답변**. 첫 인상과 수정 이력 보존 |
| `qa/*/qa-answers.sqlite3` | **자동 테스트 답변**. 실제 분석에 사용하지 않음 |

`prepare.py`는 재생성 시 같은 선정인지 검사하며, 실제 답변 DB가 생긴 뒤에는
문항을 다시 만들지 않는다. 질문·음성 변경 시 서버도 해당 DB의 사용을 거부한다.
`predict_frozen.py`는 기존 체크포인트를 읽어 추론만 한다. 모델·라벨을 수정하지 않는다.

## 사용자 응답 후 분석

```powershell
.venv-gpu/Scripts/python.exe -X utf8 python/tools/child_jamo_label_audit/analyze.py
```

실제 DB의 일관된 읽기 전용 스냅샷을 만들고, 출처·음성·답변 binding을 재검증한다.
`analysis-20260929.json`과 `docs/data/child-jamo-dataset-audit-review-20260929.json`을 만든다.
수정 전후의 답변 이력을 보존하며, 기존 질문 세트나 학습 라벨은 바꾸지 않는다.
`answer-snapshot-revision-*.json`, `analysis-revision-*.json`으로 응답 및 분석 사본도 보존한다.

사용자가 완료했다고 하면 즉시 `GET /api/export` 또는 실제 `answers.sqlite3`를
읽기 전용으로 확인한다. 자동 QA DB를 섞지 않는다. 한 문항의 마지막 revision을
사용하되 이전 답변과 표기를 보기 전 첫 인상은 보존한다.

1. 20개 응답 여부와 `binding`을 확인하고, 누락·모호함을 그대로 기록한다.
2. `evidence.json`의 `choice_roles`로 학습 표기/대조 표기 선택을 해석한다.
   대조 표기는 가상의 보기다. 선택했다는 이유만으로 정밀 발음 라벨로 확정하지 않는다.
3. 무작위 10개와 대조 10개를 분리해 집계한다. 20개를 합쳐 전체 오류율로 계산하지 않는다.
4. 청취 차이, 원본 전사→G2P 변화, 학습 구간/원본 차이, 원본 청취 응답을 교차한다.
   원본 전사와 G2P가 같고 원본에도 같은 소리 차이가 보고된 경우와,
   G2P·자르기를 아직 구분할 수 없는 경우를 구별한다.
5. 모델·학습 표기 일치/불일치를 추가로 교차한다. 모델이 라벨과 같게 출력해도
   실제 발음과 같은지는 별도다. 이것만으로 학습 원인의 인과적 기여를 단정하지 않는다.
6. 기존 72문항의 판단과 함께 **확인된 사실·불확실한 원인·추천할 다음 단계**를 보고한다.
   반복되는 차이가 확인되면 해당 대조와 화자 분리를 갖춘 검증용 발음 라벨을 먼저 만든다.
   청취 메모를 자동으로 학습 정답에 적용하거나 재학습을 시작하지 않는다.

## 검증

```powershell
.venv-gpu/Scripts/python.exe -X utf8 python/tools/child_jamo_label_audit/test_server.py
.venv/child-jamo-web-20260916/qa-venv/Scripts/python.exe -X utf8 python/tools/child_jamo_label_audit/qa_browser.py
```

서버 테스트 8개 통과. 브라우저 검증 8항목 통과: 비교 표기 숨김, 원본 재생,
대조 보기 표시, 모바일 폭, 새로고침 이어하기, 20개 완료·재접속, 답변 수정,
20개 음성 디코딩. QA는 분리된 임시 서버와 별도 답변 DB를 사용한다.
브라우저 재생은 자동 배속 검증이며 사람이 발음을 청취·판정했다는 뜻이 아니다.
