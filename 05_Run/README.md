# 실행과 페이지 열기

이 PC에서는 기존 실행·열기 바로가기를 계속 사용한다.
Git에는 PC의 절대 경로가 들어 있는 `.lnk` 대신 아래 상대 경로 스크립트와 `.url`을 보존한다.

| 용도 | 실행 스크립트 | 주소 |
|---|---|---|
| 자모 마이크 테스트 | `start_microphone.ps1` | <http://127.0.0.1:8776/> |
| 현재 실제 소리 청취 15문항 | `start_phonetic_review.ps1` | <http://127.0.0.1:8778/> |
| 이전 청취 20문항·답변 | `start_review.ps1` | <http://127.0.0.1:8777/> |

상위 `Audio-Recognition`에서 실행하는 예:

```powershell
powershell -ExecutionPolicy Bypass -File .\05_Run\start_microphone.ps1
```

실행 창을 닫으면 해당 서버도 종료된다. 스크립트는 `01_Project`의 기존 실행 도구를 호출한다.
`01_Project/.venv-gpu` 환경과 로컬 모델·청취 자료가 준비되어 있어야 한다.
