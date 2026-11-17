# Unity 오디오 준비·추론 개선 — 2026-11-18

공통 패키지 `0.9.1-preview.4`, 적용 대상 CRC_Flower 보완-45. 모델과 v5 정답 정책은 유지한다. 기존 음악·공룡 설치본은 자동 변경되지 않는다.

- Sentis 1의 `CreatePreparedAsync`는 외부 모델 파일만 작업 스레드에서 읽는다. GPU worker 생성·워밍업은 Unity 스레드에서 수행하고 준비 완료된 인식기를 반환한다. 모델의 pinned CPU weights는 worker 수명까지 유지한다.
- 호출은 Unity 스레드에서 await하고, 준비 중 취소는 `CancellationToken`으로 전달한다. 진행 중 GPU 작업은 완료 후 해제한다. 반환된 인식기는 문항마다 재사용하고 소유 객체 종료 시 Dispose한다. 기존 동기 생성자/Warmup API는 호환용으로 남긴다.
- 추론 예약은 프레임당 128레이어/CPU 2ms 기준이다. 단일 레이어는 이 시간을 넘길 수 있다. 창 순서와 확인 횟수·취소 결과 배제는 유지한다.
- Flower는 준비 중 10초 답변 시계/녹음 구간을 시작하지 않는다. Pause·이동·수동 판정·입력 교체와 준비 실패 재시도를 처리한다.
- Flower 저장 코드는 답변 PCM 소유권을 저장 작업에 넘기고 JSON은 작업 스레드에서 인코딩한다. 기존 DB/PNG 및 저장 경로·schema는 그대로다.

Unity 2022.3.62f2 / RTX 5060 Ti / 빈 Editor Play / 생성 무음, 각 fps 3회: 60fps 완료 지연 약350→200ms, 30fps 약700→400ms. 500ms 입력 간격보다 빨리 완료되었지만 전체 게임/HMD 부하에서 보장되는 수치는 아니다. 첫 준비는 총4.91초/최대1296ms 프레임, 같은 프로세스 재측정은 총3.67초/최대84.8ms였다. 초기 업로드·메모리·드라이버 지연은 남을 수 있으며 캐시 조건이 달라 첫 준비 속도 비교를 주장하지 않는다. 실제 아동 음성 정답률은 재평가하지 않았다.

화분 검증: EditMode 59/59(준비 상태·18문항·PCM 저장·장치·경로), 공유 비동기 PlayMode 15/15, 실제 GPU 연속 실행·취소 후 재사용·준비 취소, Windows Player 스크립트 컴파일 통과. 전체 Player/HMD 미검증. 상세 기록은 CRC_Flower `Docs/Implementation/AudioImprove45/`에 보관한다.

배포물: `04_Models_and_Experiments/audio-improve-20261118/package/com.domicube.phoneme-matching-0.9.1-preview.4.tgz`. SHA-256 `03fd4881c773b961310a54349fe2ef93676af4ad62ed4321cbcd1241b76c2dc5`. 이 파일을 Flower Packages에 복사하고 manifest/lock의 상대 tgz 참조를 갱신했다. 모델 재다운로드는 불필요하며 기존 EXE는 재빌드해야 한다. 원격 Release 게시/다른 프로젝트 배포는 수행하지 않았다.
