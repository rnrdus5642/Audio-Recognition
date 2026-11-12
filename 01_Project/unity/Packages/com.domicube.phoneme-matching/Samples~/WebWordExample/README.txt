웹 v4 다중 후보 판정 — Unity 6 연결 예제

1. Inference Engine 2.6.1과 이 패키지를 설치합니다.
2. Tools > Phoneme Matching > 원문 자모 아동 모델 파일에서 가져오기…로
   hubert_ko_child_jamo.sentis와 필요한 JSON을 준비합니다. 이미 설치했다면 그대로 씁니다.
3. Package Manager의 Web word matching (Unity 6) 샘플을 Import합니다.
4. 빈 GameObject에 WebWordExample을 붙이고 Model에 해당 ModelAsset을 연결합니다.
5. 함께 추가된 PronunciationListener의 이벤트를 연결합니다.
   - OnTextRecognized(string): 원래 모델 출력 표시
   - OnFrameScored(string, float, int): 후보/유사도/확인 횟수 표시
   - OnConfirmed(string, float): 정답 처리 후 다음 게임 단계로 진행
6. WebWordExample.OnReady가 발생한 뒤 StartListening()을 호출합니다.
   Candidates의 여러 단어 중 하나를 말하면 확정 시 녹음이 멈춥니다.
7. 다음 단계는 example.Begin(new[] { "놓아", "놓아요", "놓을까?" });
   사용자가 취소하거나 쉬는 경우 example.StopListening();

기본값: 임계값 0.85, 동일 후보 통과 2회(중간 불확실 1회 허용), 전체 시간 제한 없음.
정답 목록은 실행 중 바꿀 수 있습니다. targets_child.json 재생성이 필요 없습니다.
원문 텍스트와 선택된 후보는 별개이며, LastFrame.WordDecision에서 후보별 점수를 봅니다.

앱이 이미 마이크를 열고 있다면 이 예제를 함께 사용하지 말고,
README의 PronunciationSession.ForOrthographicWord 방식으로 기존 오디오를 전달하세요.
모델 로딩/워밍업과 추론은 동기식입니다. 실제 PC/VR에서 프레임 지연을 확인하세요.
