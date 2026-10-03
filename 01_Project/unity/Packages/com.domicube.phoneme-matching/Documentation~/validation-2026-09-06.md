# 설치·실행·웹 대조 검증 — 2026-09-06

Unity 2022/6의 Git 패키지 설치, Windows x64 Mono 실행, 웹의 초기 음성 버퍼 통일을
검증했다. 아래 결과는 현재 아동 모델과 명시된 입력·환경의 검사 결과이며,
모든 마이크·게임 장면·발음에 대한 무오류 보장이 아니다.

## 반영한 수정

| 항목 | 현재 동작 |
|---|---|
| 초기 세팅의 JSON 설치 | 없는 파일만 추가한다. 기존 사용자 정답·임계값·matrix·어휘는 보존하며, 별도 교체 메뉴는 확인 후 수행한다. |
| 여러 정답의 후보 선택 | 각 후보의 임계값 통과 여부를 먼저 본다. 합격 후보 중 최고 점수를 선택하고, 전부 불합격이면 표시용 최고 점수를 반환한다. |
| Editor 테스트 창 | 표준 모델 파일명에 따라 기본/아동 JSON을 연결한다. 다른 이름은 직접 선택하며, 파일 누락 시 다른 프로필로 대체하지 않는다. |
| 웹 초기 음성 버퍼 | 기본 2.5초(40,000샘플)를 처음부터 유지하고 부족한 앞부분을 무음으로 채운다. Unity `AudioWindowBuffer`와 같은 입력 정책이다. |

점수 계산식·단어별 임계값·허용 정답 범위·기본 연속 2회 확정·결과 JSON 계약은 유지했다.
웹의 무음 패딩은 경과 시간·판정 횟수에 추가하지 않으며, 실제 입력이 오면 바로 판정한다.
웹의 창 길이 변경은 다음 입력부터 음성·판정 이력을 새로 시작한다.
배포된 정적 Unity ONNX와 비교할 때는 웹도 기본 2.5초를 유지해야 한다.

`wav2vec2_ko.onnx`는 `ko_child_v1.json` / `targets.json`,
`wav2vec2_ko_child.onnx`는 `ko_child_v2.json` / `targets_child.json`과 연결한다.
공통 어휘는 `wav2vec2_ko_vocab.json`이다. 테스트 창의 자동 선택은 파일명 기준이며,
모델 가중치 자체를 분석하거나 게임의 Session 설정을 대신 변경하지 않는다.

## 설치와 실행 환경

세 가지 Unity 수정이 반영된 패키지 113개 파일을 별도 임시 Git 저장소에 복사했다.
검증 사본 커밋은 `006519da2917e77b211f3f3f5d234ad5bdd4e2d9`이며,
이는 배포 저장소의 커밋이나 릴리스 태그가 아니다.

새 소비 프로젝트에 `file:///…/snapshot.git?path=/unity/Packages/com.domicube.phoneme-matching#<전체 커밋 해시>`
형식으로 설치했다. `packages-lock.json`의 `source: git`과 커밋을 확인하고 설치 사본을 대조했다.
단순 로컬 폴더 참조가 아닌 Git 의존성 검사이지만, 수정본을 GitHub에 먼저 푸시해 HTTPS로
받은 검사는 아니다. FILE 프로토콜 형식은 [Unity 문서](https://docs.unity3d.com/2022.3/Documentation/Manual/upm-git.html#Git-FILE)를 따랐다.

Windows PC / RTX 5060 Ti에서 다음 구성을 검사했다.

| 검사 | 결과 |
|---|---|
| Python CPU 환경 전체 테스트 | 201/201 통과 |
| Python CUDA 환경 전체 테스트 | 201/201 통과 |
| 두 Python 환경 의존성 검사 | `pip check` 충돌 없음 |
| .NET C# 코어·회귀 테스트 | 46/46 통과 |
| Unity 2022.3.62f2 + Sentis 1.2.0-exp.2 | Editor 27/27, PlayMode 47/47 통과 |
| Unity 6000.3.18f1 + Inference Engine 2.6.1 | Editor 27/27, PlayMode 47/47 통과 |
| Unity 6, 추론 엔진 없는 기본 패키지 | Editor 27/27, PlayMode 47/47 통과. 실제 음향 추론 검사는 아님 |
| 두 Unity 버전의 Windows x64 Mono 개발 빌드 | 모두 성공, 빌드 오류 0 |
| 두 Windows 실행 파일의 GPU 추론 | 사과·엄마·아빠·강아지와 무음 검사 통과 |
| 두 Editor의 GPU 추론·테스트 창 준비 | 아동 사과 임계값 0.875, 준비 캐시 재사용/해제 등 통과 |
| 두 Editor의 연속 음성 입력 | 58개 합성 음성 / 199개 창, Python 고정 창 기준과 일치 |
| 웹 버퍼 수정 후 실제 모델 대조 | 같은 58개 음성 / 199개 창에서 앞선 Unity 실행 기록과 일치 |

Windows 실행 파일 검사는 동일한 16kHz 모노 PCM을 Python PyTorch와 Unity ONNX에 넣어
전사·IPA·점수·임계값·2회 확정·공통 피드백을 대조했다. 무음은 정답 처리되지 않았다.
실제 음향 추론 대상은 추가 학습한 아동 모델이다. 기본 모델은 데이터 로딩·채점 검증 범위였다.

## 초기 버퍼 통일 결과

수정 전 웹은 0.5초, 1초, 1.5초처럼 녹음된 길이만 모델에 넣었고,
Unity는 첫 입력부터 앞쪽 무음을 포함한 2.5초를 넣었다.
웹을 고정 창 방식으로 바꾼 뒤 실제 아동 모델의 웹 콜백에 저장된 동일 PCM을 재생했다.

웹 수정 때 Unity 실행 코드는 바꾸지 않았으므로 두 Editor를 다시 실행하지 않고,
앞선 실제 Unity 2022/6 GPU 실행 기록과 프레임별로 대조했다.
웹의 입력 샘플·전사·IPA·점수·연속 횟수·확정·피드백을 검사했고,
이전 3건의 정답 확정 시점 차이가 없어졌다.

| 음성 | 수정 전 웹 | 수정 후 웹 = Unity |
|---|---|---|
| 나비 (`butterfly_f`) | 2.5초 | 1.5초 |
| 책 (`book_f`) | 1.5초 | 2.5초 |
| 쳐요 (`hit_polite_f`) | 1.5초 | 2.0초 |

시간은 누적 입력 오디오 길이이며 PC의 실제 처리 지연이 아니다.
58개 모두 판정 결과·확정 시점이 일치했으며, 양쪽 모두 50개 확정/8개 미확정이었다.
미확정은 `go_f`, `book_m`, `hit_plain_f/m`, `put_polite_f/m`, `put_plain_f/m`이다.
버퍼 일치 검증이지 실제 아동 정확도 개선을 입증한 시험은 아니다.

후보 선택 변경은 저장된 아동 valid/test 인식 결과로도 대조했다.
정답 발화 241건·무관 발화 1,200건, 단독 정답 29개 및 복수 정답 4조합에서
추가 오확정·검출 손실·확정 시점 변화가 없었다. 복수 조합은 강아지/빵, 엄마/아빠,
엄마/아빠/할아버지, 엄마/아빠/할머니다. 모든 후보 조합을 검사한 것은 아니다.

## 남은 문제와 미검증 범위

- `LessonExample`을 추론 엔진보다 먼저 가져오면 `SentisPhonemeRecognizer` 참조에서
  CS0246 컴파일 오류가 난다. 기본 패키지는 정상이며 엔진 설치 후 같은 예제는 컴파일된다.
  이 샘플의 무엔진 경로 수정은 보류했다. 초기 세팅/엔진 설치를 먼저 수행한다.
- Unity 6의 PlayMode 밖 Editor GPU 추론 종료 시 Persistent 할당 경고 1개가 있다.
  생성 위치는 Inference Engine 2.6.1의 `ComputeTensorDataReaper` 정적 `ComputeBuffer`다.
  엔진 소스상 런타임 초기화에서 정리 콜백을 등록하므로 Editor 전용 경로에서 등록이
  빠지는 것으로 보인다(소스에 근거한 추론). 모델 전체 누수로 판단한 것은 아니며,
  검사한 Windows Player에는 동일 경고가 없었다.
- 당시 빌드는 Unity 6 경고 485개, Unity 2022 경고 62개였다. 추론 엔진 셰이더의
  정수 연산·부호 비교·변수 가림 등이 포함되며, 오류 0·실행 성공과 경고 없음은 구분한다.
- Windows IL2CPP 모듈이 설치되어 있지 않아 Mono만 검사했다. 실제 사용자 게임 씬,
  VR 렌더링과의 GPU 경합, 새 마이크 녹음, 장시간 청취, 모든 잡음·장치·설정은 미검증이다.
- 웹은 Python, Unity는 C#/Sentis다. 전체 런타임이 하나로 통합된 것은 아니며,
  웹의 모델 오류표 기반 설명 보류도 C# 원본 DTO와 다른 표시 정책으로 남아 있다.

## 회귀 검사와 기록

전체 저장소를 받은 환경에서는 저장소 루트에서 아래 명령을 실행한다.
Git 패키지만 설치한 Unity 프로젝트에는 Python/.NET 테스트 실행 프로젝트가 포함되지 않는다.

```powershell
python -m pytest python/tests -q
dotnet test csharp/PhonemeMatching.Tests
```

Unity 검사는 소비 프로젝트의 `testables`에 `com.domicube.phoneme-matching`을 포함해
Editor/PlayMode 테스트를 각각 실행했다. 테스트는 이 패키지의 `Tests/Editor`와
`Tests/Runtime` 소스를 사용한다. 기본 코어 테스트 통과만으로 Sentis나 실제 마이크가
검증된 것으로 취급하지 않는다.

이 개발 PC의 상세 로그·임시 소비 프로젝트·실행 파일은 저장소 기준
`.venv/check-0906-4f82/`, 웹 재검사 기록은 `.venv/buffer-parity-0906-67446b/`에 보관했다.
주요 기록은 `editmode.xml`, `playmode.xml`, `build-mono.json`, `player-mono.json`,
`rolling.json`, `python-cpu.xml`, `python-gpu.xml`, `web-unity-parity.json`이다.
이 대용량 임시 산출물은 Git에 포함하지 않았으며 다른 PC에서 자동으로 제공되지 않는다.
