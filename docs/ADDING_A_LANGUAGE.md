# 새 언어 추가 가이드

영어를 추가하는 예시로 설명합니다. 다른 언어도 동일한 흐름.

## 핵심 원칙

언어별 코드는 모두 `<lang>/` 폴더로 격리됩니다. 추가 시 **기존 파일 수정은 거의 없고**, 폴더 신설 + 두 곳의 REGISTRY에 한 줄씩만 등록하면 됩니다.

## 1단계: G2P 추가

### 1-1. 디렉토리 + 파일 만들기

```
python/build/g2p/en/
├── __init__.py          # EnglishG2P export
├── g2p.py               # 영어 G2P 구현
└── arpabet_ipa.py       # ARPAbet -> IPA 매핑 (또는 직접 IPA 출력 모듈 활용)
```

### 1-2. `g2p.py` 작성

```python
# python/build/g2p/en/g2p.py
from __future__ import annotations

from ..base import BaseG2P
from .arpabet_ipa import arpabet_to_ipa


class EnglishG2P(BaseG2P):
    def __init__(self) -> None:
        self._g2p = None

    @property
    def language(self) -> str:
        return "en"

    def _ensure(self):
        if self._g2p is None:
            from g2p_en import G2p  # pip install g2p-en
            self._g2p = G2p()

    def to_ipa(self, text: str) -> list[str]:
        self._ensure()
        arpabet_seq = self._g2p(text.lower().strip())
        return arpabet_to_ipa(arpabet_seq)
```

### 1-3. `arpabet_ipa.py` 매핑 작성

ARPAbet (CMU dict) → IPA 변환 테이블. 약 40개 음소.

### 1-4. `__init__.py`

```python
# python/build/g2p/en/__init__.py
from .g2p import EnglishG2P
__all__ = ["EnglishG2P"]
```

### 1-5. 레지스트리 등록

`python/build/g2p/__init__.py`:

```python
from .ko import KoreanG2P
from .en import EnglishG2P    # ← 한 줄 추가

G2P_REGISTRY = {
    "ko": KoreanG2P,
    "en": EnglishG2P,           # ← 한 줄 추가
}
```

## 2단계: 음성 인식기 추가

### 2-1. 디렉토리 + 파일

```
python/runtime/recognizer/en/
├── __init__.py
└── asr.py
```

### 2-2. `asr.py` 작성

```python
# python/runtime/recognizer/en/asr.py
from __future__ import annotations

import re
import numpy as np
from ..base import BaseRecognizer

DEFAULT_MODEL = "facebook/wav2vec2-base-960h"  # 또는 다른 영어 ASR

_ALPHA_RE = re.compile(r"[A-Za-z\s']+")


class EnglishASRRecognizer(BaseRecognizer):
    def __init__(self, model_name: str = DEFAULT_MODEL, device=None):
        self.model_name = model_name
        self._device = device
        self._processor = None
        self._model = None
        self._g2p = None

    @property
    def language(self) -> str:
        return "en"

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from python.build.g2p.en import EnglishG2P

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._g2p = EnglishG2P()

    def transcribe_hangul(self, audio_16k):
        # 메서드 이름은 backwards-compat용 (Hangul이 아니라 원본 텍스트)
        import torch
        self._load()
        if audio_16k.ndim != 1:
            raise ValueError(f"Expected 1-D mono audio, got {audio_16k.shape}")
        inputs = self._processor(
            audio_16k, sampling_rate=16000, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self._model(inputs.input_values.to(self._device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(pred_ids)[0]
        return self._sanitize(text)

    def recognize(self, audio_16k_mono):
        text = self.transcribe_hangul(audio_16k_mono)
        if not text.strip():
            return []
        return self._g2p.to_ipa(text)

    @staticmethod
    def _sanitize(text: str) -> str:
        return "".join(_ALPHA_RE.findall(text)).strip()
```

### 2-3. `__init__.py`

```python
# python/runtime/recognizer/en/__init__.py
from .asr import DEFAULT_MODEL, EnglishASRRecognizer
__all__ = ["EnglishASRRecognizer", "DEFAULT_MODEL"]
```

### 2-4. 레지스트리 등록

`python/runtime/recognizer/__init__.py`:

```python
from .ko import KoreanASRRecognizer
from .en import EnglishASRRecognizer   # ← 한 줄 추가

RECOGNIZER_REGISTRY = {
    "ko": KoreanASRRecognizer,
    "en": EnglishASRRecognizer,         # ← 한 줄 추가
}
```

## 3단계: Confusion Matrix 추가

`shared/confusion_matrices/en_child_v1.json` 작성. 한국어 매트릭스(`ko_child_v1.json`) 구조 그대로, 영어 음소 페어로 채움.

핵심 페어 예시:
- `θ|f` (TH-fronting, 발달지연 흔함): 0.3
- `r|w` (R-misarticulation): 0.3
- `ɹ|l`: 0.4
- 모음 인접: 0.3~0.5

(없어도 `AudioTester`가 기본 페널티 0.8로 동작은 함. 정확도가 떨어질 뿐.)

## 4단계: 정답 단어 추가

`shared/words.csv`에 영어 행 추가:

```csv
answer_id,text,language
apple_en,apple,en
banana_en,banana,en
mom_en,mom,en
...
```

`answer_id`는 언어를 통틀어 유일해야 합니다 — 한국어 `apple`(사과)과
영어 `apple`이 부딪히므로 접미사를 붙이세요.

## 5단계: 빌드 + 검증

```powershell
# 1. 새 G2P 의존성 설치 (영어의 경우)
pip install g2p-en

# 2. words.csv → targets.json 재빌드
python -m python.build.build_targets

# 3. 테스트 추가 (선택, 권장)
#    python/tests/test_g2p_en.py
#    한국어 테스트 구조 참고

# 4. 정확도 검증
python -m python.tools.web_test
#    → 0. 언어 선택 라디오에 자동으로 새 언어 옵션 표시
```

## 안 건드려도 되는 것들

다음은 **언어 추가 시 절대 수정하지 않음**:

- `python/runtime/audio.py` — 16kHz 변환 공통
- `python/runtime/matching/` — 매칭 엔진 (편집거리 등)
- `python/build/build_targets.py` — 빌드 파이프라인 디스패처
- `python/tools/` 의 모든 도구 — 자동으로 새 언어 인식
- `python/runtime/recognizer/base.py`, `python/build/g2p/base.py` — 인터페이스

## 새 언어 추가 시 변경 파일 요약

```
새로 추가:
  python/build/g2p/<lang>/__init__.py
  python/build/g2p/<lang>/g2p.py
  python/build/g2p/<lang>/(매핑 모듈).py
  python/runtime/recognizer/<lang>/__init__.py
  python/runtime/recognizer/<lang>/asr.py
  shared/confusion_matrices/<lang>_child_v1.json
  python/tests/test_g2p_<lang>.py  (선택)

수정:
  python/build/g2p/__init__.py        (2줄)
  python/runtime/recognizer/__init__.py (2줄)
  shared/words.csv                     (단어 추가)
  python/tools/web_test.py             (LANGUAGE_DISPLAY_NAMES에 이름 추가, 선택)
```

## 예상 작업 시간

| 언어 | 도구 가용성 | 예상 작업량 |
|---|---|---|
| 영어 | g2p-en, wav2vec2 영어 모델 다수 | 반나절 |
| 일본어 | pykakasi 또는 OpenJTalk + 일본어 wav2vec2 | 1~2일 |
| 중국어 | pypinyin + 성조 처리 + wav2vec2 중국어 | 2~3일 |
| 스페인어 | phonemizer (espeak), 다국어 wav2vec2 | 반나절 |
