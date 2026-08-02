"""Export the CTC vocabulary and decode reference vectors for Unity.

In Python the acoustic stage ends at `Wav2Vec2Processor.batch_decode`,
which quietly does three things: drop blanks, collapse repeats, and map
token ids to Hangul. Sentis gives us logits and nothing else, so that
decoder has to exist in C# - and it has to agree with Python exactly,
because every threshold in this project was tuned on Python's output.

This writes two files:

  * the vocabulary the decoder needs, next to the other StreamingAssets
    data (`targets.json`, the confusion matrix);
  * reference vectors pinning the decode, built from the token ids the
    shipping ONNX graph actually emits for the golden clips - not
    hand-written sequences, so the tests cover the repeat-and-blank
    patterns real audio produces.

The model emits `config.vocab_size` logits per frame, which is SMALLER
than `len(tokenizer.get_vocab())`: the tokenizer carries `<s>` and `</s>`
that the acoustic model can never produce. Exporting the tokenizer's view
would shift every id past the extras, so the vocabulary is truncated to
what the graph can actually output.

Usage:
    python -m python.tools.export_ctc_vocab
    python -m python.tools.export_ctc_vocab --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")

from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes
from python.runtime.audio import load_audio_16k_mono
from python.runtime.recognizer.ko.asr import DEFAULT_MODEL, KoreanASRRecognizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = PROJECT_ROOT / "shared" / "models" / "wav2vec2_ko.onnx"
GOLDEN_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "golden_set.json"
)
VOCAB_OUT = (
    PROJECT_ROOT / "unity" / "Assets" / "StreamingAssets"
    / "wav2vec2_ko_vocab.json"
)
VECTORS_OUT = (
    PROJECT_ROOT / "unity" / "Packages" / "com.domicube.phoneme-matching"
    / "Tests" / "Runtime" / "ctc_vectors.json"
)

# Decode paths that real audio reaches only by luck, so they are pinned
# by hand: an empty frame set, blank-only, a repeat that must collapse,
# a repeat split by a blank that must NOT collapse, and the word
# delimiter. Ids are filled in from the live vocabulary below.
SYNTHETIC = [
    ("empty", []),
    ("blank_only", ["<blank>", "<blank>"]),
    ("collapse_repeat", ["사", "사", "사", "과"]),
    ("blank_splits_repeat", ["사", "<blank>", "사"]),
    ("edge_blanks", ["<blank>", "사", "과", "<blank>"]),
    ("word_delimiter", ["사", "과", "<delim>", "엄", "마"]),
    ("unknown_token", ["사", "<unk>", "과"]),
]


def build_vocab(model_name: str) -> dict:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    config = Wav2Vec2ForCTC.from_pretrained(model_name).config
    tokenizer = processor.tokenizer
    fe = processor.feature_extractor

    size = int(config.vocab_size)
    by_id = {i: s for s, i in tokenizer.get_vocab().items()}
    missing = [i for i in range(size) if i not in by_id]
    if missing:
        raise SystemExit(f"vocab has holes at ids {missing[:5]}")

    if not fe.do_normalize:
        raise SystemExit(
            "feature extractor does not normalize; the C# port assumes it "
            "does - revisit SentisPhonemeRecognizer before shipping"
        )

    return {
        "_comment": (
            "python.tools.export_ctc_vocab 산출물. CTC 디코딩용 어휘. "
            "모델을 바꾸면 다시 생성할 것."
        ),
        "model": model_name,
        "vocab_size": size,
        "blank_id": int(tokenizer.pad_token_id),
        "unk_id": int(tokenizer.unk_token_id),
        "word_delimiter_id": int(tokenizer.word_delimiter_token_id),
        # Wav2Vec2FeatureExtractor.zero_mean_unit_var_norm
        "normalize": True,
        "normalize_epsilon": 1e-7,
        "sampling_rate": int(fe.sampling_rate),
        "tokens": [by_id[i] for i in range(size)],
    }


def _decode(processor, ids: list[int]) -> tuple[str, list[str]]:
    """Reference decode: exactly the runtime path, minus g2pkk.

    `KoreanASRRecognizer` applies phonological rules after this point,
    but the C# runtime deliberately does not (see JamoIpa), so the
    vectors stop where the port stops.
    """
    text = processor.batch_decode(np.array([ids], dtype=np.int64))[0]
    clean = KoreanASRRecognizer._sanitize(text)
    return clean, hangul_to_ipa_phonemes(clean)


def build_vectors(model_name: str, vocab: dict, limit: int) -> dict:
    import onnxruntime as ort
    from transformers import Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    cases: list[dict] = []

    named = {
        "<blank>": vocab["blank_id"],
        "<unk>": vocab["unk_id"],
        "<delim>": vocab["word_delimiter_id"],
    }
    token_id = {t: i for i, t in enumerate(vocab["tokens"])}
    for name, tokens in SYNTHETIC:
        ids = [named.get(t) if t in named else token_id[t] for t in tokens]
        text, phonemes = _decode(processor, ids)
        cases.append({
            "id": name, "source": "synthetic", "ids": ids,
            "text": text, "phonemes": phonemes,
        })

    if not ONNX_PATH.exists():
        raise SystemExit(
            f"{ONNX_PATH} 없음 - python -m python.tools.export_onnx 먼저"
        )

    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"])
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    clips = golden["cases"] if isinstance(golden, dict) else golden

    for clip in clips[:limit]:
        path = PROJECT_ROOT / clip["audio_path"]
        if not path.exists():
            continue

        audio = load_audio_16k_mono(path)
        values = processor(
            audio, sampling_rate=16000, return_tensors="np", padding=True
        ).input_values.astype(np.float32)
        logits = session.run(["logits"], {"input_values": values})[0]
        ids = [int(i) for i in np.argmax(logits[0], axis=-1)]

        text, phonemes = _decode(processor, ids)
        cases.append({
            "id": clip["case_id"], "source": "golden_onnx", "ids": ids,
            "text": text, "phonemes": phonemes,
        })

    return {
        "_comment": (
            "python.tools.export_ctc_vocab 산출물. C# CTC 디코더가 파이썬 "
            "batch_decode 와 같은 결과를 내는지 검증하는 기준. ids 는 "
            "실제 ONNX 그래프의 argmax 출력이다."
        ),
        "model": model_name,
        "vocab_size": vocab["vocab_size"],
        "cases": cases,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--vocab-output", type=Path, default=VOCAB_OUT)
    p.add_argument("--vectors-output", type=Path, default=VECTORS_OUT)
    p.add_argument("--limit", type=int, default=36,
                   help="벡터에 넣을 골든 클립 수")
    args = p.parse_args()

    vocab = build_vocab(args.model)
    args.vocab_output.parent.mkdir(parents=True, exist_ok=True)
    args.vocab_output.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OK: {args.vocab_output}  ({vocab['vocab_size']} tokens, "
          f"blank={vocab['blank_id']})")

    vectors = build_vectors(args.model, vocab, args.limit)
    args.vectors_output.parent.mkdir(parents=True, exist_ok=True)
    args.vectors_output.write_text(
        json.dumps(vectors, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n_real = sum(1 for c in vectors["cases"] if c["source"] == "golden_onnx")
    print(f"OK: {args.vectors_output}  ({len(vectors['cases'])} cases, "
          f"{n_real} from real audio)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
