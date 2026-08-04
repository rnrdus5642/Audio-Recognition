"""Reference output for the static 40000-sample graph.

The window the app feeds the model is always exactly 40000 samples, so
the clip is fitted to that length FIRST and normalized over the result -
the same order the Unity side uses. Doing it the other way round would
compare two different inputs.
"""
import base64
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import onnxruntime as ort

sys.path.insert(0, r"C:\Users\user\Desktop\AudioProject")

from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes
from python.runtime.audio import load_audio_16k_mono
from python.runtime.recognizer.ko.asr import DEFAULT_MODEL, KoreanASRRecognizer

from pathlib import Path
from transformers import Wav2Vec2Processor

ROOT = Path(r"C:\Users\user\Desktop\AudioProject")
OUT = Path(__file__).with_name("audio_cases_static.json")
SAMPLES = 40000
PICK = ["mom_f", "dad_f", "puppy_m", "grandma_f"]

processor = Wav2Vec2Processor.from_pretrained(DEFAULT_MODEL)
session = ort.InferenceSession(
    str(ROOT / "shared" / "models" / "wav2vec2_ko_static.onnx"),
    providers=["CPUExecutionProvider"],
)

golden = json.loads(
    (ROOT / "python/tests/fixtures/golden_set.json").read_text(encoding="utf-8")
)
clips = {c["case_id"]: c for c in (golden["cases"] if isinstance(golden, dict) else golden)}


def fit(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= SAMPLES:
        return audio[:SAMPLES]
    return np.pad(audio, (0, SAMPLES - len(audio)))


def normalize(audio: np.ndarray) -> np.ndarray:
    return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)


cases = []
for case_id in PICK:
    clip = clips[case_id]
    raw = fit(load_audio_16k_mono(ROOT / clip["audio_path"]).astype(np.float32))
    values = normalize(raw).astype(np.float32)[None, :]

    logits = session.run(["logits"], {"input_values": values})[0]
    ids = [int(i) for i in np.argmax(logits[0], axis=-1)]
    text = KoreanASRRecognizer._sanitize(processor.batch_decode(np.array([ids]))[0])

    cases.append({
        "id": case_id,
        "target_text": clip["target_text"],
        # RAW, already fitted to 40000: Unity normalizes it itself.
        "audio_b64": base64.b64encode(raw.tobytes()).decode("ascii"),
        "expected_frames": int(logits.shape[1]),
        "expected_ids": ids,
        "expected_text": text,
        "expected_phonemes": hangul_to_ipa_phonemes(text),
    })
    print(f"{case_id}: {logits.shape[1]} frames -> {text!r} "
          f"{hangul_to_ipa_phonemes(text)}")

OUT.write_text(json.dumps({"cases": cases}, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, f"({OUT.stat().st_size / 1024:.0f} KB)")
