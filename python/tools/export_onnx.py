"""Export the Korean ASR model to ONNX and check it still behaves.

Unity runs inference through Sentis, which consumes ONNX. Two things can
go wrong and both are silent:

  * the exported graph produces different phonemes than PyTorch, so all
    the tuning done in Python stops describing what ships;
  * the graph contains operators Sentis cannot execute, which only shows
    up when someone tries to import it in the editor.

This checks both against the golden set before any Unity work depends on
the answer.

Usage:
    python -m python.tools.export_onnx
    python -m python.tools.export_onnx --opset 15 --limit 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        if hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")

from python.runtime.audio import load_audio_16k_mono
from python.runtime.recognizer.ko.asr import DEFAULT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "shared" / "models" / "wav2vec2_ko.onnx"
GOLDEN_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "golden_set.json"
)

# Operators Sentis 2.x lists as unsupported. Anything here in the graph
# means the model will not import, no matter how good the numbers are.
# Source: com.unity.ai.inference manual, "supported operators".
SENTIS_UNSUPPORTED = {
    "AffineGrid", "Attention", "BitShift", "CenterCropPad", "Col2Im",
    "ConcatFromSequence", "ConvInteger", "DeformConv", "DequantizeLinear",
    "Det", "DynamicQuantizeLinear", "EyeLike", "GlobalLpPool",
    "GroupNormalization", "GRU", "If", "ImageDecoder", "Loop",
    "LpNormalization", "LpPool", "MatMulInteger", "MaxRoiPool",
    "MaxUnpool", "MeanVarianceNormalization", "NegativeLogLikelihoodLoss",
    "Optional", "OptionalGetElement", "OptionalHasElement", "QLinearConv",
    "QLinearMatMul", "QuantizeLinear", "RegexFullMatch", "ReverseSequence",
    "RNN", "RotaryEmbedding", "Scan", "SequenceAt", "SequenceConstruct",
    "SequenceEmpty", "SequenceErase", "SequenceInsert", "SequenceLength",
    "SequenceMap", "SoftmaxCrossEntropyLoss", "SplitToSequence",
    "StringConcat", "StringNormalizer", "StringSplit", "TensorScatter",
    "TfIdfVectorizer", "Unique",
}


def export(model_name: str, out_path: Path, opset: int) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC

    print(f"Loading {model_name} ...")
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    # Two seconds of silence is enough to capture the graph; the time
    # axis is marked dynamic so any length works at runtime.
    dummy = torch.zeros(1, 32000, dtype=torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {out_path} (opset {opset}) ...")
    t0 = time.time()

    # The TorchScript tracer dies inside transformers 5.x mask building
    # (masking_utils.sdpa_mask indexes a shape that is scalar under
    # tracing), so take the dynamo path first and only fall back if it
    # is unavailable.
    batch = torch.export.Dim("batch", min=1, max=8)
    samples = torch.export.Dim("samples", min=4000, max=480000)

    with torch.no_grad():
        try:
            torch.onnx.export(
                model,
                (dummy,),
                str(out_path),
                input_names=["input_values"],
                output_names=["logits"],
                dynamic_shapes={"input_values": {0: batch, 1: samples}},
                opset_version=opset,
                dynamo=True,
                external_data=False,
            )
        except Exception as e:
            print(f"  dynamo export 실패 ({type(e).__name__}: {e})")
            print("  TorchScript 경로로 재시도 ...")
            torch.onnx.export(
                model,
                (dummy,),
                str(out_path),
                input_names=["input_values"],
                output_names=["logits"],
                dynamic_axes={
                    "input_values": {0: "batch", 1: "samples"},
                    "logits": {0: "batch", 1: "frames"},
                },
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )

    size_mb = out_path.stat().st_size / 1e6
    print(f"  {time.time() - t0:.1f}s, {size_mb:.0f} MB")


def inspect_ops(out_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(out_path), load_external_data=False)
    ops = Counter(node.op_type for node in model.graph.node)

    print()
    print("=" * 62)
    print("연산자 (상위 15개)")
    print("=" * 62)
    for op, n in ops.most_common(15):
        print(f"  {op:24s} {n}")
    print(f"  ... 총 {len(ops)}종 {sum(ops.values())}개")

    blocked = sorted(set(ops) & SENTIS_UNSUPPORTED)
    print()
    if blocked:
        print("Sentis 미지원 연산자 발견:")
        for op in blocked:
            print(f"  - {op} x{ops[op]}")
    else:
        print("Sentis 미지원 연산자 없음")
    return blocked


def verify(model_name: str, out_path: Path, limit: int) -> bool:
    """Compare ONNX and PyTorch phoneme output on real golden clips."""
    import onnxruntime as ort
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    torch_model = Wav2Vec2ForCTC.from_pretrained(model_name).eval()
    session = ort.InferenceSession(
        str(out_path), providers=["CPUExecutionProvider"])

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    cases = golden["cases"] if isinstance(golden, dict) else golden
    cases = cases[:limit]

    print()
    print("=" * 62)
    print(f"PyTorch vs ONNX 대조 ({len(cases)}개 클립)")
    print("=" * 62)
    print(f"{'클립':22s} {'일치':>4s} {'최대오차':>10s}  ONNX 한글")

    mismatches = 0
    max_diff_all = 0.0
    t_torch = t_onnx = 0.0

    for case in cases:
        path = PROJECT_ROOT / case["audio_path"]
        if not path.exists():
            continue

        audio = load_audio_16k_mono(path)
        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt", padding=True)
        values = inputs.input_values

        t0 = time.time()
        with torch.no_grad():
            logits_pt = torch_model(values).logits.numpy()
        t_torch += time.time() - t0

        t0 = time.time()
        logits_ox = session.run(
            ["logits"], {"input_values": values.numpy()})[0]
        t_onnx += time.time() - t0

        max_diff = float(np.max(np.abs(logits_pt - logits_ox)))
        max_diff_all = max(max_diff_all, max_diff)

        text_pt = processor.batch_decode(
            np.argmax(logits_pt, axis=-1))[0]
        text_ox = processor.batch_decode(
            np.argmax(logits_ox, axis=-1))[0]
        same = text_pt == text_ox
        # What actually matters is the phoneme list the matcher sees.
        same_ipa = (hangul_to_ipa_phonemes(text_pt)
                    == hangul_to_ipa_phonemes(text_ox))
        if not (same and same_ipa):
            mismatches += 1

        print(f"{Path(case['audio_path']).name:22s} "
              f"{'O' if same and same_ipa else 'X':>4s} "
              f"{max_diff:10.2e}  {text_ox!r}")

    print()
    print(f"불일치 {mismatches}/{len(cases)} | 로짓 최대오차 {max_diff_all:.2e}")
    print(f"추론 시간  PyTorch {t_torch:.1f}s  ONNX {t_onnx:.1f}s "
          f"({t_torch / max(t_onnx, 1e-9):.2f}x)")
    return mismatches == 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--limit", type=int, default=6,
                   help="대조에 쓸 골든 클립 수")
    p.add_argument("--skip-export", action="store_true",
                   help="이미 있는 ONNX로 검증만")
    args = p.parse_args()

    if not args.skip_export:
        export(args.model, args.output, args.opset)
    elif not args.output.exists():
        print(f"ERROR: {args.output} 없음", file=sys.stderr)
        return 1

    blocked = inspect_ops(args.output)
    ok = verify(args.model, args.output, args.limit)

    print()
    print("=" * 62)
    if ok and not blocked:
        print("판정: Sentis로 진행 가능")
        return 0
    if not ok:
        print("판정: ONNX 출력이 PyTorch와 다름 — 그대로 쓰면 안 됨")
    if blocked:
        print(f"판정: Sentis 미지원 연산자 {len(blocked)}종 — 임포트 실패 예상")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
