"""Give a CTC head a row for a syllable it never had, moving nothing.

The acoustic vocabulary is 1205 syllables and the curriculum asks for one
that is not among them. Widening the head is the only way the model can
ever spell it, and the whole difficulty is doing that without disturbing
the 1205 that already work.

The obvious route does disturb them. `tokenizer.add_tokens(...)` numbers
a new token at `len(tokenizer)`, and this tokenizer is longer than its
model: it carries `<s>` and `</s>` past the last logit column. So the new
syllable lands at 1207, and `resize_token_embeddings(len(tokenizer))`
then gives the head 1208 columns, two of which are freshly random
sentence markers the model can now emit into a transcript. Nothing warns
about it.

So the tokenizer is rebuilt instead: the old map verbatim, the new
syllables appended at the old width, and no stale added-token table
carried over. The markers get re-appended by `__init__` above the new
width, which is where they were relative to the head before. Blank, unk
and the word delimiter keep their ids, which is what lets the C# decoder,
the parity vectors and every tuned threshold stay valid.

The new row is initialised inert by default - zero weights, a low bias -
so it interferes as little as possible before it is trained. That is a
reduction, NOT a guarantee: the bias is a constant, and a frame whose
trained logits all sit lower still hands the argmax to the new row. What
the verification below reports is the observed count on reference clips,
not a proof.

The result is an UNTRAINED candidate. It can spell the syllable; it has
never heard one. Training it is a separate decision and a separate run.

    python -m python.tools.child_finetune.extend_vocab \
        --model runs/child/best --add 놔 --out runs/child-vocab/init
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python.tools import vocab_coverage as vc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = PROJECT_ROOT / "python" / "tests" / "fixtures" / "golden_set.json"

# Settings that describe how text becomes ids. They are copied across
# verbatim: a rebuilt tokenizer that lower-cases differently, or spells
# the delimiter differently, is a different tokenizer wearing the same
# vocabulary.
TOKENIZER_SETTINGS = (
    "bos_token", "eos_token", "unk_token", "pad_token",
    "word_delimiter_token", "replace_word_delimiter_char",
    "do_lower_case", "target_lang",
)


class ExtensionError(RuntimeError):
    """The extension cannot be done safely, so it is not done at all."""


@dataclass(frozen=True)
class Plan:
    """What will be added, and where it will land."""

    additions: list[str]
    old_size: int

    @property
    def new_size(self) -> int:
        return self.old_size + len(self.additions)

    @property
    def ids(self) -> dict[str, int]:
        return {s: self.old_size + i for i, s in enumerate(self.additions)}


def plan_additions(requested: list[str], vocab: vc.Vocabulary) -> Plan:
    """Which of the requested syllables are new, in a fixed order.

    Sorted by code point rather than by the order they were asked for, so
    two people who name the same syllables get the same ids whatever
    order they typed them in.
    """
    additions = sorted({s for s in requested if not vocab.emits(s)})
    return Plan(additions, vocab.size)


def check_hangul(requested: list[str]) -> None:
    """Refuse anything that is not a single Hangul syllable.

    A multi-character token would be added to the tokenizer's trie and
    start eating text at tokenisation time, which is a different change
    from widening the head by one row.
    """
    bad = [s for s in requested
           if len(s) != 1 or not vc.HANGUL_FIRST <= s <= vc.HANGUL_LAST]
    if bad:
        raise ExtensionError(f"한글 음절 하나가 아닙니다: {bad}")


def check_destination(source: Path, out: Path) -> None:
    """A candidate goes somewhere new, never on top of what it came from."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ExtensionError(
            f"출력 {out} 이 원본 {source} 과 겹칩니다. 원본은 건드리지 "
            "않습니다")
    if out.exists() and any(out.iterdir()):
        raise ExtensionError(
            f"출력 {out} 이 비어 있지 않습니다. 새 디렉터리를 주세요")


def check_bias(mode: str, bias: float) -> None:
    """A bias that is not a number poisons the row and the report.

    NaN propagates through the softmax into every logit, and it cannot be
    written as JSON that anything else will read back. Caught here, before
    a single file exists.
    """
    if mode == "inert" and not math.isfinite(bias):
        raise ExtensionError(f"새 행 bias 가 유한한 값이 아닙니다: {bias}")


def publish(staging: Path, out: Path) -> None:
    """Move a fully checked candidate to the name that was asked for."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.rmdir()          # checked empty by check_destination
    os.replace(staging, out)


def rebuild_processor(source: Path, out: Path, plan: Plan):
    """Write a processor whose ids are the old ones plus the new ones.

    Built from a vocabulary file rather than from the loaded tokenizer,
    so nothing in the source's added-token table can survive into the
    result. The sentence markers are not written either: the constructor
    re-appends them at the end of whatever it is given, which is exactly
    where they belong.
    """
    from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                              Wav2Vec2Processor)

    base = json.loads((source / "vocab.json").read_text(encoding="utf-8"))
    extended = {t: int(i) for t, i in base.items() if int(i) < plan.old_size}
    extended.update(plan.ids)

    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(
        json.dumps(extended, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n", encoding="utf-8")

    settings = json.loads(
        (source / "tokenizer_config.json").read_text(encoding="utf-8"))
    kwargs = {k: settings[k] for k in TOKENIZER_SETTINGS if k in settings}

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(out / "vocab.json"), **kwargs)
    processor = Wav2Vec2Processor(
        feature_extractor=Wav2Vec2FeatureExtractor.from_pretrained(source),
        tokenizer=tokenizer)
    processor.save_pretrained(out)
    return processor


def check_processor(processor, source_vocab: vc.Vocabulary, plan: Plan) -> dict:
    """Every old id where it was, every new one where it was promised."""
    tokenizer = processor.tokenizer
    if tokenizer.vocab_size != plan.new_size:
        raise ExtensionError(
            f"토크나이저 음향 폭 {tokenizer.vocab_size} != {plan.new_size}")

    live = tokenizer.get_vocab()
    moved = {t: (i, live.get(t)) for t, i in source_vocab.ids.items()
             if live.get(t) != i}
    if moved:
        sample = list(moved.items())[:5]
        raise ExtensionError(f"기존 id 가 움직였습니다: {sample}")

    for syllable, expected in plan.ids.items():
        if live.get(syllable) != expected:
            raise ExtensionError(
                f"{syllable} 이 {live.get(syllable)} 에 있습니다 "
                f"(기대 {expected})")

    named = {"blank_id": tokenizer.pad_token_id,
             "unk_id": tokenizer.unk_token_id,
             "delimiter_id": tokenizer.word_delimiter_token_id}
    was = {"blank_id": source_vocab.blank_id, "unk_id": source_vocab.unk_id,
           "delimiter_id": source_vocab.delimiter_id}
    if named != was:
        raise ExtensionError(f"특수 토큰이 움직였습니다: {was} -> {named}")

    beyond = {t: i for t, i in live.items() if i >= plan.new_size}
    inside = {t: i for t, i in live.items()
              if i < plan.new_size and t not in source_vocab.ids
              and t not in plan.ids}
    if inside:
        raise ExtensionError(
            f"음향 범위 안에 새 특수 토큰이 생겼습니다: {inside}")
    if len(live) != plan.new_size + len(beyond):
        raise ExtensionError(f"토크나이저 크기 {len(live)} 가 맞지 않습니다")

    return {"acoustic_width": tokenizer.vocab_size,
            "tokenizer_length": len(live),
            "beyond_acoustic_range": dict(
                sorted(beyond.items(), key=lambda kv: kv[1])),
            **named}


def extend_head(model, plan: Plan, *, mode: str = "inert",
                bias: float = -10.0, seed: int = 0):
    """Widen `lm_head`, carrying the old rows across untouched.

    `resize_token_embeddings` is not usable here - wav2vec2 has no input
    embeddings to resize. Replacing the head and rewriting
    `config.vocab_size` is what the library itself does when an adapter
    arrives with a different width (`Wav2Vec2PreTrainedModel.load_adapter`).
    """
    import torch
    from torch import nn

    old = model.lm_head
    if old.out_features != plan.old_size:
        raise ExtensionError(
            f"헤드 폭 {old.out_features} 이 어휘 {plan.old_size} 과 다릅니다")

    new = nn.Linear(old.in_features, plan.new_size,
                    dtype=old.weight.dtype, device=old.weight.device)
    with torch.no_grad():
        new.weight[:plan.old_size].copy_(old.weight)
        new.bias[:plan.old_size].copy_(old.bias)
        if mode == "inert":
            new.weight[plan.old_size:].zero_()
            new.bias[plan.old_size:].fill_(bias)
        else:
            generator = torch.Generator().manual_seed(seed)
            rows = torch.empty(
                len(plan.additions), old.in_features).normal_(
                    0.0, model.config.initializer_range, generator=generator)
            new.weight[plan.old_size:].copy_(rows.to(new.weight.dtype))
            new.bias[plan.old_size:].zero_()

    if not (torch.isfinite(new.weight[plan.old_size:]).all()
            and torch.isfinite(new.bias[plan.old_size:]).all()):
        raise ExtensionError("새 행에 유한하지 않은 값이 들어갔습니다")

    model.lm_head = new
    model.config.vocab_size = plan.new_size
    return model


def check_weights(before: dict, candidate, plan: Plan) -> dict:
    """Nothing outside the new rows may have changed by a single bit."""
    import torch

    after = candidate.state_dict()
    if set(before) != set(after):
        raise ExtensionError(
            f"파라미터 목록이 다릅니다: {sorted(set(before) ^ set(after))[:5]}")

    for key, tensor in before.items():
        other = after[key]
        if key.startswith("lm_head."):
            other = other[:plan.old_size]
        if not torch.equal(tensor, other):
            raise ExtensionError(f"{key} 가 바뀌었습니다")

    head = after["lm_head.weight"]
    return {"parameters_compared": len(before),
            "head_shape": list(head.shape),
            "rows_carried_over": plan.old_size,
            "new_row_bias": [float(b) for b
                             in after["lm_head.bias"][plan.old_size:]]}


def reference_clips(additions: list[str], limit: int) -> list[dict]:
    """A small, fixed set of the synthetic golden clips.

    Clips for the words that motivated the change come first so a short
    run still looks at them; the rest fill up in golden-set order. These
    are text-to-speech, not children, and they are here to show the graph
    did not move - not to say anything about how well anything is heard.
    """
    if not GOLDEN.exists() or limit <= 0:
        return []
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data

    wanted = [c for c in cases
              if any(s in c.get("target_text", "") for s in additions)]
    rest = [c for c in cases if c not in wanted]
    chosen = (wanted + rest)[:limit]
    return [c for c in chosen
            if (PROJECT_ROOT / c["audio_path"]).exists()]


def reference_pass(model, processor, clips: list[dict]) -> list[dict]:
    """Logits and text for each clip, taken before anything is changed.

    Run first and kept, rather than re-running the source model after the
    surgery: the weights are 1.27GB and there is no reason for two copies
    of them to be resident at once.
    """
    import numpy as np
    import torch

    from python.runtime.audio import load_audio_16k_mono

    model.eval()
    out = []
    for clip in clips:
        audio = load_audio_16k_mono(PROJECT_ROOT / clip["audio_path"])
        values = processor(audio, sampling_rate=16000, return_tensors="pt",
                           padding=True).input_values
        with torch.no_grad():
            logits = model(values).logits
        out.append({
            "clip": clip["case_id"], "target": clip.get("target_text", ""),
            "values": values, "logits": logits,
            "text": processor.batch_decode(np.asarray(logits.argmax(-1)))[0],
        })
    return out


def compare_on_clips(reference: list[dict], candidate, processor,
                     plan: Plan) -> dict:
    """Put the same audio through the widened head and report the drift.

    Two numbers matter. Whether the transcript changed at all, and how
    often the untrained row won a frame. The second is the observed
    answer to "does a low bias keep it out of the way" - a measurement on
    these clips, not a property that holds by construction. A frame whose
    trained logits all sit below the new row's constant bias would still
    go to the new row.
    """
    import numpy as np
    import torch

    candidate.eval()
    cases, changed, new_frames, worst = [], 0, 0, 0.0

    for item in reference:
        with torch.no_grad():
            after = candidate(item["values"]).logits

        drift = float(torch.max(torch.abs(
            item["logits"] - after[..., :plan.old_size])))
        worst = max(worst, drift)
        picked = after.argmax(-1)
        frames = int((picked >= plan.old_size).sum())
        new_frames += frames
        text = processor.batch_decode(np.asarray(picked))[0]
        if text != item["text"]:
            changed += 1
        cases.append({"clip": item["clip"], "target": item["target"],
                      "source_text": item["text"], "candidate_text": text,
                      "new_token_frames": frames,
                      "max_old_column_diff": drift})

    return {"clips": len(reference), "transcripts_changed": changed,
            "new_token_frames": new_frames,
            "max_old_column_diff": worst, "cases": cases}


def provenance(source: Path, plan: Plan, tokenizer_check: dict,
               weight_check: dict, audio_check: dict | None,
               init: dict, config_ids: dict) -> dict:
    import torch
    import transformers

    return {
        "_comment": ("python.tools.child_finetune.extend_vocab 산출물. "
                     "학습되지 않은 실험용 후보이며 배포 모델이 아니다."),
        "created": date.today().isoformat(),
        "source_model": str(source),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "acoustic_width": {"before": plan.old_size, "after": plan.new_size},
        "added": plan.ids,
        "new_row_init": init,
        "config_token_ids": config_ids,
        "tokenizer": tokenizer_check,
        "weights": weight_check,
        "reference_audio": audio_check,
        "status": "UNTRAINED",
        "gate": ("새 행은 오디오를 본 적이 없다. 학습 · 음향 검증 · ONNX "
                 "재출력 · 어휘/벡터 재생성 전에는 배포 경로에 넣지 않는다."),
    }


def extend(source: Path, out: Path, requested: list[str], *,
           init: str = "inert", bias: float = -10.0, seed: int = 0,
           verify_clips: int = 6,
           keep_config_token_ids: bool = False) -> dict:
    """Produce a widened, untrained candidate and prove it moved nothing."""
    source, out = Path(source), Path(out)
    check_hangul(requested)
    check_bias(init, bias)
    vocab = vc.load_checkpoint(source)
    plan = plan_additions(requested, vocab)
    if not plan.additions:
        return {"added": {}, "acoustic_width": {"before": plan.old_size,
                                                "after": plan.old_size},
                "status": "NOTHING_TO_ADD", "source_model": str(source)}

    check_destination(source, out)

    # Everything is built and checked under a name nobody asked for, and
    # only takes the requested one once it has passed. A checkpoint that
    # failed validation must not be left sitting at the final path
    # looking loadable, and must not block the retry either.
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{out.name}.staging-", dir=out.parent))
    try:
        report = _build_candidate(
            source, staging, vocab, plan, init, bias, seed, verify_clips,
            keep_config_token_ids)
        # Inside the boundary: a rename that fails would otherwise strand
        # 1.27GB of staging per attempt. `publish` only ever removes an
        # empty `out`, so a directory another writer has filled in the
        # meantime stops the move instead of being deleted.
        publish(staging, out)
    except BaseException:
        # Only ever the directory this call created.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _build_candidate(source: Path, out: Path, vocab: vc.Vocabulary,
                     plan: Plan, init: str, bias: float, seed: int,
                     verify_clips: int, keep_config_token_ids: bool) -> dict:
    """Write and check the whole candidate inside one directory."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model = Wav2Vec2ForCTC.from_pretrained(source)
    # Captured before the surgery. Widening replaces `lm_head` with a new
    # module rather than writing into the old tensors, so these stay the
    # source's own weights and can be compared against what lands on disk.
    pristine = dict(model.state_dict())

    clips = reference_clips(plan.additions, verify_clips)
    reference = reference_pass(
        model, Wav2Vec2Processor.from_pretrained(source), clips) \
        if clips else []

    rebuild_processor(source, out, plan)
    processor = Wav2Vec2Processor.from_pretrained(out)
    tokenizer_check = check_processor(processor, vocab, plan)

    extend_head(model, plan, mode=init, bias=bias, seed=seed)
    config_ids = {"pad_token_id": model.config.pad_token_id,
                  "bos_token_id_before": model.config.bos_token_id,
                  "eos_token_id_before": model.config.eos_token_id,
                  "tokenizer_bos_id": processor.tokenizer.bos_token_id,
                  "tokenizer_eos_id": processor.tokenizer.eos_token_id}
    if not keep_config_token_ids:
        # The source config names ids 1 and 2, which are Hangul syllables
        # - vestigial values that predate this checkpoint and never
        # agreed with the tokenizer. The honest replacement is None, not
        # the tokenizer's marker ids: this field is about the model's own
        # vocabulary, a CTC head has no column for a sentence marker, and
        # transformers rejects an id past the width. The markers are
        # still on the tokenizer, and the ids are recorded above.
        model.config.bos_token_id = None
        model.config.eos_token_id = None
    config_ids.update(bos_token_id=model.config.bos_token_id,
                      eos_token_id=model.config.eos_token_id)
    model.save_pretrained(out)

    candidate = Wav2Vec2ForCTC.from_pretrained(out)
    if candidate.config.vocab_size != plan.new_size:
        raise ExtensionError(
            f"저장된 config.vocab_size {candidate.config.vocab_size}")
    weight_check = check_weights(pristine, candidate, plan)
    audio_check = (compare_on_clips(reference, candidate, processor, plan)
                   if reference else None)

    report = provenance(
        source, plan, tokenizer_check, weight_check, audio_check,
        {"mode": init, "bias": bias, "seed": seed}, config_ids)
    (out / "vocab_extension.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    if report.get("status") == "NOTHING_TO_ADD":
        return "추가할 음절이 없습니다. 이미 전부 어휘 안에 있습니다."

    width = report["acoustic_width"]
    lines = [
        f"음향 출력 {width['before']} → {width['after']}",
        "  " + "  ".join(f"{t} {i}" for t, i in report["added"].items()),
        f"  유지된 id {report['weights']['rows_carried_over']}개 · "
        f"blank {report['tokenizer']['blank_id']} · "
        f"unk {report['tokenizer']['unk_id']} · "
        f"구분자 {report['tokenizer']['delimiter_id']}",
        "  출력 밖: " + "  ".join(
            f"{t} {i}" for t, i
            in report["tokenizer"]["beyond_acoustic_range"].items()),
        f"  토크나이저 길이 {report['tokenizer']['tokenizer_length']} "
        "(음향 폭 아님)",
        f"  파라미터 {report['weights']['parameters_compared']}개 대조, "
        "새 행 외 전부 일치",
    ]

    audio = report.get("reference_audio")
    if audio:
        lines += [
            "",
            f"합성 골든 {audio['clips']}클립 (원본 vs 후보)",
            f"  전사 달라진 클립 {audio['transcripts_changed']}개 · "
            f"새 토큰이 이긴 프레임 {audio['new_token_frames']}개 · "
            f"기존 열 최대차 {audio['max_old_column_diff']:.2e}",
        ]
        for case in audio["cases"]:
            mark = "=" if case["source_text"] == case["candidate_text"] else "!"
            lines.append(f"  {mark} {case['clip']:16} {case['target']:6} "
                         f"{case['candidate_text']!r}")
        lines.append("  합성 TTS 이며, 정답률이 아니라 그래프가 그대로인지를")
        lines.append("  보는 검사입니다.")

    lines += ["", "이 체크포인트는 학습되지 않았습니다. 새 행은 오디오를 본",
              "적이 없습니다."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", type=Path, required=True,
                   help="원본 체크포인트 디렉터리 (건드리지 않음)")
    p.add_argument("--out", type=Path, required=True,
                   help="새 실험용 체크포인트를 쓸 빈 디렉터리")
    p.add_argument("--add", action="append", default=[],
                   help="추가할 한글 음절 (여러 번 가능)")
    p.add_argument("--add-from", type=Path, action="append", default=[],
                   help="targets.json 이나 words.csv 에서 빠진 음절을 가져옴")
    p.add_argument("--new-row-init", choices=("inert", "hf"), default="inert",
                   help="inert: 0 가중치 + 낮은 bias. hf: 표준 초기화")
    p.add_argument("--new-row-bias", type=float, default=-10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify-clips", type=int, default=6,
                   help="원본과 대조할 합성 골든 클립 수 (0 이면 생략)")
    p.add_argument("--keep-config-token-ids", action="store_true",
                   help="config 의 bos/eos id 를 원본 값 그대로 둠")
    args = p.parse_args(argv)

    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    requested = list(args.add)
    try:
        if args.add_from:
            vocab = vc.load_checkpoint(args.model)
            for path in args.add_from:
                report = vc.audit(vc.read_source(path), vocab)
                requested += list(report["unsupported"])
        if not requested:
            print("ERROR: --add 나 --add-from 이 필요합니다", file=sys.stderr)
            return 1

        report = extend(
            args.model, args.out, requested, init=args.new_row_init,
            bias=args.new_row_bias, seed=args.seed,
            verify_clips=args.verify_clips,
            keep_config_token_ids=args.keep_config_token_ids)
    except (ExtensionError, vc.VocabularyError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(format_report(report))
    if report.get("status") != "NOTHING_TO_ADD":
        print(f"\n→ {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
