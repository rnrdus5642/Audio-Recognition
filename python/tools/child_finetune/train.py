"""Fine-tune the recogniser on children's speech.

Only the acoustic stage changes. Everything downstream - the
phonological rules, the IPA mapping, the matcher - keeps working on
whatever Hangul comes out, so this script's whole job is to make that
Hangul right more often for a five year old.

It is worth being clear about why. On adults the recogniser makes a 1.7%
character error and the product confirms 87.5% of words. On children it
confirms 48.7%, and only 52.8% of cases could pass the thresholds at
all - the other half are not borderline, they are heard as a different
word entirely. No threshold reaches those. This does.

The adult model is left alone; this writes a second one. Mixing adult
audio in to hold onto adult accuracy is the standard move and is
deliberately skipped, because the adult path keeps its own weights.

Batches are formed by audio seconds rather than by utterance count.
Lengths run from 1 to 15 seconds, and a fixed count either wastes most
of a batch on padding or runs out of VRAM on the long tail.

    python -m python.tools.child_finetune.train --steps 100   # 속도 측정
    python -m python.tools.child_finetune.train
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL = "kresnik/wav2vec2-large-xlsr-korean"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RUNS = HERE / "runs"

# Trimming to the labelled speech region drops about half a second per
# utterance. The margin keeps a little room either side so the model
# still sees onsets and releases rather than clipped ones.
TRIM_MARGIN = 0.3

# Dev is always scored at this batch size, whatever the training one
# is. Padding shifts the result by a few tenths of a point under
# fp16, which is small but enough to muddy an epoch-to-epoch
# comparison if the two runs batched differently.
EVAL_BATCH_SECONDS = 50.0


class Utterances(Dataset):
    """Manifest rows, decoded to audio on demand.

    Audio is read per item rather than cached: 200 hours is 23 GB as
    16-bit PCM and there is no reason for it to be resident.
    """

    def __init__(self, path: Path, processor: Wav2Vec2Processor) -> None:
        with open(path, encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def seconds(self, i: int) -> float:
        row = self.rows[i]
        start, end = row["start"], row["end"]
        if start and end:
            span = float(end) - float(start) + 2 * TRIM_MARGIN
            return min(span, float(row["seconds"]))
        return float(row["seconds"])

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        audio, rate = sf.read(row["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if row["start"] and row["end"]:
            lo = max(0, int((float(row["start"]) - TRIM_MARGIN) * rate))
            hi = min(len(audio),
                     int((float(row["end"]) + TRIM_MARGIN) * rate))
            if hi - lo > rate * 0.5:
                audio = audio[lo:hi]

        # Zero mean, unit variance, over the real samples only - this is
        # what the checkpoint's feature extractor does and the model is
        # useless without it. Normalising after padding would let the
        # zeros shift the statistics of every short utterance.
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)

        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"audio": audio, "labels": labels}


def collate(items: list[dict], processor: Wav2Vec2Processor) -> dict:
    """Pad audio with zeros and labels with -100 so CTC ignores them.

    The tokenizer's pad id is 1204, which is also the CTC blank, so
    padding labels with it would teach the model to emit blanks.
    """
    audio = [it["audio"] for it in items]
    width = max(len(a) for a in audio)
    batch = np.zeros((len(audio), width), dtype=np.float32)
    mask = np.zeros((len(audio), width), dtype=np.int64)
    for i, a in enumerate(audio):
        batch[i, :len(a)] = a
        mask[i, :len(a)] = 1

    longest = max(len(it["labels"]) for it in items)
    labels = np.full((len(items), longest), -100, dtype=np.int64)
    for i, it in enumerate(items):
        labels[i, :len(it["labels"])] = it["labels"]

    return {
        "input_values": torch.from_numpy(batch),
        "attention_mask": torch.from_numpy(mask),
        "labels": torch.from_numpy(labels),
    }


def batches(data: Utterances, budget: float, shuffle: bool,
            seed: int = 0) -> list[list[int]]:
    """Group indices so each batch's padded audio stays under `budget`.

    Sorting by length first is what makes the padding cheap; the batches
    themselves are then shuffled so the model does not see every short
    utterance before every long one.
    """
    order = sorted(range(len(data)), key=data.seconds)
    out: list[list[int]] = []
    current: list[int] = []
    longest = 0.0
    for i in order:
        length = max(longest, data.seconds(i))
        if current and length * (len(current) + 1) > budget:
            out.append(current)
            current, longest = [i], data.seconds(i)
        else:
            current.append(i)
            longest = length
    if current:
        out.append(current)

    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def edits(a: str, b: str) -> int:
    """Levenshtein distance, one row at a time."""
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@torch.no_grad()
def character_error(model, processor, data: Utterances,
                    device: str) -> tuple[float, list]:
    """Greedy CER over the dev set, spaces excluded.

    Spaces are dropped because the matcher never sees them - the answers
    are single words and the pipeline strips to Hangul before G2P.
    """
    model.eval()
    total = 0
    wrong = 0
    samples = []
    for group in batches(data, EVAL_BATCH_SECONDS, shuffle=False):
        batch = collate([data[i] for i in group], processor)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(
                batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device)).logits
        predicted = processor.batch_decode(logits.argmax(-1).cpu().numpy())
        for i, hypothesis in zip(group, predicted):
            reference = data.rows[i]["text"].replace(" ", "")
            hypothesis = hypothesis.replace(" ", "")
            wrong += edits(reference, hypothesis)
            total += len(reference)
            if len(samples) < 8:
                samples.append((reference, hypothesis))
    model.train()
    return (wrong / max(total, 1)), samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=0,
                        help="stop after this many optimiser steps; used "
                             "to measure throughput before committing to "
                             "a full run")
    parser.add_argument("--batch-seconds", type=float, default=50.0)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=RUNS / "child")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 없음. GPU 빌드 torch 가 필요합니다.", file=sys.stderr)
        return 1
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL,
        # Without this a target longer than the frame count yields an
        # infinite loss and takes the weights with it.
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        # SpecAugment is the only regularisation wav2vec2 fine-tuning
        # really has, and the checkpoint ships with it nearly off.
        mask_time_prob=0.065,
        mask_feature_prob=0.1,
    ).to(device)

    # The convolutional front end already hears phones fine; leaving it
    # frozen is standard and saves the activations it would otherwise
    # keep for 240,000 samples per utterance.
    model.freeze_feature_encoder()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터 {trainable/1e6:.0f}M / {total/1e6:.0f}M")

    train = Utterances(DATA / "train.csv", processor)
    dev = Utterances(DATA / "dev.csv", processor)
    print(f"train {len(train):,} 발화 · dev {len(dev):,} 발화")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), eps=1e-8,
                                  weight_decay=0.005)
    scaler = torch.amp.GradScaler("cuda")

    per_epoch = math.ceil(
        len(batches(train, args.batch_seconds, shuffle=False))
        / args.accumulate)
    total_steps = args.steps or per_epoch * args.epochs

    def learning_rate(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        done = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return args.lr * max(0.0, 1.0 - done)

    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.jsonl", "a", encoding="utf-8")
    print(f"epoch 당 {per_epoch:,} step · 총 {total_steps:,} step")

    history: list[dict] = []

    def record(cer: float, epoch: int | None, keep: bool) -> None:
        """Log a dev score, and keep the weights that produced it.

        Every epoch is kept rather than only the best. Dev CER is one
        number over 50 children and the thing we actually care about is
        the confirmation rate over the matcher, which cannot be measured
        until the frame cache is rebuilt - so the epoch that wins on CER
        may not be the epoch that wins on the product, and throwing the
        others away would mean training again to find out.
        """
        entry = {"step": step, "epoch": epoch, "dev_cer": cer}
        history.append(entry)
        log.write(json.dumps(entry) + "\n")
        log.flush()
        (args.out / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8")
        if not keep:
            return
        target = args.out / (f"epoch-{epoch:02d}" if epoch else "best")
        model.save_pretrained(target)
        processor.save_pretrained(target)
        (target / "metrics.json").write_text(
            json.dumps(entry, indent=2), encoding="utf-8")

    step = 0
    # Scoring the untouched checkpoint first gives the history a zero
    # point, and proves the evaluation path works before a night of
    # training rides on it.
    best, _ = character_error(
        model, processor, dev, device)
    record(best, 0, keep=False)
    print(f"기준선 dev CER {best*100:.2f}%")

    audio_seconds = 0.0
    started = time.time()
    running = []
    stop = False

    for epoch in range(args.epochs):
        groups = batches(train, args.batch_seconds, shuffle=True, seed=epoch)
        # partial rather than a closure: Windows starts workers by
        # spawning, which pickles collate_fn, and a local lambda has no
        # importable name.
        loader = DataLoader(
            train, batch_sampler=groups, num_workers=4,
            collate_fn=functools.partial(collate, processor=processor),
            pin_memory=True, persistent_workers=True)

        optimiser.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(
                    batch["input_values"].to(device, non_blocking=True),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device)).loss
            scaler.scale(loss / args.accumulate).backward()
            running.append(loss.item())
            shape = batch["input_values"].shape
            audio_seconds += shape[0] * shape[1] / 16000

            if (i + 1) % args.accumulate:
                continue

            step += 1
            for group in optimiser.param_groups:
                group["lr"] = learning_rate(step)
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            if step % 25 == 0:
                elapsed = time.time() - started
                mean = sum(running) / len(running)
                rate = audio_seconds / elapsed
                remaining = (total_steps - step) / (step / elapsed)
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"step {step:6,}/{total_steps:,}  loss {mean:6.3f}  "
                      f"{rate:5.0f} 오디오초/초  VRAM {peak:4.1f}GB  "
                      f"남은 시간 {remaining/3600:5.2f}h")
                log.write(json.dumps({"step": step, "loss": mean,
                                      "audio_per_s": rate}) + "\n")
                log.flush()
                running = []

            if args.eval_every and step % args.eval_every == 0:
                cer, samples = character_error(
                    model, processor, dev, device)
                improved = cer < best
                if improved:
                    best = cer
                record(cer, None, keep=improved)
                mark = "  <- 최고" if improved else ""
                print(f"  dev CER {cer*100:.2f}%{mark}")
                for reference, hypothesis in samples[:3]:
                    print(f"    {reference}")
                    print(f"    {hypothesis}")

            if args.steps and step >= args.steps:
                stop = True
                break

        if stop:
            break

        cer, _ = character_error(
            model, processor, dev, device)
        if cer < best:
            best = cer
        record(cer, epoch + 1, keep=True)
        print(f"epoch {epoch+1} 끝  dev CER {cer*100:.2f}%  "
              f"-> {args.out / f'epoch-{epoch+1:02d}'}")

    if args.steps:
        elapsed = time.time() - started
        per_step = elapsed / args.steps
        print(f"\n{args.steps} step 에 {elapsed/60:.1f}분")
        print(f"epoch 당 {per_epoch*per_step/3600:.2f}h · "
              f"{args.epochs} epoch 이면 "
              f"{per_epoch*args.epochs*per_step/3600:.1f}h")
    else:
        cer, _ = character_error(
            model, processor, dev, device)
        print(f"최종 dev CER {cer*100:.2f}%  (최고 {best*100:.2f}%)")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
