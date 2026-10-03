# Child jamo fine-tuning: dataset and training preflight

2026-09-14. Dataset preparation and a bounded GPU preflight for fine-tuning
`team-lucid/hubert-xlarge-korean` on AI Hub Korean child speech with a fresh
42-output jamo CTC head.

**Status update, 2026-09-28: the main run and final test are complete.**
The saved `main/result.json` reports test JER 2.8905% (1,956 edits / 67,669
reference jamo; 2,999 recordings). `main/best.pt` is the model used by the
microphone page. This transcript-derived metric is not pronunciation accuracy.
[Current status, later experiments and document index](CHILD_JAMO_STATUS.md).

The sections below preserve the preflight and recovery record from September
14–15. References to a running job, projected duration or an unscored test split
describe that historical stage, not the current state. The main run was
launched on 2026-09-14 at 17:33.

Everything below is measured, not projected from assumption, except where it
says "projected".

Machine-readable: `.venv/hubert-child-20260914/claude-launch-report.json`
(run status), `claude-preflight-report.json`,
`claude-control-experiment.json`, `claude-finalreview-report.json`.

Monitor with `.venv/hubert-child-20260914/main/progress.json` and
`.venv/hubert-child-20260914/run/main.log`. If it is ever interrupted:

```powershell
powershell -File python\tools\child_jamo_finetune\launch.ps1 -Resume
```

## What the numbers mean

Targets are the corpus transcript (`Transcription.LabelText`) put through the
project's Korean phonological rules (`KoreanG2P.apply_rules`) and decomposed to
compatibility jamo - the **expected surface form** of that transcript.
LabelText is documented as a transcription and some rows carry realisation or
insertion notation, so it cannot be asserted that every row is simply the
prompt a child was given; rows whose notation cannot be resolved are excluded
rather than guessed at.

They are not annotations of what the child actually produced. In the rows
sampled here `ErrorTagged` was `N/A` - sample evidence, not an exhaustive
schema audit of the corpus. `QualityStatus=Good` is not a statement about
phonetic correctness.

So the jamo error rate (JER) here measures recovery of the intended
pronunciation form. **It is not a phoneme or IPA error rate, and it does not
bound the product's false-accept rate.**

## Dataset

Rebuilt from the original archives and `index.csv`; the previous manifest's
`AIHUB정리/audio` paths no longer exist and were not used.

| Split | Recordings | Hours | Speakers | age 5 / 6 / 7 (h) |
|---|---|---|---|---|
| train | 150,253 | 207.93 | 890 | 69.30 / 69.32 / 69.31 |
| valid | 1,500 | 2.16 | 224 | 0.63 / 0.62 / 0.91 |
| test | 2,999 | 4.41 | 234 | 1.32 / 1.28 / 1.82 |

Hours are measured decoded audio over the annotated speech span plus 0.25 s
either side, not the index's file lengths. Ages are balanced by construction:
the budget is split evenly over the three ages, then filled round-robin one
recording per child per pass. That reaches **every one of the 890 eligible
train speakers** rather than letting the most prolific children dominate.

SHA-256: train `a9fdaae1…`, valid `6ad17c37…`, test `6ccddc4a…`.
Source `index.csv` `e9bb4fe2…`, `splits.json` `f2016691…`.

### Holdouts, preserved from initialisation

- The corpus speaker split's own 224 valid and 234 test speakers. `splits.json`
  is unmodified and no valid/test speaker was moved into train.
- All 50 speakers in `child_finetune/data/dev.csv`. These sit *inside* the
  corpus `train` pool and were the previous model's development data;
  excluding them dropped 22,366 otherwise-eligible rows.

Training starts from the verified original pretrained encoder
(`9aeb683b…`, safetensors `14977266…`) and a fresh head, **not** from the
400-update pilot, whose training speakers included some of those 50.

Verified, not assumed: zero speaker overlap between all three split pairs, and
zero identical decoded PCM between them.

These speakers are unseen by *this* run. They were used to evaluate a previous
model, so they are not pristine never-touched subjects.

### Exclusions, counted

From 654,677 age-5-7 rows: 22,366 reserved dev-speaker rows, 6,325 unsupported
transcripts (1.0%), 787 too long, 17 duplicate index rows, 17 ambiguous
archive members, 4 too short, 1 unmappable jamo.

"Unsupported" means digits, Latin, parentheses or brackets. These are excluded
rather than stripped, because the corpus uses brackets inconsistently - in
`읽었어요(일저더요)` the bracket holds an actual realisation, in
`미끄럼을(틀)` it holds an insertion. Deleting the markup would have put
invented text into the targets. Sentence punctuation and typographic
whitespace *are* removed, deliberately, and `test_labels.py` pins that.

The audit then decoded all 154,801 selected recordings and dropped 49: 12
unreadable, 6 silent, 31 duplicate PCM (none cross-split). No recording
disagreed with the index on length.

### Target expression coverage

Reported, not trained toward. Train counts as a standalone word / inside a
longer word: 들어 236 / 2,980; 와 330 / 4,718; 쳐 30 / 1,119; 잡아 99 / 314;
작아 88 / 303; 두드려 85 / 178; 흔들어 61 / 72; 놓아 37 / 75; 놔 3 / 61;
때려 0 / 6; 연주해 0 / 1. 실로폰채, 북채 and 내려놔 do not occur.

What that does and does not mean. The head predicts jamo, not words, so an
expression the corpus never contains whole is still assembled from units it
drills heavily: the rarest jamo any of these expressions needs is ㅍ at
17,696 occurrences, and the next rarest are ㅘ 19,831, ㄸ 23,140 and ㅊ 32,310.
Absence of an exact expression limits **direct examples of that word**; it does
not mean the model cannot produce it. Whether accuracy on these particular
words improves is a separate question that has to be measured separately -
nothing here predicts it, in either direction.

The 53 owned recordings were not used for training or for any hyperparameter
choice.

## Source integrity, and no second copy of the audio

200 hours is ~100 GiB and only ~113 GiB is free, so nothing was extracted. The
archives are plain uncompressed tars, so `catalog.py` recorded
`(archive, offset, size)` for all **2,765,817** WAV members in one header-only
pass - `tarfile` seeks over the payloads, so the 530 GiB was never read.
Training seeks and reads exactly the bytes it needs.

The member count matches the index row count exactly. 70 basenames appear in
more than one archive with different contents; all are excluded. The archives
are opened read-only and were not modified. Selected payloads and their
decoded PCM were hashed during the audit; the 530 GiB was not.

## Preflight results

RTX 5060 Ti, 15.93 GiB. Encoder 962,497,408 parameters, 48 layers, hidden 1280;
958,287,232 trainable with the 4,210,176-parameter convolutional frontend
frozen. SDPA attention confirmed active, gradient checkpointing on.

**Input parity.** This checkpoint's own `preprocessor_config.json` sets
`do_normalize: false`, so the correct input is the raw waveform and the
previous wav2vec2 normalisation would have been wrong here. Checked rather
than assumed: the local feature extractor's output and `data.collate`'s differ
by **exactly 0.0**, and the attention mask this package builds matches the one
the extractor produces when asked. Applying normalisation would have shifted
the input by up to 5.85.

**Masking, decided not just measured.** The same config sets
`return_attention_mask: false`, which is only safe for a single unpadded
input, so the mask is built here. Three FP32 measurements on the same
utterance:

| | max abs logit difference |
|---|---|
| batch of identical copies (no padding at all) — noise floor | 0.000178 |
| short utterance batched with the longest, masked | 0.000179 |
| the same batch with the mask withheld | 1.634 |

Masked padding costs nothing beyond the noise floor, and withholding the mask
is 9,139x larger - so the test can actually detect a broken mask, which a
single unbatched comparison cannot. Tolerance and verdict are asserted in
code, not left to the reader. The earlier 0.0217 figure came from comparing
different-length batches and is superseded by this control.

**Decoding.** On a deliberately mixed-length batch, greedy decoding over each
true length stays inside the vocabulary, and decoding the full padded width
instead adds spurious tokens - which is why eval uses explicit lengths.

**One real update.** Finite CTC loss 33.87 on real train audio (a fresh head
sits near this; it is initialisation, not a score), gradient norm 73.8 before
clipping. The head, a CPU-offloaded layer and a GPU-resident layer all moved;
the frozen frontend did not.

**Restoration is exact.** Loading the real production checkpoint (offload 26)
into a freshly built model, head and optimizer reproduces every encoder tensor,
both AdamW moment sets, the CPU master copies and all four RNG streams with a
maximum difference of **exactly 0.0**. No kernel runs in that check, so it
isolates save/load from anything numerical.

**A resumed run is the same run.** The update that follows a resume is not
bit-identical across processes - but neither are two uninterrupted runs. Three
checkpoints were taken at the same encoder update with the real 962M encoder:

| comparison | worst component difference |
|---|---|
| two **uninterrupted** runs (baseline) | 5.58e-05 |
| resumed vs uninterrupted | 6.04e-05 |

A ratio of 1.08: the drift from interrupting is the same order as the drift
between two runs that were never interrupted. Data cursor and audio exposure
agree exactly. On validation: all three runs agree to every digit on the two
scores taken *before* any encoder update (4.72845953002611 and
3.4882506527415145). After encoder updates the two uninterrupted runs differ
from **each other** - 2.856396866840731 against 2.85378590078329, one
reference edit out of 383 jamo - so the resumed run matching one control is
not evidence that interruption contributes nothing. What the experiment
supports is narrower and sufficient: interruption is not distinguishable from
ordinary run-to-run variation at this scale. The tensor that looked worst earlier,
`encoder.layers.5.attention.k_proj.bias`, differs by 4.49e-05 in the baseline
and 4.31e-05 when resumed - *smaller* when resumed - against its own RMS
magnitude of 0.0104, so it is not resume-specific. `ctc_loss` documents
nondeterministic CUDA gradients, which is why a baseline was measured rather
than assumed; no kernel is named as the confirmed cause, no bitwise claim is
made, and exact byte reproducibility across processes is not promised.

**State machine.** The real entry point is driven through every phase,
interruption point and refusal by `test_state_machine.py`, using a
~1M-parameter stand-in encoder so 11 cases run in about 70 seconds without
multi-gigabyte checkpoints: head handover, a last-group interruption that must
still run the epoch epilogue, the resumable final-evaluation stage, an
idempotent completed run, a refused frozen-setting change, a refused manifest
the audit never vouched for, a refused second writer, and a refused
out-of-root run directory. That suite found a real bug - charging the phase
clock crashed on the new `final_evaluation` phase *after* the result was
written, which is precisely the window that would have stranded a finished
run.

### Memory and throughput, on real planned batches

Measured with gradient accumulation as the run actually uses it, data loading
included.

| Offload layers | Batch budget | s/update | audio s / wall s | Peak reserved | Projected h/epoch |
|---|---|---|---|---|---|
| 26 | 640,000 | 5.56 | 53.1 | 13.39 GiB | 3.92 |
| 26 | 960,000 | 8.09 | 56.5 | 14.57 GiB | 3.68 |
| 20 | 640,000 | 5.33 | 55.6 | 14.33 GiB | 3.74 |
| 20 | 960,000 | 31.04 | 18.3 | 16.81 GiB | 11.35 |

The last row is the cliff: 16.81 GiB exceeds the card and throughput collapses
about threefold. The two middle rows are within ~1.5 GiB of it once the ~1.5
GiB of desktop contexts are counted, so they are not safe for a multi-day run.

**Recommended: 26 offloaded layers, 640,000-sample budget, accumulate 8.**
13.39 GiB reserved leaves real margin. 3.92 h/epoch, so **~31 h for the full
8 epochs**, less if early stopping fires.

Length grouping works almost perfectly here: corpus lengths are quantized to
4096-sample multiples, so padding waste across a full epoch is **0.006%**.
20,171 batches/epoch, 7.4 recordings each, 2,521 updates/epoch.

Why offload at all: fp32 weights, gradients and AdamW's two moments come to
15.4 GiB before any activations. The moments and update for the lowest 26
layers run on the CPU (511.6M parameters offloaded, 446.7M resident). AdamW's
equations, learning rates, weight decay and the global clip are unchanged;
only where the arithmetic happens moves.

## Training configuration

Phase 1: head only, encoder frozen, 1,000 updates at head LR 1e-3.
Phase 2: transformer stack and head, encoder LR 2e-5, head LR 1e-4, 500-update
warmup then linear decay over 20,171 updates. AdamW weight decay 0.01, global
grad clip 1.0, BF16 autocast on FP32 masters, SpecAugment time 0.05 /
feature 0.02, no layerdrop.

Masking is short-sequence-safe: `mask_time_min_masks` is set to 0 so the mask
count follows each sequence's own length. At the default of 2 a half-second
utterance (24 frames) would have had 20 of them masked.

Stopping: at most 8 epochs; after each completed encoder epoch from the second
onward, stop if best valid JER has not improved by 0.002 against where it
stood two epochs earlier. That decision reads an end-of-epoch validation taken
at the boundary, and survives a checkpoint landing on the epoch's last group.
Checkpoint selection uses valid only; test is touched once, at the end.

The run's states are `head -> encoder -> final_evaluation -> completed`, with
`paused` reachable throughout. `final_evaluation` is a real state: training
finishing and the test score existing are separate events, so a crash between
them is recovered by an ordinary `--resume` that takes no further optimizer
step, and `completed` is recorded only once `result.json` is on disk. Resuming
a completed run is an idempotent no-op, so the test split stays spent exactly
once.

A resume re-derives the configuration and refuses to start if anything frozen
differs - learning rates, batch and evaluation settings, the stopping
threshold, and the whole data contract. That contract binds each archive by
resolved path, size and mtime plus catalogue hash; each manifest to the audit
that produced it rather than to itself; the label contract including
`labels.py`'s own bytes; and the encoder's weights together with its
`config.json` and `preprocessor_config.json`. Only one process may write a run
directory, enforced by an OS byte-range lock held for the process lifetime.

Disk: one `last.pt` and one `best.pt` are kept. Both are currently 3.85 GB
(observed), because the encoder's AdamW moments are only allocated on the
first encoder optimizer step and so join the *next* regular `last.pt`, normally
at encoder update 500. That file is estimated at about 11.5 GB; on those
figures the peak including atomic-write temporaries is an estimated ~30.7 GB,
against 105.3 GiB free. `last.pt` is already fully resumable either way - it
carries the weights, the head optimizer moments, the RNG streams, the data
cursor and the frozen config.

## Historical run in progress — 2026-09-14–15

Launched 2026-09-14 17:33 with the verified original XLarge encoder
(`14977266…`) and a fresh 42-output head, detached so it survives the shell
that started it. The launcher clears `CHILD_JAMO_ENCODER` and asserts the
weight hash before starting, so the test fixture's stand-in encoder cannot
reach production training.

Observed through encoder update 150 (2026-09-14 18:13):

| | |
|---|---|
| encoder throughput | 5.73 s/update, 51.9 audio-s per wall-s |
| peak allocated VRAM | 13,749,938,688 B = 12.81 GiB (preflight predicted 12.81) |
| peak reserved VRAM | 15,705,571,328 B = 14.63 GiB, of 15.93 GiB on the device |
| updates per epoch | 2,522 |
| head warmup | 1,000 updates, completed in 25 min |
| training loss, encoder updates 25 -> 150 | 0.9674 -> 0.7814, finite throughout |

Peak reserved is the caching allocator's pool, set by the FP32 validation
pass, and it sits close to device capacity. These are measurements, not
guarantees: nothing here promises the run finishes without an out-of-memory
failure. If one happens the run stops and has to be resumed explicitly with
the command below - there is no watchdog restarting it.

Head warmup is several times faster than encoder training because it has no
encoder backward, so the estimate below uses encoder-phase throughput only.
The validation JER recorded during head warmup measures a linear probe on a
frozen encoder and is not an accuracy result for this run.

**Estimated ~34.4 h to run all 8 configured epochs** at the throughput
observed so far: ~32.1 h of encoder training plus ~2.3 h of validations and
checkpoint writes. That is an estimate for the maximum *configured* schedule,
not a bound on wall-clock time - throughput can change and an interruption
pauses the run until it is resumed. Early stopping can end it sooner; the
earliest is the end of encoder epoch 2.

```powershell
powershell -File python\tools\child_jamo_finetune\launch.ps1          # start
powershell -File python\tools\child_jamo_finetune\launch.ps1 -Resume  # continue
```

### Recovery, 2026-09-15

The run was interrupted by a **Windows Update-initiated restart**, not a
training fault. In order: System event 1074 at 18:29:08 records
`MoUsoCoreWorker.exe` starting a restart one second before the last training
line; a second restart followed at 18:30:33 from `TrustedInstaller.exe`; the
machine booted at 18:30:59; and WindowsUpdateClient event 19 at 18:33:25
records the successful install of the September 2026 security update
**KB5129195**, OS build 26200.9457. There was no event 41 and stderr held no
exception. KB5129195 was installed in this reboot sequence - that does not
establish it was the only update applied.

Training had reached encoder update 14825; the last regular checkpoint was
update 14500, so **325 unsaved updates had to be redone** - that, not the
stale `progress.json` reading, is the real loss. Resumed at 19:10:56 with the
existing `launch.ps1 -Resume`; the dead owner had already released the OS
lock, so nothing was deleted by hand, and the frozen config and data contract
guards passed. In the two logged samples after the resume the losses were
close to the pre-reboot run - update 14525 logged 0.1038 both times, and
14550 logged 0.0973 against 0.0972. Those are two rounded logging samples,
not a formal bound on reproducibility for this run.

Idle time was 41.8 minutes. New trainer PID 7064 (launcher shim 10884).
`best.pt` was untouched and still holds update 14500 at validation JER
0.030348 (1,140/1,500 exact). Wall time from before the reboot is excluded
from the estimates below, which use the 6.40 s/update measured *after* the
resume: roughly 10.5 h for the full 8 epochs - an estimate at that
throughput, not a bound on wall-clock time.

Five encoder epochs are complete and the sixth is running, so the **next
early-stopping decision comes after six completed epochs, at update
6 x 2,522 = 15,132** - 582 steps on from update 14,550, about 1.03 h of pure
training or roughly 1.1 h with evaluation and save overhead. At that boundary
the rule compares the running best against the best after four completed
epochs, 0.032306672292206615 - 0.002 = 0.030306672292206615. The current best
of 0.030347778916279912 sits just above that threshold, so the rule would
fire unless validation improves enough first; whether it does is not
predicted here. Should the run continue, the following decision at seven
completed epochs (update 17,654) compares against 0.03107106262431439
instead.

One observation, without a mechanism: peak reserved VRAM is 17,622,368,256 B
against 17,103,323,136 B of device memory, and per-update time is 6.40 s
against 5.68 s before the reboot. Windows shared-memory residency is a
possible but unconfirmed explanation for a reserved figure above device
capacity - the CUDA counters alone do not measure that, and no profiling was
done. The slower per-update time is a separate observation; no causal link
between the two is claimed, and no out-of-memory-free completion is promised.

## Files added

`python/tools/child_jamo_finetune/`: `common.py`, `labels.py`, `catalog.py`,
`audio.py`, `manifest.py`, `data.py`, `integrity.py`, `ownership.py`,
`optim.py`, `train.py`, `preflight.py`, `restore_check.py`,
`control_experiment.py`, `smoke.py`, `report.py`, `finalreview.py`,
`launch.ps1`, `test_labels.py`, `test_state_machine.py`, `README.md`; and this
document.

No existing file was modified: all 383 tracked files hash to their task-start
baseline and HEAD is unchanged at `366a046`. Nothing was committed. All large
artifacts live under `.venv/hubert-child-20260914/`, which `.gitignore`
excludes.

## Identities recorded for this run

| | sha256 |
|---|---|
| train manifest | `6b15a659…` |
| valid manifest | `4d7839eb…` |
| test manifest | `b1997497…` |
| label contract `child-jamo-2026-09-15` | `ae3bef23…` |
| encoder weights | `14977266…` |

## Limitations

- The preflight proves the training path, memory and throughput. It says
  nothing about convergence or final accuracy.
- JER on these targets does not measure pronunciation-error diagnosis.
- Valid/test speakers are unseen by this run but were used to evaluate a
  previous model.
- Unity compatibility for this encoder holds under Unity 6 GPUCompute with
  Inference Engine 2.6.1. Legacy Sentis 1.2 did not pass for XLarge; no claim
  is made there.
- Three target expressions never appear as whole words and two more are
  effectively absent. That limits direct examples of those words; with a jamo
  head it does not prevent the model from producing them, and any change in
  accuracy on them must be measured separately rather than assumed in either
  direction.
- Separate CUDA processes are not bit-identical, and no bitwise claim is made.
  Restoration itself is exact; the remaining variation is bounded by a
  measured uninterrupted-run baseline.
- `ErrorTagged` was `N/A` in the rows sampled, which is sample evidence rather
  than an exhaustive schema audit.
