# Child jamo fine-tuning

Fine-tunes the verified `team-lucid/hubert-xlarge-korean` encoder on AI Hub
Korean child speech with a fresh 42-output jamo CTC head.

This package is separate from `child_finetune/` (the previous model's training
code) and from `encoder_screen/` (the encoder comparison). Neither is imported
for anything that could change their recorded results; the only shared code is
the project's Korean phonological rules in `python/build/g2p/ko/`.

Everything large - manifests, the archive catalogue, logs and checkpoints -
is written under `.venv/hubert-child-20260914/`. That path is inside the
working tree but `.gitignore` excludes `.venv/`, so none of it is tracked and
nothing large can reach a commit.

## What the targets are, and are not

Targets are the corpus transcript (`Transcription.LabelText`) put through
`KoreanG2P.apply_rules` and decomposed into compatibility jamo - the expected
surface form of that transcript. LabelText is documented as a transcription
and some rows carry realisation or insertion notation, so it cannot be said
that every row is simply the prompt the child was given; rows whose notation
we cannot resolve are excluded rather than guessed at.

They are not expert annotations of what the child actually produced. In the
`ErrorTagged` fields sampled here the value was `N/A`; that is sample
evidence, not an exhaustive schema audit of the corpus. `QualityStatus=Good`
is not a statement about phonetic correctness.

So the jamo error rate (JER) reported here measures how well the model
recovers that expected surface form. It is not a phoneme or IPA error rate,
and it says nothing about how often the product would wrongly accept a
mispronunciation.

## Pipeline

```
catalog.py    where every source WAV starts inside its archive
manifest.py   select the recordings, then verify every one of them
quality.py    flag weak inputs and crops that discard strong audio for review
reviewed.py   apply source-bound, human-confirmed repair decisions
repair.py     materialize a separate repaired training split, preserving baseline
integrity.py  the immutable identity of the data a run is bound to
ownership.py  one live writer per run, enforced by an OS lock
train.py      the two-phase, resumable fine-tune
preflight.py  prove the path before spending days of GPU time
labels.py     transcript -> jamo targets, versioned and hashed
smoke.py      a tiny smoke-only view of the train pool, for testing
```

Tests:

```
test_labels.py         the target contract and the crop policy
test_quality_repairs.py lost speech, noise-only pairs, evidence drift and repair isolation
test_state_machine.py  the real entry point driven through every phase,
                       interruption point and refusal, on a ~1M-parameter
                       stand-in encoder so it runs in about a minute
control_experiment.py  resumed vs uninterrupted on the real 962M encoder
restore_check.py       restoration is exact, with no kernel in the way
```

```bash
PY=.venv-gpu/Scripts/python.exe
$PY python/tools/child_jamo_finetune/catalog.py          # ~7 min, one pass
$PY python/tools/child_jamo_finetune/manifest.py select  # ~5 min
$PY python/tools/child_jamo_finetune/manifest.py audit   # decodes everything
$PY python/tools/child_jamo_finetune/preflight.py
$PY -m pytest python/tools/child_jamo_finetune/test_labels.py
```

## Verified data repairs (2026-09-27)

The two listening-confirmed training defects are recorded in
`reviewed-repairs.json`. The `작은 다람쥐` row now uses the verified whole
2.88-second source; the noise-only `하늘에서 눈이 내렸어요` pair is excluded.
The original source archives, train/valid/test manifests and model are retained
for comparison. Text and jamo targets are not rewritten.

```powershell
.venv-gpu/Scripts/python.exe -X utf8 python/tools/child_jamo_finetune/repair.py --workers 6
```

This produces `manifest-train-repaired-20260927.jsonl`, its own
`audit-train-repaired-20260927.json` and a quality review report under WORK.
It verifies every original source and crop hash, preserves all other manifest
rows byte for byte, checks speaker/PCM separation from the holdouts, and refuses
to overwrite an existing repair. Use a new `--name train-repaired-...` for a
separate rebuild. The original audit must remain in place: the repair's provenance
is bound to it, the listening decisions, their evidence and the quality report.

A future training comparison selects this data with
`--train-split train-repaired-20260927` and a new `--name`; validation and test
retain their existing splits. Creating the repaired data does not train or
replace the model. A fresh training run refuses either confirmed defective
pair before loading the encoder. `--resume` preserves the data contract of a
historical run and cannot switch that run to the repaired data.

The regular `manifest.py audit` also applies the same verified decisions and
writes `quality-review.json`. It measures both the source and actual training
crop, including RMS and retained energy, so nonzero background noise and
strong audio excluded by a crop can be surfaced. These are review signals,
not speech recognition or proof of a bad label. Quiet unreviewed recordings
are kept; only explicit listening decisions authorize repairs or exclusions.
Do not re-run the regular audit over the historical baseline when creating a
comparison: use `repair.py` instead.

## Audio is read in place

The isolated 2026-09-27 robustness comparison is implemented in
`augmentation.py` and `robustness.py`. It starts both arms from the same
completed baseline, retains the repaired data provenance, and changes only
the presence of training waveform variations. It does not change inference
audio, the original training entry point, or the deployed checkpoint. See
`docs/CHILD_JAMO_ROBUSTNESS_PILOT_2026-09-27.md` for the fixed comparison,
validation criteria and limitations. `test_augmentation.py` and
`test_robustness.py` exercise its signal, CTC, selection and source invariants.

Nothing is extracted. The archives are plain, uncompressed tars, so
`catalog.py` records `(archive, offset, size)` for all 2,765,817 WAV members
in one header-only pass - it seeks over the payloads and never reads the
530 GiB. Training then opens an archive read-only, seeks, and reads exactly
the bytes it needs.

Reading in place is not mainly about size: 208 hours of 16 kHz mono PCM16 is
about 23 GB (~46 GB if decoded to float32), which would fit. It is about
having one copy of the truth. A second copy is a second thing that can drift
from the manifest, and the manifest addresses audio by byte range, so the
catalogue is bound to each archive's resolved path, size and mtime plus its
own hash. `catalog.verify()` re-checks all of that at training start and
refuses to run against an archive that changed, because those offsets would
then point at different audio. The archives are opened read-only and never
modified.

## Holdouts

Two, and both are fixed before anything is scored:

- the corpus speaker split's own `valid` (224) and `test` (234) speakers;
- the 50 speakers in `child_finetune/data/dev.csv`, which sit *inside* the
  corpus `train` pool and were development data for the previous model. They
  are excluded from training from initialisation onward, and the run starts
  from the original pretrained encoder rather than the 400-update pilot, whose
  training speakers included some of them.

`test` is for final evaluation only; checkpoint selection uses `valid`.
These speakers were used to evaluate a previous model, so they are not
untouched research subjects - only unseen by *this* run's training.

## Training shape

Phase 1 warms up the fresh head with the encoder frozen, so the first real
encoder gradients are not dominated by a random output layer. Phase 2 trains
the transformer stack and the head together; only the convolutional waveform
frontend stays frozen.

FP32 master weights, BF16 autocast, gradient checkpointing, SDPA attention.
Weights plus gradients plus AdamW's two moments come to 15.4 GB in fp32, which
does not fit a 16 GB card alongside activations, so the moments and update for
the lowest `--offload-layers` transformer layers run on the CPU. AdamW's
equations, learning rates, weight decay and the global gradient clip are
unchanged; only the placement of the arithmetic moves.

Stopping: at most 8 epochs. After each completed encoder epoch from the second
onward, the run stops if the best validation JER has not improved by at least
0.002 against where it stood two epochs earlier. The end-of-epoch validation
that decision reads is taken at the epoch boundary, and a checkpoint landing on
an epoch's last group still runs that epilogue when it resumes.

## States

```
head -> encoder -> final_evaluation -> completed
                \-> paused (any point; --resume continues)
```

`final_evaluation` is a state, not a step at the end of a function. Training
finishing and the test score existing are separate events, so a crash between
them is recoverable: an ordinary `--resume` finishes the evaluation and takes
no further optimizer step. A run is marked `completed` only once `result.json`
is on disk, and resuming a completed run is an idempotent no-op - it neither
retrains nor re-scores, so the test split stays spent exactly once.

A resume re-derives the configuration from the current arguments and data and
refuses to start if anything frozen disagrees with the checkpoint: the learning
rates, batch and evaluation settings, the stopping threshold, and the whole
data contract. It never adopts the saved config while running with different
arguments.

## What is frozen, and against what

`integrity.py` binds a run to:

- each source archive by resolved path, size and mtime, plus the catalogue's
  own hash and member count;
- each manifest, checked against the **audit that produced it** rather than
  against itself, so editing `labels.py` and keeping an old manifest is
  refused instead of being recorded as if the new contract had been used;
- the label contract, which hashes `labels.py`'s own bytes along with the
  rule tables;
- the encoder's weights, `config.json` and `preprocessor_config.json` - the
  preprocessor is what declares this checkpoint takes raw unnormalised audio.

Only one process may write a run directory. `ownership.py` takes a real
byte-range lock and holds it for the process lifetime, so a second writer is
refused by the kernel rather than by a check-then-write race, and a crashed run
releases it automatically.

## Running the long job

**The original `main` run is complete.** Its final result is recorded in
`.venv/hubert-child-20260914/main/result.json` (test JER 2.89054%). The repaired
data has not been used to retrain the model. The launcher below refers to the
historical run; new comparisons need their own run name and repaired train split.

```powershell
powershell -File python\tools\child_jamo_finetune\launch.ps1          # start
powershell -File python\tools\child_jamo_finetune\launch.ps1 -Resume  # continue
```

`launch.ps1` detaches the process, so it survives the shell that started it.
It also clears `CHILD_JAMO_ENCODER` and asserts the encoder weight hash first,
so the test fixture's stand-in encoder cannot reach a production run.
`-Resume` restores weights, both AdamW moment sets, the data cursor and all
four RNGs and continues from the exact next update; it is refused if any
frozen setting or the data contract has changed, and it is a no-op on a run
that already finished.

Progress is `.venv/hubert-child-20260914/<name>/progress.json`; the log is
`.venv/hubert-child-20260914/run/<name>.log`. Note that `run.owner.json`
holds the training process's own PID - `launch.json` records the launcher
shim that spawned it, which is its parent, not the trainer.
