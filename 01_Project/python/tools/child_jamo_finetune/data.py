"""Batches of real audio, grouped by length and bounded by padded size.

Utterances here run from half a second to fifteen. Batching them by
count would either waste most of a batch on padding or blow up on the
long end, so a batch is admitted while `count x longest` stays under a
measured sample budget. Sorting by length first keeps the padding small.

Batch order is a pure function of (seed, epoch), so resuming mid-epoch
replays exactly the batches the interrupted run would have seen.
"""
from pathlib import Path
import json
import queue
import random
import sys
import threading

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio
from common import WORK


def rows(split):
    path = WORK / f'manifest-{split}.jsonl'
    return [json.loads(line) for line in
            path.read_text(encoding='utf-8').splitlines()]


def plan(rows_, budget, max_batch, seed, epoch):
    """Batch index lists for one epoch, deterministic in (seed, epoch)."""
    rng = random.Random(f'{seed}:{epoch}')
    order = list(range(len(rows_)))
    rng.shuffle(order)
    # Shuffle first, then sort by length: equal-length utterances still
    # land in a different batch each epoch.
    order.sort(key=lambda i: rows_[i]['samples'])
    batches, current, longest = [], [], 0
    for i in order:
        n = rows_[i]['samples']
        width = max(longest, n)
        if current and ((len(current) + 1) * width > budget
                        or len(current) >= max_batch):
            batches.append(current)
            current, longest = [i], n
        else:
            current.append(i)
            longest = width
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def groups(batches, accumulate):
    """Contiguous slices of the batch plan, one per optimizer update.

    The last slice of an epoch is usually short. It is kept rather than
    dropped; `train.py` divides each update by the utterances it actually
    covers, so a short tail carries its own weight instead of a wrong one.
    """
    return [batches[i:i + accumulate]
            for i in range(0, len(batches), accumulate)]


def collate(rows_, indices, device):
    """Padded waveforms, a true-length mask, and flat CTC targets."""
    waves = []
    for i in indices:
        r = rows_[i]
        blob = audio.payload(r['archive'], r['offset'], r['size'])
        x, policy = audio.crop(audio.decode(blob), r['speech_start'],
                               r['speech_end'])
        # The manifest recorded both the length and the crop policy the
        # audit measured. Disagreeing on either means the bytes behind
        # this offset are not the ones that were audited.
        if len(x) != r['samples'] or policy != r['span_policy']:
            raise ValueError(
                f'{r["wav"]}: {len(x)} samples / {policy}, manifest says '
                f'{r["samples"]} / {r["span_policy"]}')
        waves.append(x)
    width = max(len(x) for x in waves)
    values = np.zeros((len(waves), width), dtype=np.float32)
    mask = np.zeros((len(waves), width), dtype=np.int64)
    for b, x in enumerate(waves):
        values[b, :len(x)] = x
        mask[b, :len(x)] = 1
    targets = [t for i in indices for t in rows_[i]['label_ids']]
    return {
        'input_values': torch.from_numpy(values).to(device, non_blocking=True),
        'attention_mask': torch.from_numpy(mask).to(device, non_blocking=True),
        'targets': torch.tensor(targets, dtype=torch.long),
        'target_lengths': torch.tensor([len(rows_[i]['label_ids'])
                                        for i in indices], dtype=torch.long),
        'samples': [len(x) for x in waves],
    }


class Prefetch:
    """Decode the next few batches on one worker thread, in order.

    Reading and decoding a batch is a megabyte or so off an SSD; the
    encoder step it hides behind is far longer. One thread is enough,
    and it keeps batch order exactly as `plan` laid it out, which is what
    mid-epoch resume relies on.
    """

    def __init__(self, rows_, batches, device, start=0, depth=4):
        self.batches, self.device, self.start = batches, device, start
        self.queue = queue.Queue(maxsize=depth)
        self.thread = threading.Thread(
            target=self._work, args=(rows_,), daemon=True)
        self.thread.start()

    def _work(self, rows_):
        for i in range(self.start, len(self.batches)):
            try:
                self.queue.put((i, collate(rows_, self.batches[i], 'cpu'), None))
            except Exception as exc:                       # surfaced in __iter__
                self.queue.put((i, None, exc))
                return

    def __iter__(self):
        for _ in range(len(self)):
            i, batch, error = self.queue.get()
            if error is not None:
                raise error
            yield i, {k: (v.to(self.device, non_blocking=True)
                          if k in ('input_values', 'attention_mask') else v)
                      for k, v in batch.items()}

    def __len__(self):
        return len(self.batches) - self.start
