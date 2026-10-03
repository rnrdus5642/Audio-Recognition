"""Reference-free CTC search and exact, post-hoc sequence likelihoods.

Kept outside the application. Search never receives a transcript; a reference
sequence is scored only after search, as a diagnostic comparison.
"""
from collections import defaultdict
from itertools import product
import math

import numpy as np


def collapse(path, blank=0):
    previous = None
    result = []
    for token in path:
        if token != previous and token != blank:
            result.append(int(token))
        previous = token
    return tuple(result)


def log_softmax(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def sequence_logp(logp, target, blank=0):
    """Sum every CTC alignment for one sequence, without a beam/pruning."""
    expanded = np.full(2 * len(target) + 1, blank, dtype=np.int64)
    expanded[1::2] = target
    alpha = np.full(len(expanded), -math.inf)
    alpha[0] = 0.0
    skip = np.zeros(len(expanded), dtype=bool)
    skip[2:] = ((expanded[2:] != blank)
                & (expanded[2:] != expanded[:-2]))
    for frame in logp:
        one = np.concatenate(([-math.inf], alpha[:-1]))
        two = np.full_like(alpha, -math.inf)
        if len(alpha) > 2:
            two[2:] = alpha[:-2]
        two[~skip] = -math.inf
        alpha = np.logaddexp(np.logaddexp(alpha, one), two) + frame[expanded]
    return float(np.logaddexp(alpha[-1], alpha[-2])
                 if len(alpha) > 1 else alpha[-1])


def prefix_beam(logp, width=25, blank=0):
    """Unconditioned prefix beam: all tokens, no lexicon or length bonus."""
    neg = -math.inf
    beams = {(): (0.0, neg)}
    for frame in logp:
        nxt = defaultdict(lambda: (neg, neg))

        def add(prefix, pb, pn):
            a, b = nxt[prefix]
            nxt[prefix] = (float(np.logaddexp(a, pb)),
                           float(np.logaddexp(b, pn)))

        for prefix, (pb, pn) in beams.items():
            total = float(np.logaddexp(pb, pn))
            add(prefix, total + frame[blank], neg)
            for token, value in enumerate(frame):
                if token == blank:
                    continue
                if prefix and prefix[-1] == token:
                    add(prefix, neg, pn + value)
                    add(prefix + (token,), neg, pb + value)
                else:
                    add(prefix + (token,), neg, total + value)
        beams = dict(sorted(nxt.items(),
                            key=lambda item: -np.logaddexp(*item[1]))[:width])
    return [(list(seq), float(np.logaddexp(*prob)))
            for seq, prob in sorted(beams.items(),
                                    key=lambda item: -np.logaddexp(*item[1]))]


def self_check():
    """Independent exhaustive enumeration, including repeats and blanks."""
    import torch

    rng = np.random.default_rng(20260925)
    worst = 0.0
    trials = 0
    for frames in (1, 2, 4):
        logp = log_softmax(rng.normal(size=(frames, 3)))
        enumerated = defaultdict(float)
        for path in product(range(3), repeat=frames):
            probability = math.exp(sum(logp[t, c] for t, c in enumerate(path)))
            enumerated[collapse(path)] += probability
        beam = dict((tuple(s), math.exp(p))
                    for s, p in prefix_beam(logp, width=1000))
        assert abs(sum(enumerated.values()) - 1.0) < 1e-12
        for target, probability in enumerated.items():
            exact = sequence_logp(logp, target)
            expected = math.log(probability)
            torch_loss = torch.nn.functional.ctc_loss(
                torch.tensor(logp[:, None, :], dtype=torch.float64),
                torch.tensor(target, dtype=torch.long),
                torch.tensor([frames]), torch.tensor([len(target)]),
                blank=0, reduction='sum')
            error = max(abs(exact - expected), abs(float(torch_loss) + expected),
                        abs(beam[tuple(target)] - probability))
            worst = max(worst, error)
            assert error < 1e-10, (target, error)
            trials += 1
    assert collapse([1, 1, 0, 1]) == (1, 1)
    return {'passed': True, 'sequence_comparisons': trials,
            'max_error': worst,
            'independent_checks': ['exhaustive paths', 'torch CTCLoss',
                                   'unpruned prefix beam']}
