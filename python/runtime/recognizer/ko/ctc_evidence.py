"""CTC forward likelihood of an exact token sequence, with no language model."""
import numpy as np


def sequence_log_probability(log_probabilities, tokens, blank):
    probs = np.asarray(log_probabilities, dtype=np.float64)
    if probs.ndim != 2 or not np.isfinite(probs).all():
        return float("-inf")
    labels = [blank]
    for token in tokens:
        if token == blank or not 0 <= token < probs.shape[1]:
            return float("-inf")
        labels.extend((token, blank))
    previous = np.full(len(labels), -np.inf)
    previous[0] = 0
    for frame in probs:
        current = np.full(len(labels), -np.inf)
        for i, token in enumerate(labels):
            value = previous[i]
            if i > 0:
                value = np.logaddexp(value, previous[i - 1])
            if i > 1 and token != blank and token != labels[i - 2]:
                value = np.logaddexp(value, previous[i - 2])
            current[i] = value + frame[token]
        previous = current
    return float(np.logaddexp(previous[-1], previous[-2]) if tokens else previous[0])
