"""Metrics used in the experiments."""
import math

import numpy as np

from watermarks.unigram import tpr_at_fpr


def auc(pos, neg):
    """ROC AUC = P(score of a watermarked text > score of a plain text), ties count 1/2.
    Same as auc() in demos/demo_synthid.py."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def wilson_interval(k, n, z=1.959964):
    """95% confidence interval for a proportion k/n (Wilson). With 20 texts a TPR of 1.0
    still means anything from about 0.84 to 1.0, so the tables always show it."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def rate(flags):
    flags = [bool(f) for f in flags]
    k, n = sum(flags), len(flags)
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "low": low, "high": high}


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def std(values):
    values = [v for v in values if v is not None]
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def tpr_at_fixed_fpr(pos, neg, fpr):
    """TPR when the threshold is chosen so that the given fraction of plain texts is above it
    (tpr_at_fpr from watermarks/unigram). With n plain texts, FPR below 1/n is not measurable."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    tpr, threshold = tpr_at_fpr(np.array(pos), np.array(neg), fpr)
    return {"fpr": fpr, "tpr": float(tpr), "threshold": float(threshold), "reliable": len(neg) * fpr >= 1}
