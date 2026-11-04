"""Core of Unigram-Watermark.

Zhao, Ananth, Li, Wang, "Provable Robust Watermarking for AI-Generated Text",
arXiv:2306.17439 (ICLR 2024).

- Algorithm 1 (Watermark): a secret key fixes a split of the vocabulary into a
  green list G (gamma * |V| tokens) and a red list R. The logits of green tokens
  are increased by delta.
- Algorithm 2 (Detect): count green tokens and compute the z-score (Eq. 2).
- Algorithm 4 (Detect, "Unique"): the same on the deduplicated text, with the
  threshold from Eq. 9.
"""
import hashlib
from math import erfc, sqrt
from statistics import NormalDist

import numpy as np


class UnigramWatermark:
    def __init__(self, key, vocab_size, gamma=0.5, delta=2.0):
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        self.key = int(key)
        self.vocab_size = int(vocab_size)
        self.gamma = float(gamma)
        self.delta = float(delta)

        # Key -> seed of the random generator F -> random permutation of the vocabulary.
        # The first gamma*|V| tokens of the permutation form the green list G.
        # The list is FIXED for the whole text (K = 1, hence "unigram").
        seed = int.from_bytes(hashlib.sha256(str(self.key).encode()).digest()[:8], "little")
        perm = np.random.default_rng(seed).permutation(self.vocab_size)
        self.green_size = int(self.gamma * self.vocab_size)
        self.green_mask = np.zeros(self.vocab_size, dtype=bool)
        self.green_mask[perm[: self.green_size]] = True
        # exact green fraction (gamma*|V| may not be an integer), used in the z-score
        self.gamma_eff = self.green_size / self.vocab_size

    # ---------------- Algorithm 1, Eq. (1) ----------------
    def watermark_logits(self, logits):
        """l_hat[v] = l[v] + delta * 1(v in G)."""
        logits = np.asarray(logits, dtype=np.float64)
        return logits + self.delta * self.green_mask

    # ---------------- Algorithm 2, Eq. (2) ----------------
    def z_score(self, tokens):
        """z = (|y|_G - gamma*n) / sqrt(n * gamma * (1 - gamma)). Returns (z, green_count)."""
        tokens = np.asarray(tokens, dtype=np.int64)
        n = len(tokens)
        if n == 0:
            return 0.0, 0
        g = self.gamma_eff
        green = int(self.green_mask[tokens].sum())
        return float((green - g * n) / sqrt(n * g * (1 - g))), green

    def detect(self, tokens, tau=6.0):
        """Algorithm 2. Default tau = 6.0 is the empirical threshold from the paper (Fig. 1a)."""
        z, green = self.z_score(tokens)
        n = len(tokens)
        return {
            "n": n,
            "green": green,
            "green_frac": green / max(n, 1),
            "z": z,
            "p_value": 0.5 * erfc(z / sqrt(2)),
            "watermarked": z > tau,
        }

    # ---------------- Algorithm 4, "Unique" detector ----------------
    def unique_threshold(self, m, alpha=0.01):
        """Eq. (9): tau = sqrt(1 - (m-1)/(N-1)) * Phi^{-1}(1 - alpha), N = |V|."""
        N = self.vocab_size
        return sqrt(max(0.0, 1 - (m - 1) / (N - 1))) * NormalDist().inv_cdf(1 - alpha)

    def detect_unique(self, tokens, alpha=0.01):
        """Algorithm 2 run on Unique(y) (each token counted once), threshold from Eq. (9).
        Keeps the false positive rate at about alpha."""
        uniq = list(dict.fromkeys(int(t) for t in tokens))  # Unique(y), order preserved
        z, green = self.z_score(uniq)
        tau = self.unique_threshold(len(uniq), alpha)
        return {"m": len(uniq), "green": green, "z": z, "tau": tau, "watermarked": z > tau}
