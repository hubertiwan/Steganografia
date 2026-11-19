"""Core of the KGW watermark (soft red list).

Kirchenbauer, Geiping, Wen, Katz, Miers, Goldstein, "A Watermark for Large Language
Models", ICML 2023, arXiv:2301.10226.

- Algorithm 2 (Soft red list): the previous token and a secret key give a seed. The seed
  splits the vocabulary into a green list G (gamma * |V| tokens) and a red list R.
  The logits of green tokens are increased by delta.
- Detection: count green tokens, where the green list of each token is rebuilt from the
  token before it, and compute the z-score.

Difference to Unigram-Watermark: there the green list is fixed for the whole text,
here it changes at every position (it depends on the previous token).
"""
import hashlib
from math import erfc, sqrt

import numpy as np


class KGWWatermark:
    def __init__(self, key, vocab_size, gamma=0.25, delta=2.0):
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        self.key = int(key)
        self.vocab_size = int(vocab_size)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.green_size = int(self.gamma * self.vocab_size)
        # exact green fraction (gamma*|V| may not be an integer), used in the z-score
        self.gamma_eff = self.green_size / self.vocab_size

    # ---------------- green list from the previous token ----------------
    def green_mask(self, prev_token):
        """hash(key, previous token) -> seed -> random permutation of the vocabulary.
        The first gamma*|V| tokens of the permutation form the green list."""
        data = f"{self.key}|{int(prev_token)}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        perm = np.random.default_rng(seed).permutation(self.vocab_size)

        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[perm[:self.green_size]] = True
        return mask

    # ---------------- Algorithm 2: watermark ----------------
    def watermark_logits(self, logits, prev_token):
        """l_hat[v] = l[v] + delta * 1(v in G(prev_token))."""
        logits = np.asarray(logits, dtype=np.float64)
        return logits + self.delta * self.green_mask(prev_token)

    # ---------------- Detection ----------------
    def z_score(self, tokens):
        """z = (|s|_G - gamma*T) / sqrt(T * gamma * (1 - gamma)). Returns (z, green_count).
        The first token has no previous token in the text, so T = len(tokens) - 1."""
        tokens = [int(t) for t in tokens]
        n = len(tokens) - 1
        if n <= 0:
            return 0.0, 0

        green = 0
        for t in range(1, len(tokens)):
            mask = self.green_mask(tokens[t - 1])
            if mask[tokens[t]]:
                green += 1

        g = self.gamma_eff
        z = (green - g * n) / sqrt(n * g * (1 - g))
        return float(z), green

    def detect(self, tokens, z_threshold=4.0):
        """Default threshold z = 4 is the one used in the paper."""
        z, green = self.z_score(tokens)
        n = max(len(tokens) - 1, 0)
        if n > 0:
            green_frac = green / n
        else:
            green_frac = 0.0
        return {
            "n": n,
            "green": green,
            "green_frac": green_frac,
            "z": z,
            "p_value": 0.5 * erfc(z / sqrt(2)),
            "watermarked": z > z_threshold,
        }
