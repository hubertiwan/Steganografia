"""Core of DiPmark, a distribution-preserving watermark.

Wu, Hu, Guo, Zhang, Huang, "A Resilient and Accessible Distribution-Preserving
Watermark for Large Language Models", ICML 2024, arXiv:2310.07710.

- Cipher: the last a tokens (texture key) and a secret key give a random permutation
  theta of the vocabulary.
- DiP-reweight (Definition 4.2): the token probabilities are laid out on [0, 1] in the
  order of theta. The probability mass in [0, alpha] is moved to [1 - alpha, 1], i.e. to
  the tokens at the end of the permutation. Averaged over all permutations the
  distribution does not change (Theorem 4.3), so the text quality is preserved.
- Detection (Algorithm 2): the green list is the end of the permutation,
  theta[ceil(gamma*N):]. Count green tokens L_G and compute Phi = L_G/n - (1 - gamma).
"""
import hashlib
from math import ceil, exp, log, sqrt

import numpy as np


def kl_bernoulli(p, q):
    """KL(p || q) for two coins with P(heads) = p and q."""
    result = 0.0
    if p > 0:
        result += p * log(p / q)
    if p < 1:
        result += (1 - p) * log((1 - p) / (1 - q))
    return result


class DiPmarkWatermark:
    def __init__(self, key, vocab_size, alpha=0.45, gamma=0.5, context_size=1):
        if not 0 <= alpha <= 0.5:
            raise ValueError("alpha must be in [0, 0.5]")
        if not 0 < gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if context_size < 1:
            raise ValueError("context_size must be >= 1")
        self.key = int(key)
        self.vocab_size = int(vocab_size)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.context_size = int(context_size)  # a in the paper

    # ---------------- cipher: permutation from the texture key ----------------
    def permutation(self, context):
        """theta = h(key, last a tokens): a random permutation of the vocabulary."""
        last_tokens = []
        for t in context[-self.context_size:]:
            last_tokens.append(str(int(t)))
        data = (str(self.key) + "|" + ",".join(last_tokens)).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        return np.random.default_rng(seed).permutation(self.vocab_size)

    # ---------------- Definition 4.2: DiP-reweight ----------------
    def reweight(self, probs, context):
        """F(i) = max(cum_i - alpha, 0) + max(cum_i - (1 - alpha), 0), P_W(t_i) = F(i) - F(i-1),
        where cum_i is the total probability of the first i tokens of the permutation."""
        probs = np.asarray(probs, dtype=np.float64)
        perm = self.permutation(context)

        cum = np.cumsum(probs[perm])
        first_part = np.maximum(cum - self.alpha, 0.0)
        second_part = np.maximum(cum - (1 - self.alpha), 0.0)
        F = first_part + second_part

        F_before = np.zeros_like(F)
        F_before[1:] = F[:-1]
        new_probs_in_perm_order = F - F_before

        new_probs = np.zeros_like(probs)
        new_probs[perm] = new_probs_in_perm_order
        new_probs = np.maximum(new_probs, 0.0)  # remove tiny negative rounding errors
        return new_probs / new_probs.sum()

    # ---------------- Algorithm 2: detection ----------------
    def green_mask(self, context):
        """Green list = the end of the permutation, theta[ceil(gamma*N):]."""
        perm = self.permutation(context)
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[perm[ceil(self.gamma * self.vocab_size):]] = True
        return mask

    def score(self, tokens):
        """Phi = L_G/n - (1 - gamma). Returns (phi, green_count, n).
        The first a tokens have no full texture key and are skipped. A texture key that was
        already seen is skipped too (during generation it was not watermarked)."""
        tokens = [int(t) for t in tokens]
        seen = set()
        green = 0
        n = 0

        for t in range(self.context_size, len(tokens)):
            context = tuple(tokens[t - self.context_size:t])
            if context in seen:
                continue
            seen.add(context)

            mask = self.green_mask(context)
            if mask[tokens[t]]:
                green += 1
            n += 1

        if n == 0:
            return 0.0, 0, 0
        phi = green / n - (1 - self.gamma)
        return phi, green, n

    def p_value(self, phi, n):
        """Theorem 5.2: P(Phi >= phi) <= exp(-n * KL(phi + 1 - gamma || 1 - gamma))."""
        if n == 0 or phi <= 0:
            return 1.0
        q = 1 - self.gamma
        return exp(-n * kl_bernoulli(phi + q, q))

    def detect(self, tokens, fpr=0.01):
        """Watermarked if Phi >= sqrt(ln(1/fpr) / (2n)). For fpr = 1% this is the paper's
        threshold 1.517 / sqrt(n) (Hoeffding bound)."""
        phi, green, n = self.score(tokens)
        if n > 0:
            threshold = sqrt(log(1 / fpr) / (2 * n))
            green_frac = green / n
        else:
            threshold = float("inf")
            green_frac = 0.0
        return {
            "n": n,
            "green": green,
            "green_frac": green_frac,
            "phi": phi,
            "threshold": threshold,
            "p_value": self.p_value(phi, n),
            "watermarked": phi >= threshold,
        }
