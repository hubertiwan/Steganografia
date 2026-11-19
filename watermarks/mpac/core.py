"""Core of MPAC, a multi-bit watermark via position allocation.

Yoo, Ahn, Kwak, "Advancing Beyond Identification: Multi-bit Watermark for Large Language
Models via Position Allocation", NAACL 2024, arXiv:2308.00221.

The watermark carries a b-bit message (e.g. a user ID), not only "watermarked or not".

- The message is written in radix r = floor(1/gamma), which gives b~ = ceil(b / log2 r)
  digits (positions). For gamma = 0.25: r = 4, an 8-bit message has 4 positions.
- MPAC (Section 3.2), for every new token:
    1. s = hash(key, previous h tokens)
    2. p = a random position in [b~], using s as seed
    3. m = message digit at position p
    4. permute the vocabulary using s as seed
    5. split it into r colorlists of floor(gamma*|V|) tokens (the remainder is dropped)
    6. add delta to the logits of colorlist m
- Decoding (Algorithm 1): for every token, rebuild p and the colorlists and count the token
  in W[p][colorlist]. The decoded digit at position p is argmax W[p], and
  w = sum_p max W[p] is the number of tokens in the predicted colorlists.
- Detection (Section 3.3): z = (w - gamma*T) / sqrt(T*gamma*(1-gamma)).

This z is biased upwards for text without a watermark, because it takes the maximum over
r colorlists at every position (the paper shows this in Fig. 8 and Appendix A.2). So the
decision uses an exact p-value instead: under the null hypothesis W[p] is multinomial
(Eq. 3), the distribution of max W[p] is computed exactly with the Poisson representation
from Levin (1981) (the paper uses the same representation with a normal approximation,
Appendix A.2.2), and the distribution of w is the convolution over positions.
"""
import hashlib
from collections import OrderedDict
from math import ceil, erfc, floor, lgamma, log, log2, sqrt

import numpy as np


def message_to_digits(bits, radix, positions):
    """'10110010' -> list of `positions` digits in base `radix` (most significant first)."""
    value = int(bits, 2)
    digits = []
    for _ in range(positions):
        value, d = divmod(value, radix)
        digits.append(d)
    return digits[::-1]


def digits_to_message(digits, radix, num_bits):
    """Inverse of message_to_digits. If radix**positions > 2**num_bits, the decoded value
    can be too large; then only the lowest num_bits bits are kept."""
    value = 0
    for d in digits:
        value = value * radix + int(d)
    value %= 2 ** num_bits
    return format(value, f"0{num_bits}b")


def bit_accuracy(true_bits, decoded_bits):
    """Fraction of bits decoded correctly (1 - bit error rate)."""
    same = 0
    for a, b in zip(true_bits, decoded_bits):
        if a == b:
            same += 1
    return same / len(true_bits)


class MPACWatermark:
    def __init__(self, key, vocab_size, message_bits=8, gamma=0.25, delta=2.0, context_size=1):
        if not 0 < gamma <= 0.5:
            raise ValueError("gamma must be in (0, 0.5], so that there are at least 2 colorlists")
        if message_bits < 1:
            raise ValueError("message_bits must be >= 1")
        if context_size < 1:
            raise ValueError("context_size must be >= 1")
        self.key = int(key)
        self.vocab_size = int(vocab_size)
        self.message_bits = int(message_bits)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.context_size = int(context_size)  # h in the paper

        self.radix = floor(1 / self.gamma + 1e-9)  # r; 1e-9 guards against 1/0.2 = 4.999...
        self.list_size = int(self.gamma * self.vocab_size)
        if self.list_size < 1:
            raise ValueError("gamma * vocab_size must be at least 1")
        self.positions = ceil(self.message_bits / log2(self.radix))  # b~ in the paper
        while self.radix ** self.positions < 2 ** self.message_bits:  # float safety
            self.positions += 1
        # exact colorlist fraction (gamma*|V| may not be an integer), used in the statistics
        self.gamma_eff = self.list_size / self.vocab_size
        self._cdf_cache = {}
        self._color_cache = OrderedDict()  # seed -> (position, colorlist index of every token)

    # ---------------- message ----------------
    def check_message(self, bits):
        if len(bits) != self.message_bits or set(bits) - {"0", "1"}:
            raise ValueError(f"message must be a string of {self.message_bits} zeros and ones")
        return message_to_digits(bits, self.radix, self.positions)

    # ---------------- steps 1, 2, 4, 5 ----------------
    def seed(self, context):
        """s = hash(key, last h tokens)."""
        last_tokens = []
        for t in context[-self.context_size:]:
            last_tokens.append(str(int(t)))
        data = (str(self.key) + "|" + ",".join(last_tokens)).encode("utf-8")
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")

    def position_and_permutation(self, context):
        """Position p and permuted vocabulary, both drawn from one generator seeded with s."""
        rng = np.random.default_rng(self.seed(context))
        position = int(rng.integers(self.positions))
        perm = rng.permutation(self.vocab_size)
        return position, perm

    def colorlist(self, perm, color):
        """Colorlist number `color` = the color-th block of floor(gamma*|V|) permuted tokens."""
        return perm[color * self.list_size:(color + 1) * self.list_size]

    def colors(self, context):
        """(position, array with the colorlist number of every token, -1 = remainder).
        Kept in a small cache: with h = 1 the same previous token appears many times and a
        permutation of a 150k vocabulary takes a few milliseconds."""
        seed = self.seed(context)
        if seed in self._color_cache:
            self._color_cache.move_to_end(seed)
            return self._color_cache[seed]
        position, perm = self.position_and_permutation(context)
        colors = np.full(self.vocab_size, -1, dtype=np.int16)
        for m in range(self.radix):
            colors[self.colorlist(perm, m)] = m
        self._color_cache[seed] = (position, colors)
        if len(self._color_cache) > 256:
            self._color_cache.popitem(last=False)
        return position, colors

    def locate(self, context, token):
        """(position, colorlist of the token) for one token; colorlist is None if the token
        is in the dropped remainder."""
        position, colors = self.colors(context)
        color = int(colors[int(token)])
        if color < 0:
            color = None
        return position, color

    # ---------------- step 3 and 6: embedding ----------------
    def watermark_logits(self, logits, context, digits):
        """l_hat[v] = l[v] + delta * 1(v in colorlist m), m = digits[p]."""
        logits = np.array(logits, dtype=np.float64)
        position, colors = self.colors(context)
        logits[colors == digits[position]] += self.delta
        return logits

    # ---------------- Algorithm 1: decoding ----------------
    def count(self, tokens):
        """Counter W[p][m] and the number of tokens per position. The first h tokens have no
        full context in the text and are skipped (during generation they used the prompt)."""
        tokens = [int(t) for t in tokens]
        W = np.zeros((self.positions, self.radix), dtype=int)
        per_position = np.zeros(self.positions, dtype=int)
        for t in range(self.context_size, len(tokens)):
            position, color = self.locate(tokens[t - self.context_size:t], tokens[t])
            per_position[position] += 1
            if color is not None:
                W[position, color] += 1
        return W, per_position

    def decode(self, tokens):
        W, per_position = self.count(tokens)
        digits = W.argmax(axis=1)  # ties and empty positions -> digit 0
        w = int(W.max(axis=1).sum())
        message = digits_to_message(digits, self.radix, self.message_bits)
        return message, w, W, per_position

    # ---------------- Section 3.3: detection ----------------
    def z_score(self, w, n):
        """Eq. (1): z = (w - gamma*T) / sqrt(T*gamma*(1-gamma))."""
        if n == 0:
            return 0.0
        g = self.gamma_eff
        return (w - g * n) / sqrt(n * g * (1 - g))

    def max_cell_cdf(self, n):
        """F[a] = P(max_m W[p][m] <= a) for a position with n tokens, under the null hypothesis
        W[p] ~ Multinomial(n, [g, ..., g, 1 - r*g]) (the last cell is the dropped remainder).

        Levin (1981): with independent X_m ~ Poisson(n*p_m),
        P(all colorlists <= a) = P(X_1..X_r <= a and sum X = n) / P(sum X = n).
        The numerator is a convolution of truncated Poisson pmfs, so the result is exact."""
        if n in self._cdf_cache:
            return self._cdf_cache[n]
        F = np.ones(n + 1)
        if n > 0:
            g = self.gamma_eff
            rest = max(1.0 - self.radix * g, 0.0)
            k = np.arange(n + 1)
            log_fact = np.array([lgamma(i + 1) for i in range(n + 1)])
            pois = np.exp(k * log(n * g) - n * g - log_fact)
            if rest > 0:
                pois_rest = np.exp(k * log(n * rest) - n * rest - log_fact)
            else:
                pois_rest = np.zeros(n + 1)
                pois_rest[0] = 1.0
            p_sum_is_n = np.exp(n * log(n) - n - lgamma(n + 1))
            for a in range(n + 1):
                part = pois[:a + 1]
                conv = part
                for _ in range(self.radix - 1):
                    conv = np.convolve(conv, part)[:n + 1]
                conv = np.convolve(conv, pois_rest)[:n + 1]
                F[a] = conv[n] / p_sum_is_n if len(conv) > n else 0.0
                if F[a] > 1 - 1e-15:
                    break  # the rest stays 1
            F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
        self._cdf_cache[n] = F
        return F

    def p_value(self, w, per_position):
        """P(sum_p max W[p] >= w) under the null hypothesis, for the observed tokens per position.
        Values below about 1e-15 are limited by floating point precision."""
        total = np.array([1.0])
        for n in per_position:
            F = self.max_cell_cdf(int(n))
            pmf = np.diff(np.concatenate([[0.0], F]))
            total = np.convolve(total, np.maximum(pmf, 0.0))
        tail = float(total[int(w):].sum()) if w < len(total) else 0.0
        return min(max(tail, 0.0), 1.0)

    def detect(self, tokens, alpha=0.001):
        """Decode the message and test for the watermark. Watermarked if p-value <= alpha."""
        message, w, W, per_position = self.decode(tokens)
        n = int(per_position.sum())
        p = self.p_value(w, per_position) if n > 0 else 1.0
        return {
            "n": n,
            "message": message,
            "w": w,
            "z": self.z_score(w, n),
            "p_value": p,
            "p_value_normal": 0.5 * erfc(self.z_score(w, n) / sqrt(2)),  # paper's z, for reference
            "watermarked": n > 0 and p <= alpha,
            "counts": W.tolist(),
            "tokens_per_position": per_position.tolist(),
            "empty_positions": int(np.sum(per_position == 0)),
        }
