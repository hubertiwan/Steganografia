"""Core of MPTour, a multi-bit watermark: MPAC position allocation + SynthID Tournament sampling.

Our own combination of two algorithms from the repository:
- MPAC (Yoo et al., NAACL 2024): a b-bit message written in radix r, every token carries one
  digit of the message at a pseudorandom position p.
- SynthID-Text (Dathathri et al., Nature 2024): the token is chosen by a tournament on
  pseudorandom g-values, which keeps the model distribution (in expectation over the key).

- The message has b~ = ceil(b / log2 r) digits (positions). For r = 4: 8 bits -> 4 positions.
- For every new token:
    1. s = hash(key, previous H tokens)                      (SynthID seed)
    2. p = hash("pos", key, s) mod b~                        (MPAC position)
    3. d = message digit at position p
    4. s_d = hash("digit", key, s, d): every digit has its own g-functions
       g_l(x) = hash("g", key, s_d, x, l) & 1, a fair coin for every token and layer
    5. Tournament sampling with these g-values (SynthID Algorithm 1)
- Decoding: for every token and every candidate digit d' sum the g-values over the m layers,
  S[p][d'] += sum_l g_l(x, s_d'). The decoded digit at p is argmax S[p], and
  w = sum_p max S[p] is the score of the decoded message.
- Detection: without a watermark every S[p][d'] is Binomial(n_p * m, 1/2) and the r counters of
  one position are independent (different hashes), so P(max S[p] <= a) = F_bin(a)^r exactly.
  The distribution of w is the convolution over positions, the same as in MPAC.
  z = (w - n*m/2) / sqrt(n*m/4) is shown too, but like the z of MPAC it is biased upwards.
"""
from math import ceil, erfc, lgamma, log, log2, sqrt

import numpy as np

from watermarks.mpac import bit_accuracy, digits_to_message, message_to_digits  # noqa: F401 (bit_accuracy re-exported)
from watermarks.synthid.core import TournamentWatermark, hash_to_int


class MPTourWatermark(TournamentWatermark):
    def __init__(self, key, message_bits=8, radix=4, context_size=4, num_layers=8):
        if message_bits < 1:
            raise ValueError("message_bits must be >= 1")
        if radix < 2:
            raise ValueError("radix must be >= 2")
        super().__init__(key, context_size, num_layers)
        self.message_bits = int(message_bits)
        self.radix = int(radix)  # r
        self.positions = ceil(self.message_bits / log2(self.radix))  # b~
        while self.radix ** self.positions < 2 ** self.message_bits:  # float safety
            self.positions += 1
        self._cdf_cache = {}

    # ---------------- message ----------------
    def check_message(self, bits):
        if len(bits) != self.message_bits or set(bits) - {"0", "1"}:
            raise ValueError(f"message must be a string of {self.message_bits} zeros and ones")
        return message_to_digits(bits, self.radix, self.positions)

    # ---------------- steps 2 and 4 ----------------
    def position(self, seed):
        return hash_to_int("pos", self.key, seed) % self.positions

    def digit_seed(self, seed, digit):
        """Seed of the g-functions of one digit: a separate fair coin per (token, layer, digit)."""
        return hash_to_int("digit", self.key, seed, int(digit))

    # ---------------- step 5: embedding ----------------
    def tournament_sample(self, probs, context, digits, rng):
        seed = self.seed(context)
        return self.tournament(probs, self.digit_seed(seed, digits[self.position(seed)]), rng)

    def token_g_sums(self, token, seed):
        """(position, sum of g-values over the layers for every candidate digit)."""
        sums = []
        for d in range(self.radix):
            s = self.digit_seed(seed, d)
            sums.append(sum(self.g_value(token, layer, s) for layer in range(1, self.num_layers + 1)))
        return self.position(seed), sums

    # ---------------- decoding ----------------
    def count(self, tokens):
        """Counter S[p][d'] and the number of scored tokens per position. The first H tokens
        and repeated contexts are skipped, the same as in SynthID generation and detection."""
        tokens = [int(t) for t in tokens]
        S = np.zeros((self.positions, self.radix), dtype=int)
        per_position = np.zeros(self.positions, dtype=int)
        seen = set()
        for t in range(self.context_size, len(tokens)):
            context = tuple(tokens[t - self.context_size:t])
            if context in seen:
                continue
            seen.add(context)
            position, sums = self.token_g_sums(tokens[t], self.seed(context))
            S[position] += sums
            per_position[position] += 1
        return S, per_position

    def decode(self, tokens):
        S, per_position = self.count(tokens)
        digits = S.argmax(axis=1)  # ties and empty positions -> digit 0
        w = int(S.max(axis=1).sum())
        message = digits_to_message(digits, self.radix, self.message_bits)
        return message, w, S, per_position

    # ---------------- detection ----------------
    def z_score(self, w, n):
        if n == 0:
            return 0.0
        total = n * self.num_layers
        return (w - total / 2) / sqrt(total / 4)

    def max_cell_cdf(self, n):
        """F[a] = P(max_d' S[p][d'] <= a) for a position with n tokens under the null hypothesis:
        r independent Binomial(n*m, 1/2) counters."""
        if n in self._cdf_cache:
            return self._cdf_cache[n]
        N = n * self.num_layers
        k = np.arange(N + 1)
        log_pmf = np.array([lgamma(N + 1) - lgamma(i + 1) - lgamma(N - i + 1) for i in k]) - N * log(2)
        F = np.minimum(np.cumsum(np.exp(log_pmf)), 1.0) ** self.radix
        F[-1] = 1.0
        self._cdf_cache[n] = F
        return F

    def p_value(self, w, per_position):
        """P(sum_p max S[p] >= w) under the null hypothesis (same convolution as MPAC)."""
        total = np.array([1.0])
        for n in per_position:
            F = self.max_cell_cdf(int(n))
            pmf = np.diff(np.concatenate([[0.0], F]))
            total = np.convolve(total, np.maximum(pmf, 0.0))
        tail = float(total[int(w):].sum()) if w < len(total) else 0.0
        return min(max(tail, 0.0), 1.0)

    def detect(self, tokens, alpha=0.001):
        """Decode the message and test for the watermark. Watermarked if p-value <= alpha."""
        message, w, S, per_position = self.decode(tokens)
        n = int(per_position.sum())
        p = self.p_value(w, per_position) if n > 0 else 1.0
        return {
            "n": n,
            "message": message,
            "w": w,
            "z": self.z_score(w, n),
            "p_value": p,
            "watermarked": n > 0 and p <= alpha,
            "counts": S.tolist(),
            "tokens_per_position": per_position.tolist(),
            "empty_positions": int(np.sum(per_position == 0)),
        }

    def verify(self, tokens, message):
        """Test for one known message: the g-values of its digits only, so no maximum and z is
        a plain SynthID z-score (about N(0, 1) without the watermark). Returns (z, p_value)."""
        digits = self.check_message(message)
        S, per_position = self.count(tokens)
        g_sum = sum(int(S[p, d]) for p, d in enumerate(digits))
        z = self.z_score(g_sum, int(per_position.sum()))
        return z, 0.5 * erfc(z / sqrt(2))
