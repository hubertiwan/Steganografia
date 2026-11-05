"""Core of the SynthID-Text watermark (Tournament sampling).

Dathathri et al., "Scalable watermarking for identifying large language model
outputs", Nature 634 (2024).

- Random seed: r_t = h(x_{t-H}, ..., x_{t-1}, key), from the last H tokens.
- g-functions: g_l(x, r_t) in {0, 1}, one per layer l = 1..m, pseudorandom from the key.
- Tournament sampling (Algorithm 1): draw 2^m candidates from the model, then play
  m knockout layers. In layer l the token with the larger g_l wins (tie -> random).
- Detection: the mean g-value of a watermarked text is above 1/2. The detector
  computes a z-score of the sum of g-values (no model needed, only the key).
"""
import hashlib
from math import erfc, sqrt


def hash_to_int(*parts):
    """Deterministic 64-bit integer from any values (blake2b hash)."""
    data = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "little")


class TournamentWatermark:
    def __init__(self, key, context_size=4, num_layers=8):
        if context_size < 1:
            raise ValueError("context_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.key = str(key)
        self.context_size = int(context_size)  # H in the paper
        self.num_layers = int(num_layers)      # m in the paper
        self.num_candidates = 2 ** self.num_layers

    # ---------------- seed and g-values ----------------
    def seed(self, context):
        """r_t = h(last H tokens, key)."""
        last_tokens = [int(t) for t in context[-self.context_size:]]
        return hash_to_int("seed", self.key, *last_tokens)

    def g_value(self, token, layer, seed):
        """g_l(x, r_t): one pseudorandom bit, the same for the same (key, seed, token, layer)."""
        return hash_to_int("g", self.key, seed, int(token), layer) & 1

    # ---------------- Algorithm 1: Tournament sampling ----------------
    def tournament_sample(self, probs, context, rng):
        """Pick the next token from the distribution probs (numpy array over the vocabulary).

        The candidates are drawn i.i.d., so neighbouring pairs are already random pairs
        and no shuffling is needed between layers."""
        candidates = rng.choice(len(probs), size=self.num_candidates, p=probs)
        seed = self.seed(context)

        for layer in range(1, self.num_layers + 1):
            winners = []
            for i in range(0, len(candidates), 2):
                left = int(candidates[i])
                right = int(candidates[i + 1])
                left_g = self.g_value(left, layer, seed)
                right_g = self.g_value(right, layer, seed)

                if left_g > right_g:
                    winners.append(left)
                elif right_g > left_g:
                    winners.append(right)
                elif rng.random() < 0.5:
                    winners.append(left)
                else:
                    winners.append(right)
            candidates = winners

        return candidates[0]

    # ---------------- Detection ----------------
    def z_score(self, tokens):
        """z-score of the sum of g-values. Returns (z, number_of_scored_tokens).

        Without a watermark every g-value is a fair coin, so the sum over n tokens and
        m layers is Binomial(n*m, 1/2) and z is about N(0, 1).
        The first H tokens have no full context and are skipped. A context that was
        already seen is skipped too (same as during generation)."""
        tokens = [int(t) for t in tokens]
        seen = set()
        g_sum = 0
        n = 0

        for t in range(self.context_size, len(tokens)):
            context = tuple(tokens[t - self.context_size:t])
            if context in seen:
                continue
            seen.add(context)

            seed = self.seed(context)
            for layer in range(1, self.num_layers + 1):
                g_sum += self.g_value(tokens[t], layer, seed)
            n += 1

        if n == 0:
            return 0.0, 0
        total = n * self.num_layers
        z = (g_sum - total / 2) / sqrt(total / 4)
        return z, n

    def detect(self, tokens, threshold=2.326):
        """Default threshold 2.326 = one-sided z-test with false positive rate 1%."""
        z, n = self.z_score(tokens)
        return {
            "n": n,
            "z": z,
            "p_value": 0.5 * erfc(z / sqrt(2)),
            "watermarked": z > threshold,
        }
