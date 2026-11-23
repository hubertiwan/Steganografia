"""MPTour-Lists: variant A of MPTour, with MPAC colorlists as the g-functions.

The same as MPTour (watermarks/mptour), only the g-function of a digit is different:
- B (mptour):       g_l(x, d) = hash("g", key, hash("digit", key, s, d), x, l) & 1, a fair coin.
- A (this package): for every layer l the vocabulary is permuted with seed hash("perm", key, s, l)
  and split into r = floor(1/gamma) colorlists of floor(gamma*|V|) tokens (the remainder is
  dropped), as in MPAC. g_l(x, d) = 1 if x is in colorlist d of layer l, else 0.

So P(g = 1) = gamma_eff instead of 1/2, and in one layer a token is green for at most one digit.
Under the null hypothesis S[p] is Multinomial(n_p * m, [g, ..., g, rest]) (one draw per token and
layer), so the exact distribution of max S[p] is the one of MPAC (Levin 1981) with n*m draws.
"""
from collections import OrderedDict
from math import floor, sqrt

import numpy as np

from watermarks.mpac.core import MPACWatermark
from watermarks.mptour import MPTourWatermark
from watermarks.synthid.core import hash_to_int


class MPTourListsWatermark(MPTourWatermark):
    def __init__(self, key, vocab_size, message_bits=8, gamma=0.25, context_size=4, num_layers=8):
        if not 0 < gamma <= 0.5:
            raise ValueError("gamma must be in (0, 0.5], so that there are at least 2 colorlists")
        radix = floor(1 / gamma + 1e-9)  # r; 1e-9 guards against 1/0.2 = 4.999...
        super().__init__(key, message_bits, radix, context_size, num_layers)
        self.vocab_size = int(vocab_size)
        self.gamma = float(gamma)
        self.list_size = int(self.gamma * self.vocab_size)
        if self.list_size < 1:
            raise ValueError("gamma * vocab_size must be at least 1")
        self.gamma_eff = self.list_size / self.vocab_size
        self._color_cache = OrderedDict()

    def layer_colors(self, seed, layer):
        """Colorlist number of every token in one layer (-1 = remainder), cached like in MPAC."""
        k = (seed, layer)
        if k in self._color_cache:
            self._color_cache.move_to_end(k)
            return self._color_cache[k]
        perm = np.random.default_rng(hash_to_int("perm", self.key, seed, layer)).permutation(self.vocab_size)
        colors = np.full(self.vocab_size, -1, dtype=np.int16)
        colors[perm[:self.radix * self.list_size]] = np.repeat(np.arange(self.radix), self.list_size)
        self._color_cache[k] = colors
        if len(self._color_cache) > 256 * self.num_layers:
            self._color_cache.popitem(last=False)
        return colors

    # the inherited tournament and counting call g_value(token, layer, digit_seed(seed, d))
    def digit_seed(self, seed, digit):
        return seed, int(digit)

    def g_value(self, token, layer, seed):
        seed, digit = seed
        return int(self.layer_colors(seed, layer)[int(token)] == digit)

    def z_score(self, w, n):
        if n == 0:
            return 0.0
        total, g = n * self.num_layers, self.gamma_eff
        return (w - g * total) / sqrt(total * g * (1 - g))

    def max_cell_cdf(self, n):
        return MPACWatermark.max_cell_cdf(self, n * self.num_layers)
