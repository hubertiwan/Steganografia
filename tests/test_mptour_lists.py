"""Correctness tests for MPTour-Lists (variant A: MPAC colorlists as g-functions, no model needed).

Run with:  pytest -q
"""
import numpy as np
import pytest

from test_mptour import V, fake_generate
from watermarks.mptour import bit_accuracy
from watermarks.mptour_lists import MPTourListsWatermark


def test_radix_from_gamma_and_disjoint_lists():
    wm = MPTourListsWatermark(1, V, gamma=0.25)
    assert (wm.radix, wm.positions, wm.list_size) == (4, 4, 250)
    seed = wm.seed((1, 2, 3, 4))
    colors = wm.layer_colors(seed, 1)
    assert [int(np.sum(colors == d)) for d in range(4)] == [250] * 4
    assert not np.array_equal(colors, wm.layer_colors(seed, 2))  # new permutation per layer
    g = [[wm.g_value(x, 1, wm.digit_seed(seed, d)) for d in range(4)] for x in range(V)]
    assert max(sum(row) for row in g) == 1                         # green for at most one digit
    with pytest.raises(ValueError):
        MPTourListsWatermark(1, V, gamma=0.6)


def test_distribution_is_preserved_on_average_over_keys():
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    rng = np.random.default_rng(0)
    for digit in (0, 1):
        counts = np.zeros(4)
        for k in range(4000):
            wm = MPTourListsWatermark(key=k, vocab_size=4, gamma=0.5, num_layers=3)
            counts[wm.tournament(probs, wm.digit_seed(wm.seed((1, 2, 3, 4)), digit), rng)] += 1
        assert np.max(np.abs(counts / counts.sum() - probs)) < 0.03, digit


@pytest.mark.parametrize("message", ["10110010", "00000000"])
def test_message_is_decoded_and_plain_text_is_not_detected(message):
    rng = np.random.default_rng(int(message, 2))
    wm = MPTourListsWatermark(key=42, vocab_size=V)
    d = wm.detect(fake_generate(wm, message, 200, rng))
    assert d["message"] == message and d["watermarked"] and d["p_value"] < 1e-6
    assert not wm.detect(fake_generate(None, None, 200, rng))["watermarked"]
    assert bit_accuracy(message, d["message"]) == 1.0


def test_max_cell_cdf_matches_monte_carlo():
    wm = MPTourListsWatermark(key=1, vocab_size=1003, gamma=0.25, num_layers=4)  # remainder cell
    g = wm.gamma_eff
    rng = np.random.default_rng(0)
    for n in (1, 5):
        N = n * wm.num_layers
        F = wm.max_cell_cdf(n)
        samples = rng.multinomial(N, [g] * 4 + [1 - 4 * g], size=200_000)[:, :4].max(axis=1)
        empirical = np.array([np.mean(samples <= a) for a in range(N + 1)])
        assert np.max(np.abs(F - empirical)) < 0.005, n


def test_false_positive_rate():
    rng = np.random.default_rng(4)
    text = list(rng.integers(400, size=120))
    keys = 200
    flagged = sum(MPTourListsWatermark(key=k, vocab_size=400, num_layers=4).detect(text, alpha=0.05)["watermarked"]
                  for k in range(keys))
    assert flagged / keys < 0.1
