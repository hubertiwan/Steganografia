"""Correctness tests for the MPTour multi-bit watermark (MPAC + SynthID Tournament, no model needed).

Run with:  pytest -q
"""
import numpy as np
import pytest

from watermarks.mptour import MPTourWatermark, bit_accuracy

V = 1000


def fake_generate(wm, message, length, rng):
    """Fake 'model': random logits at every step (as in test_mpac.py), sampled with Tournament
    sampling, or plainly if wm is None. Repeated contexts are sampled plainly, as in generate()."""
    digits = wm.check_message(message) if wm is not None else None
    tokens = [int(t) for t in rng.integers(V, size=4)]  # random start, like a prompt
    seen = set()
    for _ in range(length):
        logits = rng.normal(size=V) * 3
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        context = tuple(tokens[-4:])
        if wm is None or context in seen:
            tokens.append(int(rng.choice(V, p=probs)))
        else:
            seen.add(context)
            tokens.append(wm.tournament_sample(probs, context, digits, rng))
    return tokens


def test_radix_and_number_of_positions():
    assert MPTourWatermark(1, 8, radix=4).positions == 4
    assert MPTourWatermark(1, 8, radix=2).positions == 8
    assert MPTourWatermark(1, 8, radix=3).positions == 6  # 3**6 = 729 >= 256
    assert MPTourWatermark(1, 16, radix=16).positions == 4


def test_invalid_arguments():
    with pytest.raises(ValueError):
        MPTourWatermark(1, radix=1)
    with pytest.raises(ValueError):
        MPTourWatermark(1, message_bits=0)
    wm = MPTourWatermark(1)
    for bad in ["0101", "0101010a", "010101010"]:
        with pytest.raises(ValueError):
            wm.check_message(bad)


def test_digits_have_independent_fair_g_values():
    wm = MPTourWatermark(key=3)
    seed = wm.seed((1, 2, 3, 4))
    g = np.array([[wm.g_value(x, 1, wm.digit_seed(seed, d)) for x in range(400)] for d in range(wm.radix)])
    assert np.all(np.abs(g.mean(axis=1) - 0.5) < 0.1)          # fair coin for every digit
    assert abs(np.mean(g[0] == g[1]) - 0.5) < 0.1                # digit tables overlap ~50%, not equal


def test_distribution_is_preserved_on_average_over_keys():
    """Tournament output, averaged over keys, has the model distribution for any digit."""
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    rng = np.random.default_rng(0)
    for digit in (0, 3):
        counts = np.zeros(4)
        for k in range(4000):
            wm = MPTourWatermark(key=k, num_layers=3)
            counts[wm.tournament(probs, wm.digit_seed(wm.seed((1, 2, 3, 4)), digit), rng)] += 1
        assert np.max(np.abs(counts / counts.sum() - probs)) < 0.03, digit


@pytest.mark.parametrize("message", ["10110010", "00000000", "11111111"])
def test_message_is_decoded_from_watermarked_text(message):
    rng = np.random.default_rng(int(message, 2))
    wm = MPTourWatermark(key=42)
    tokens = fake_generate(wm, message, 200, rng)
    d = wm.detect(tokens)
    assert d["message"] == message
    assert d["watermarked"] and d["p_value"] < 1e-6
    assert d["n"] == 200 and d["empty_positions"] == 0
    z, p = wm.verify(tokens, message)
    assert z > 5 and p < 1e-6


def test_longer_message_and_robustness_to_edits():
    from watermarks.unigram import random_edits
    rng = np.random.default_rng(9)
    wm = MPTourWatermark(key=5, message_bits=16)
    msg = "1100101011110000"
    tokens = fake_generate(wm, msg, 300, rng)
    assert bit_accuracy(msg, wm.detect(tokens)["message"]) == 1.0
    d = wm.detect(random_edits(tokens, 0.2, V, np.random.default_rng(1)))
    assert d["watermarked"]
    assert bit_accuracy(msg, d["message"]) >= 0.85


def test_plain_text_and_wrong_key_are_not_detected():
    rng = np.random.default_rng(11)
    wm = MPTourWatermark(key=42)
    watermarked = fake_generate(wm, "01011100", 200, rng)
    plain = fake_generate(None, None, 200, rng)
    assert not wm.detect(plain)["watermarked"]
    assert not MPTourWatermark(key=43).detect(watermarked)["watermarked"]
    assert abs(wm.verify(plain, "01011100")[0]) < 3


def test_max_cell_cdf_matches_monte_carlo():
    wm = MPTourWatermark(key=1, num_layers=4)
    rng = np.random.default_rng(0)
    for n in (1, 5, 20):
        F = wm.max_cell_cdf(n)
        samples = rng.binomial(n * wm.num_layers, 0.5, size=(200_000, wm.radix)).max(axis=1)
        empirical = np.array([np.mean(samples <= a) for a in range(n * wm.num_layers + 1)])
        assert np.max(np.abs(F - empirical)) < 0.005, n
        assert F[-1] == 1.0 and np.all(np.diff(F) >= 0)


def test_false_positive_rate():
    """Random text, random keys: the exact p-value keeps the false positive rate at alpha."""
    rng = np.random.default_rng(4)
    text = list(rng.integers(V, size=120))
    keys = 300
    flagged = sum(MPTourWatermark(key=k, num_layers=4).detect(text, alpha=0.05)["watermarked"] for k in range(keys))
    assert flagged / keys < 0.09


def test_short_text():
    d = MPTourWatermark(key=1).detect([5, 6, 7])
    assert d["n"] == 0 and not d["watermarked"] and d["p_value"] == 1.0
