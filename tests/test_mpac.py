"""Correctness tests for the MPAC multi-bit watermark (no model needed).

Run with:  pytest -q
"""
import numpy as np
import pytest

from watermarks.mpac import MPACWatermark, bit_accuracy, digits_to_message, message_to_digits

V = 2000


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def fake_generate(wm, message, length, rng):
    """Fake 'model': random logits at every step, sampled with or without the watermark."""
    digits = wm.check_message(message) if wm is not None else None
    tokens = [int(rng.integers(V))]  # start token, like the end of a prompt
    for _ in range(length):
        logits = rng.normal(size=V) * 3
        if wm is not None:
            logits = wm.watermark_logits(logits, tokens[-wm.context_size:], digits)
        tokens.append(int(rng.choice(V, p=softmax(logits))))
    return tokens


def test_radix_and_number_of_positions():
    assert (MPACWatermark(1, V, 8, gamma=0.25).radix, MPACWatermark(1, V, 8, gamma=0.25).positions) == (4, 4)
    assert (MPACWatermark(1, V, 8, gamma=0.5).radix, MPACWatermark(1, V, 8, gamma=0.5).positions) == (2, 8)
    assert MPACWatermark(1, V, 16, gamma=0.125).radix == 8
    wm = MPACWatermark(1, V, 8, gamma=0.3)  # r = 3, 3**6 = 729 >= 256
    assert wm.radix == 3 and wm.positions == 6
    assert MPACWatermark(1, V, 8, gamma=0.2).radix == 5  # 1/0.2 is 4.999... in floating point


@pytest.mark.parametrize("radix,bits", [(2, 8), (4, 8), (4, 7), (3, 8), (8, 16), (4, 32)])
def test_message_digits_round_trip(radix, bits):
    wm = MPACWatermark(1, V, bits, gamma=1 / radix)
    rng = np.random.default_rng(radix * 100 + bits)
    for _ in range(20):
        msg = "".join(rng.choice(["0", "1"], size=bits))
        digits = message_to_digits(msg, radix, wm.positions)
        assert len(digits) == wm.positions and all(0 <= d < radix for d in digits)
        assert digits_to_message(digits, radix, bits) == msg


def test_invalid_arguments():
    with pytest.raises(ValueError):
        MPACWatermark(1, V, gamma=0.6)  # only one colorlist
    with pytest.raises(ValueError):
        MPACWatermark(1, V, message_bits=0)
    wm = MPACWatermark(1, V, message_bits=8)
    for bad in ["0101", "0101010a", "010101010"]:
        with pytest.raises(ValueError):
            wm.check_message(bad)


def test_colorlists_are_disjoint_and_deterministic():
    a = MPACWatermark(key=7, vocab_size=V, gamma=0.25)
    b = MPACWatermark(key=7, vocab_size=V, gamma=0.25)
    c = MPACWatermark(key=8, vocab_size=V, gamma=0.25)
    pa, perm_a = a.position_and_permutation([5])
    pb, perm_b = b.position_and_permutation([5])
    assert pa == pb and np.array_equal(perm_a, perm_b)            # same key -> same lists
    assert not np.array_equal(perm_a, c.position_and_permutation([5])[1])
    lists = [set(a.colorlist(perm_a, m).tolist()) for m in range(a.radix)]
    assert all(len(s) == int(0.25 * V) for s in lists)
    assert len(set().union(*lists)) == a.radix * a.list_size      # no token in two lists


def test_watermark_logits_adds_delta_only_to_selected_colorlist():
    wm = MPACWatermark(key=3, vocab_size=V, message_bits=8, gamma=0.25, delta=2.0)
    digits = wm.check_message("10110010")  # radix 4 -> digits [2, 3, 0, 2]
    assert digits == [2, 3, 0, 2]
    logits = np.zeros(V)
    out = wm.watermark_logits(logits, [11], digits)
    position, perm = wm.position_and_permutation([11])
    expected = np.zeros(V)
    expected[wm.colorlist(perm, digits[position])] = 2.0
    assert np.array_equal(out, expected)
    assert np.array_equal(logits, np.zeros(V))  # input not modified


def test_positions_are_spread_over_the_message():
    wm = MPACWatermark(key=1, vocab_size=V, message_bits=8)
    counts = np.zeros(wm.positions)
    for prev in range(2000):
        counts[wm.position_and_permutation([prev])[0]] += 1
    assert counts.min() > 0.8 * 2000 / wm.positions  # each of the 4 positions gets ~25%


@pytest.mark.parametrize("message", ["10110010", "00000000", "11111111"])
def test_message_is_decoded_from_watermarked_text(message):
    rng = np.random.default_rng(int(message, 2))
    wm = MPACWatermark(key=42, vocab_size=V, message_bits=8)
    tokens = fake_generate(wm, message, 200, rng)
    d = wm.detect(tokens)
    assert d["message"] == message
    assert d["watermarked"] and d["p_value"] < 1e-6
    assert d["n"] == 200 and d["empty_positions"] == 0


def test_longer_message_and_robustness_to_edits():
    from watermarks.unigram import random_edits
    rng = np.random.default_rng(9)
    wm = MPACWatermark(key=5, vocab_size=V, message_bits=16)
    msg = "1100101011110000"
    tokens = fake_generate(wm, msg, 300, rng)
    assert bit_accuracy(msg, wm.detect(tokens)["message"]) == 1.0
    edited = random_edits(tokens, 0.2, V, np.random.default_rng(1))
    d = wm.detect(edited)
    assert d["watermarked"]
    assert bit_accuracy(msg, d["message"]) >= 0.85


def test_plain_text_and_wrong_key_are_not_detected():
    rng = np.random.default_rng(11)
    wm = MPACWatermark(key=42, vocab_size=V)
    other = MPACWatermark(key=43, vocab_size=V)
    watermarked = fake_generate(wm, "01011100", 200, rng)
    plain = fake_generate(None, None, 200, rng)
    assert not wm.detect(plain)["watermarked"]
    assert not other.detect(watermarked)["watermarked"]


def test_max_cell_cdf_matches_monte_carlo():
    """The exact null distribution of max W[p] against direct multinomial sampling."""
    wm = MPACWatermark(key=1, vocab_size=1003, gamma=0.25)  # 1003 is not divisible by 4 -> remainder cell
    rng = np.random.default_rng(0)
    g = wm.gamma_eff
    probs = [g] * wm.radix + [1 - wm.radix * g]
    for n in (1, 5, 20, 60):
        F = wm.max_cell_cdf(n)
        samples = rng.multinomial(n, probs, size=200_000)[:, :wm.radix].max(axis=1)
        empirical = np.array([np.mean(samples <= a) for a in range(n + 1)])
        assert np.max(np.abs(F - empirical)) < 0.005, n
        assert F[-1] == 1.0 and np.all(np.diff(F) >= 0)


def test_false_positive_rate_and_bias_of_paper_z():
    """Random text, random keys: the exact p-value keeps the false positive rate at alpha,
    while the paper's z-score is shifted above 0 (Fig. 8 / Appendix A.2 of the paper)."""
    small_v = 400
    rng = np.random.default_rng(4)
    text = list(rng.integers(small_v, size=150))
    zs, flagged = [], 0
    keys = 400
    for k in range(keys):
        d = MPACWatermark(key=k, vocab_size=small_v, message_bits=8).detect(text, alpha=0.05)
        zs.append(d["z"])
        flagged += d["watermarked"]
    assert flagged / keys < 0.09
    assert np.mean(zs) > 1.0  # biased: a z threshold of 2 would flag far more than 5%


def test_short_text():
    wm = MPACWatermark(key=1, vocab_size=V)
    d = wm.detect([5])
    assert d["n"] == 0 and not d["watermarked"] and d["p_value"] == 1.0
