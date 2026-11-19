"""Correctness tests for the KGW watermark (no model needed).

Run with:  pytest -q
"""
import numpy as np

from kgw_watermark import KGWWatermark

V = 2000


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def fake_generate(wm, length, rng):
    """Fake 'model': random logits at every step, sampled with or without the watermark."""
    tokens = [int(rng.integers(V))]  # start token, like the end of a prompt
    for _ in range(length):
        logits = rng.normal(size=V) * 3
        if wm is not None:
            logits = wm.watermark_logits(logits, tokens[-1])
        tokens.append(int(rng.choice(V, p=softmax(logits))))
    return tokens


def test_green_list_size_and_determinism():
    a = KGWWatermark(key=123, vocab_size=V, gamma=0.25)
    b = KGWWatermark(key=123, vocab_size=V, gamma=0.25)
    c = KGWWatermark(key=124, vocab_size=V, gamma=0.25)
    assert a.green_mask(5).sum() == int(0.25 * V)
    assert np.array_equal(a.green_mask(5), b.green_mask(5))       # same key -> same list
    assert not np.array_equal(a.green_mask(5), c.green_mask(5))   # different key -> different list


def test_green_list_depends_on_previous_token():
    wm = KGWWatermark(key=1, vocab_size=V)
    assert not np.array_equal(wm.green_mask(10), wm.green_mask(11))


def test_watermark_logits_adds_delta_only_to_green():
    wm = KGWWatermark(key=1, vocab_size=V, delta=2.0)
    logits = np.random.default_rng(0).normal(size=V)
    mask = wm.green_mask(7)
    diff = wm.watermark_logits(logits, 7) - logits
    assert np.allclose(diff[mask], 2.0)
    assert np.allclose(diff[~mask], 0.0)


def test_z_score_null_distribution():
    """Text without repeats and a random key: z has mean ~0 and variance ~1."""
    rng = np.random.default_rng(2)
    text = rng.choice(V, size=200, replace=False)
    zs = []
    for k in range(500):
        zs.append(KGWWatermark(k, V).z_score(text)[0])
    assert abs(np.mean(zs)) < 0.15
    assert 0.8 < np.var(zs) < 1.2


def test_watermarked_text_is_detected():
    rng = np.random.default_rng(3)
    wm = KGWWatermark(key=42, vocab_size=V)
    watermarked = fake_generate(wm, 200, rng)
    plain = fake_generate(None, 200, rng)
    assert wm.detect(watermarked)["watermarked"]
    assert not wm.detect(plain)["watermarked"]


def test_other_key_does_not_detect():
    rng = np.random.default_rng(4)
    wm = KGWWatermark(key=42, vocab_size=V)
    other = KGWWatermark(key=43, vocab_size=V)
    watermarked = fake_generate(wm, 200, rng)
    assert not other.detect(watermarked)["watermarked"]


def test_short_text_gives_zero():
    wm = KGWWatermark(key=1, vocab_size=V)
    assert wm.z_score([5]) == (0.0, 0)
