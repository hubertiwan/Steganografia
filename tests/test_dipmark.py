"""Correctness tests for DiPmark (no model needed).

Run with:  pytest -q
"""
import numpy as np

from watermarks.dipmark import DiPmarkWatermark

V = 2000


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def fake_generate(wm, length, rng):
    """Fake 'model': random logits at every step, sampled with or without the watermark."""
    tokens = [int(rng.integers(V))]  # start token, like the end of a prompt
    for _ in range(length):
        p = softmax(rng.normal(size=V) * 3)
        if wm is not None:
            p = wm.reweight(p, tokens)
        tokens.append(int(rng.choice(V, p=p)))
    return tokens


def test_permutation_is_deterministic_and_depends_on_context():
    a = DiPmarkWatermark(key=1, vocab_size=V)
    b = DiPmarkWatermark(key=1, vocab_size=V)
    c = DiPmarkWatermark(key=2, vocab_size=V)
    assert np.array_equal(a.permutation([5]), b.permutation([5]))      # same key -> same cipher
    assert not np.array_equal(a.permutation([5]), c.permutation([5]))  # different key
    assert not np.array_equal(a.permutation([5]), a.permutation([6]))  # different context


def test_reweight_is_a_distribution():
    wm = DiPmarkWatermark(key=1, vocab_size=V)
    p = softmax(np.random.default_rng(0).normal(size=V) * 3)
    q = wm.reweight(p, [7])
    assert np.all(q >= 0)
    assert abs(q.sum() - 1) < 1e-9


def test_alpha_zero_changes_nothing():
    wm = DiPmarkWatermark(key=1, vocab_size=V, alpha=0.0)
    p = softmax(np.random.default_rng(1).normal(size=V))
    assert np.allclose(wm.reweight(p, [7]), p)


def test_theorem_4_3_distribution_preserving():
    """Averaged over random keys, the reweighted distribution equals the original one."""
    small_v = 30
    p = np.random.default_rng(2).dirichlet(np.ones(small_v))
    total = np.zeros(small_v)
    keys = 5000
    for k in range(keys):
        wm = DiPmarkWatermark(key=k, vocab_size=small_v, alpha=0.45)
        total += wm.reweight(p, [0])
    average = total / keys
    assert np.max(np.abs(average - p)) < 0.01


def test_reweight_moves_mass_to_green_list():
    """For a flat distribution the green half gets 0.5 + alpha of the mass."""
    wm = DiPmarkWatermark(key=3, vocab_size=V, alpha=0.45, gamma=0.5)
    p = np.ones(V) / V
    q = wm.reweight(p, [11])
    mask = wm.green_mask([11])
    assert abs(q[mask].sum() - 0.95) < 0.01


def test_false_positive_rate_without_watermark():
    """Text without a watermark and random keys: about 1% or less is flagged at fpr = 1%."""
    small_v = 300  # smaller vocabulary -> faster permutations, same statistics
    rng = np.random.default_rng(4)
    text = rng.choice(small_v, size=200, replace=False)
    flagged = 0
    keys = 1000
    for k in range(keys):
        if DiPmarkWatermark(key=k, vocab_size=small_v).detect(text, fpr=0.01)["watermarked"]:
            flagged += 1
    assert flagged / keys < 0.02


def test_watermarked_text_is_detected():
    rng = np.random.default_rng(5)
    wm = DiPmarkWatermark(key=42, vocab_size=V)
    other = DiPmarkWatermark(key=43, vocab_size=V)
    watermarked = fake_generate(wm, 200, rng)
    plain = fake_generate(None, 200, rng)
    assert wm.detect(watermarked)["watermarked"]
    assert not wm.detect(plain)["watermarked"]
    assert not other.detect(watermarked)["watermarked"]


def test_repeated_contexts_are_scored_once():
    wm = DiPmarkWatermark(key=1, vocab_size=V, context_size=1)
    tokens = [1, 2, 3] * 50
    phi, green, n = wm.score(tokens)
    assert n == 3   # only 3 different texture keys


def test_short_text_gives_zero():
    wm = DiPmarkWatermark(key=1, vocab_size=V)
    assert wm.score([5]) == (0.0, 0, 0)
    assert not wm.detect([5])["watermarked"]
