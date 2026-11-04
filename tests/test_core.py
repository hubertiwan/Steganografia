"""Correctness tests: checks the implementation against the theorems in the paper.

Run with:  pytest -q
"""
import numpy as np
import pytest

from unigram_watermark import UnigramWatermark

V = 5000


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def renyi(p, q, a):
    m = (p > 0) & (q > 0)
    if a == 1:  # alpha = 1 -> KL divergence
        return float(np.sum(p[m] * np.log(p[m] / q[m])))
    return float(np.log(np.sum(p[m] ** a * q[m] ** (1 - a))) / (a - 1))


def test_green_list_size_and_determinism():
    a = UnigramWatermark(key=123, vocab_size=V, gamma=0.5)
    b = UnigramWatermark(key=123, vocab_size=V, gamma=0.5)
    c = UnigramWatermark(key=124, vocab_size=V, gamma=0.5)
    assert a.green_mask.sum() == int(0.5 * V)
    assert np.array_equal(a.green_mask, b.green_mask)       # same key -> same list
    assert not np.array_equal(a.green_mask, c.green_mask)   # different key -> different list


def test_watermark_logits_adds_delta_only_to_green():
    wm = UnigramWatermark(key=1, vocab_size=V, delta=2.0)
    logits = np.random.default_rng(0).normal(size=V)
    diff = wm.watermark_logits(logits) - logits
    assert np.allclose(diff[wm.green_mask], 2.0)
    assert np.allclose(diff[~wm.green_mask], 0.0)


@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0, 4.0])
@pytest.mark.parametrize("a", [0.5, 1, 2, 5])
def test_theorem_3_1_quality(delta, a):
    """max(D_a(p_hat || p), D_a(p || p_hat)) <= min(delta, a * delta^2 / 8)."""
    rng = np.random.default_rng(0)
    wm = UnigramWatermark(key=1, vocab_size=V, delta=delta)
    bound = min(delta, a * delta ** 2 / 8)
    for _ in range(100):
        logits = rng.normal(size=V) * rng.uniform(0.1, 8)  # from flat to very peaked
        p, ph = softmax(logits), softmax(wm.watermark_logits(logits))
        assert max(renyi(ph, p, a), renyi(p, ph, a)) <= bound + 1e-9


@pytest.mark.parametrize("repeats", [False, True])
def test_type_one_error_unique_detector(repeats):
    """For a fixed unwatermarked text and a random key, the Unique detector with the
    Eq. (9) threshold should give a false positive rate of about alpha (cf. Fig. 5)."""
    rng = np.random.default_rng(1)
    if repeats:
        text = rng.choice(rng.choice(V, size=40, replace=False), 200)
    else:
        text = rng.choice(V, size=200, replace=False)
    alpha, keys = 0.05, 4000
    fp = np.mean([UnigramWatermark(k, V).detect_unique(text, alpha)["watermarked"] for k in range(keys)])
    assert abs(fp - alpha) < 0.015


def test_z_score_null_distribution():
    """Text without repeats and a random key: z has mean ~0 and variance ~1."""
    rng = np.random.default_rng(2)
    text = rng.choice(V, size=200, replace=False)
    zs = [UnigramWatermark(k, V).z_score(text)[0] for k in range(3000)]
    assert abs(np.mean(zs)) < 0.1
    assert 0.85 < np.var(zs) < 1.15


@pytest.mark.parametrize("gamma", [0.25, 0.5])
def test_theorem_3_7_robustness(gamma):
    """After eta edits the statistic drops by at most max{(1+g/2)eta/sqrt(n), (1-g/2)eta/sqrt(n-eta)}.

    Note: the proof (Appendix C.2) uses f = (|y|_G - g*n)/sqrt(n), i.e. the z-score
    WITHOUT the sqrt(g(1-g)) factor of Eq. (2). For the Eq. (2) z-score the bound must
    therefore be divided by sqrt(g(1-g)). Both versions are checked here."""
    rng = np.random.default_rng(3)
    n = 200
    wm = UnigramWatermark(key=7, vocab_size=V, gamma=gamma, delta=2.0)
    g = wm.gamma_eff

    def f_stat(t):
        return (wm.green_mask[np.asarray(t)].sum() - g * len(t)) / np.sqrt(len(t))

    for _ in range(500):
        y = list(rng.choice(V, size=n, p=softmax(wm.watermark_logits(rng.normal(size=V) * 3))))
        u, eta = list(y), int(rng.integers(1, n // 2))
        for _ in range(eta):  # eta random edits -> edit distance <= eta
            op, pos = rng.integers(3), int(rng.integers(len(u)))
            if op == 0:
                u.insert(pos, int(rng.integers(V)))
            elif op == 1:
                u.pop(pos)
            else:
                u[pos] = int(rng.integers(V))
        bound = max((1 + g / 2) * eta / np.sqrt(n), (1 - g / 2) * eta / np.sqrt(n - eta))
        assert f_stat(u) >= f_stat(y) - bound - 1e-9
        assert wm.z_score(u)[0] >= wm.z_score(y)[0] - bound / np.sqrt(g * (1 - g)) - 1e-9


def test_unique_detector_ignores_repeated_tokens():
    """Repeating one green token gives a huge z-score in Algorithm 2 but not in Algorithm 4."""
    wm = UnigramWatermark(key=9, vocab_size=V)
    green_tok = int(np.flatnonzero(wm.green_mask)[0])
    red_tok = int(np.flatnonzero(~wm.green_mask)[0])
    spam = [green_tok] * 150 + [red_tok] * 50
    assert wm.detect(spam, tau=4.0)["watermarked"]
    assert not wm.detect_unique(spam)["watermarked"]
