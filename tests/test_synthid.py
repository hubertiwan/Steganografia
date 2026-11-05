"""Correctness tests for the SynthID-Text Tournament watermark (no model needed).

Run with:  pytest -q
"""
import numpy as np

from synthid_watermark import TournamentWatermark

V = 1000


def generate_tokens(wm, probs, length, rng):
    """Fake 'model' with a fixed distribution: generate length tokens with Tournament sampling."""
    tokens = list(rng.choice(V, size=wm.context_size))  # random start, like a prompt
    for _ in range(length):
        context = tuple(tokens[-wm.context_size:])
        tokens.append(wm.tournament_sample(probs, context, rng))
    return tokens


def test_seed_and_g_values_are_deterministic():
    a = TournamentWatermark(key="abc")
    b = TournamentWatermark(key="abc")
    c = TournamentWatermark(key="xyz")
    context = (1, 2, 3, 4)
    assert a.seed(context) == b.seed(context)   # same key -> same seed
    assert a.seed(context) != c.seed(context)   # different key -> different seed

    seed = a.seed(context)
    g_a = []
    g_c = []
    for token in range(200):
        g_a.append(a.g_value(token, 1, seed))
        g_c.append(c.g_value(token, 1, seed))
    assert g_a != g_c
    assert 0.35 < np.mean(g_a) < 0.65             # g is a fair coin


def test_seed_uses_only_last_h_tokens():
    wm = TournamentWatermark(key=1, context_size=4)
    assert wm.seed((9, 9, 1, 2, 3, 4)) == wm.seed((1, 2, 3, 4))


def test_one_hot_distribution_is_not_changed():
    """If the model is sure about one token, the watermark must not change it."""
    wm = TournamentWatermark(key=1)
    rng = np.random.default_rng(0)
    probs = np.zeros(V)
    probs[42] = 1.0
    for i in range(20):
        assert wm.tournament_sample(probs, (i, 1, 2, 3), rng) == 42


def test_tournament_raises_mean_g():
    """For a flat distribution each layer picks g_l = 1 with probability 3/4."""
    wm = TournamentWatermark(key=5)
    rng = np.random.default_rng(1)
    probs = np.ones(V) / V
    g_values = []
    for i in range(300):
        context = (i, i + 1, i + 2, i + 3)
        token = wm.tournament_sample(probs, context, rng)
        seed = wm.seed(context)
        for layer in range(1, wm.num_layers + 1):
            g_values.append(wm.g_value(token, layer, seed))
    assert abs(np.mean(g_values) - 0.75) < 0.03


def test_watermarked_text_is_detected():
    wm = TournamentWatermark(key=7)
    rng = np.random.default_rng(2)
    probs = np.ones(V) / V
    tokens = generate_tokens(wm, probs, 200, rng)
    assert wm.detect(tokens)["watermarked"]

    plain = list(rng.choice(V, size=200, p=probs))
    assert not wm.detect(plain)["watermarked"]


def test_z_score_null_distribution():
    """Text without a watermark and a random key: z has mean ~0 and variance ~1."""
    rng = np.random.default_rng(3)
    text = list(rng.choice(V, size=100))
    zs = []
    for k in range(1000):
        zs.append(TournamentWatermark(key=k).z_score(text)[0])
    assert abs(np.mean(zs)) < 0.1
    assert 0.85 < np.var(zs) < 1.15


def test_repeated_contexts_are_scored_once():
    wm = TournamentWatermark(key=1, context_size=4)
    tokens = [1, 2, 3, 4, 5] * 40
    z, n = wm.z_score(tokens)
    assert n == 5   # only 5 different windows of 4 tokens in the text


def test_short_text_gives_zero():
    wm = TournamentWatermark(key=1, context_size=4)
    assert wm.z_score([1, 2, 3]) == (0.0, 0)
