"""Tests of the application logic (app/), with a fake language model.

Round trip for every algorithm: prompt + key -> generation with the watermark -> detection.
The algorithms are probabilistic, so the checks are: detected in watermarked text, not detected
in plain text and not detected with a wrong key (fixed seeds, large margins). For MPAC also:
the decoded message equals the embedded one.
"""
import math

import numpy as np
import pytest

from app import algorithms, experiments, history, metrics
from app.errors import AppError, Cancelled

IDS = list(algorithms.ALGORITHMS)


def defaults(algo_id):
    return algorithms.get(algo_id).defaults()


def test_all_repository_algorithms_are_listed():
    assert IDS == ["unigram", "synthid", "kgw", "dipmark", "mpac"]
    from watermarks import dipmark, kgw, mpac, synthid, unigram
    classes = [unigram.UnigramWatermark, synthid.TournamentWatermark, kgw.KGWWatermark,
               dipmark.DiPmarkWatermark, mpac.MPACWatermark]
    for algo_id, cls in zip(IDS, classes):
        assert isinstance(algorithms.get(algo_id).build(defaults(algo_id), 1000), cls)


def test_defaults_match_the_demo_scripts():
    """Default parameters must be the ones in the demo scripts (taken from the papers)."""
    import sys
    from demos import demo_dipmark, demo_kgw, demo_mpac, demo_synthid, demo_unigram
    argv, sys.argv = sys.argv, ["demo"]
    try:
        u, s, k, d, m = (mod.parse_args() for mod in (demo_unigram, demo_synthid, demo_kgw, demo_dipmark, demo_mpac))
    finally:
        sys.argv = argv
    assert defaults("unigram") == {"key": u.key, "gamma": u.gamma, "delta": u.delta, "tau": u.tau,
                                   "temperature": u.temperature, "top_p": u.top_p}
    assert defaults("synthid") == {"key": s.key, "context_size": s.context_size, "num_layers": s.num_layers,
                                   "threshold": s.threshold, "temperature": s.temperature, "top_k": s.top_k}
    assert defaults("kgw") == {"key": k.key, "gamma": k.gamma, "delta": k.delta, "z_threshold": k.z_threshold,
                               "temperature": k.temperature, "top_p": k.top_p}
    assert defaults("dipmark") == {"key": d.key, "alpha": d.alpha, "gamma": d.gamma, "context_size": d.context_size,
                                   "fpr": d.fpr, "temperature": d.temperature, "top_p": d.top_p}
    p = defaults("mpac")
    assert (p["key"], len(p["message"]), p["gamma"], p["delta"], p["context_size"], p["alpha"], p["temperature"],
            p["top_p"]) == (m.key, m.bits, m.gamma, m.delta, m.context_size, m.alpha, m.temperature, m.top_p)


@pytest.mark.parametrize("algo_id,bad", [
    ("unigram", {"gamma": 0}), ("unigram", {"gamma": 1}), ("unigram", {"key": "abc"}), ("unigram", {"top_p": 0}),
    ("synthid", {"key": "  "}), ("synthid", {"num_layers": 11}), ("synthid", {"temperature": 0}),
    ("kgw", {"delta": -1}), ("dipmark", {"alpha": 0.6}), ("dipmark", {"fpr": 0}),
    ("mpac", {"message": "0102"}), ("mpac", {"message": ""}), ("mpac", {"message": "1" * 65}),
    ("mpac", {"gamma": 0.6}), ("mpac", {"nothing": 1}),
])
def test_bad_parameters_are_rejected(algo_id, bad):
    with pytest.raises(AppError) as e:
        algorithms.get(algo_id).check(bad)
    assert e.value.message


@pytest.mark.parametrize("algo_id", IDS)
def test_demo_round_trip(algo_id, fake_lm):
    r = experiments.run_demo(fake_lm, algo_id, defaults(algo_id), "Why is the sky blue?", 120, seed=1)
    wm, plain = r["watermarked"], r["plain"]
    assert len(wm["tokens"]) == 120
    assert wm["detection"]["detected"], wm["detection"]
    assert not plain["detection"]["detected"]
    assert not wm["other_key"]["detected"]
    assert len(wm["pieces"]) == len(wm["annotation"]) == len(wm["tokens"])
    assert "".join(wm["pieces"]) == wm["text"]
    assert wm["perplexity"] > 1 and plain["perplexity"] > 1
    if algo_id == "mpac":
        assert wm["detection"]["message"] == "10110010" and wm["detection"]["bit_accuracy"] == 1.0


@pytest.mark.parametrize("algo_id", IDS)
def test_series_round_trip(algo_id, fake_lm):
    s = {"prompts": experiments.PROMPTS[:6], "max_new_tokens": 100, "perplexity": False}
    r = experiments.run_test(fake_lm, algo_id, defaults(algo_id), s)
    summary = r["summary"]
    assert summary["tpr"]["rate"] == 1.0
    assert summary["fpr"]["rate"] == 0.0
    assert summary["other_key"]["rate"] == 0.0
    assert summary["auc"] == 1.0
    rates = [e["detected"]["rate"] for e in r["edits"]]
    assert rates[0] == 1.0 and rates[-1] <= rates[0]
    if algo_id == "mpac":
        assert summary["bit_accuracy"] == 1.0
        assert r["edits"][-1]["bit_accuracy"] < 1.0


def test_no_watermark_when_delta_is_zero(fake_lm):
    """Control: delta = 0 turns the KGW watermark off, so nothing may be detected.
    This shows that the detections above come from the algorithm."""
    s = {"prompts": experiments.PROMPTS[:6], "max_new_tokens": 100, "perplexity": False, "edit_fractions": [0]}
    summary = experiments.run_test(fake_lm, "kgw", dict(defaults("kgw"), delta=0.0), s)["summary"]
    assert summary["tpr"]["rate"] == 0.0


def test_detect_text_matches_demo(fake_lm):
    demo = experiments.run_demo(fake_lm, "kgw", defaults("kgw"), "Hello", 80, 3)
    det = experiments.detect_text(fake_lm, "kgw", defaults("kgw"), demo["watermarked"]["text"])
    assert det["detection"]["score"] == pytest.approx(demo["watermarked"]["detection"]["score"])


def test_annotations_agree_with_detectors(fake_lm):
    for algo_id in IDS:
        demo = experiments.run_demo(fake_lm, algo_id, defaults(algo_id), "Hello", 80, 2)
        ann = [a for a in demo["watermarked"]["annotation"] if a is not None]
        d = demo["watermarked"]["detection"]
        assert len(ann) == d["n"], algo_id
        if algo_id in ("unigram", "kgw", "dipmark"):
            assert d["details"]["zielone tokeny"].startswith(f"{sum(ann)} z {d['n']}"), algo_id
        if algo_id == "mpac":
            assert sum(ann) == int(d["details"]["tokeny w wybranych listach"].split()[0])


def test_edit_text(fake_lm):
    text = experiments.run_demo(fake_lm, "unigram", defaults("unigram"), "Hi", 50, 0)["watermarked"]["text"]
    assert experiments.edit_text(fake_lm, text, 0.0) == text
    assert experiments.edit_text(fake_lm, text, 0.5, seed=1) != text
    with pytest.raises(AppError):
        experiments.edit_text(fake_lm, text, 2)


def test_comparison_notes(fake_lm):
    s = {"prompts": experiments.PROMPTS[:2], "max_new_tokens": 30, "perplexity": False}
    r = experiments.run_comparison(fake_lm, [("unigram", defaults("unigram")), ("synthid", defaults("synthid"))], s)
    assert [x["algorithm"] for x in r["runs"]] == ["unigram", "synthid"]
    assert any("różne ustawienia próbkowania" in n for n in r["notes"])
    with pytest.raises(AppError):
        experiments.run_comparison(fake_lm, [("unigram", defaults("unigram"))], s)


def test_cancel_between_texts(fake_lm):
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(Cancelled) as e:
        experiments.run_test(fake_lm, "unigram", defaults("unigram"), {"prompts": experiments.PROMPTS, "max_new_tokens": 10},
                             should_stop=stop)
    assert e.value.done == 1


def test_progress_goes_up_to_one(fake_lm):
    seen = []
    experiments.run_test(fake_lm, "unigram", defaults("unigram"),
                         {"prompts": experiments.PROMPTS[:3], "max_new_tokens": 10, "perplexity": False},
                         progress=lambda f, t: seen.append(f))
    assert seen[-1] == 1.0 and all(b >= a for a, b in zip(seen, seen[1:]))


def test_bad_input(tmp_path, fake_lm):
    with pytest.raises(AppError):
        experiments.read_prompts_file(tmp_path / "missing.txt")
    (tmp_path / "empty.txt").write_text(" \n\n", encoding="utf-8")
    with pytest.raises(AppError):
        experiments.read_prompts_file(tmp_path / "empty.txt")
    (tmp_path / "binary.txt").write_bytes(b"\x89PNG\x00\x00")
    with pytest.raises(AppError):
        experiments.read_text_file(tmp_path / "binary.txt")
    (tmp_path / "cp1250.txt").write_bytes("Zażółć gęślą jaźń\nDrugie".encode("cp1250"))
    assert experiments.read_prompts_file(tmp_path / "cp1250.txt") == ["Zażółć gęślą jaźń", "Drugie"]
    for bad in ({"prompts": []}, {"prompts": ["a"], "max_new_tokens": 0}, {"prompts": ["a"], "edit_fractions": [2]}):
        with pytest.raises(AppError):
            experiments.check_settings(bad)
    with pytest.raises(AppError):
        experiments.run_demo(fake_lm, "kgw", defaults("kgw"), "  ")
    with pytest.raises(AppError):
        experiments.detect_text(fake_lm, "kgw", defaults("kgw"), "")


def test_wilson_interval():
    low, high = metrics.wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-3)
    low, high = metrics.wilson_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(0.7225, abs=1e-3)
    assert metrics.wilson_interval(0, 0) == (None, None)


def test_auc_and_tpr_at_fpr():
    assert metrics.auc([3, 4], [1, 2]) == 1.0
    assert metrics.auc([1, 2], [1, 2]) == 0.5
    assert metrics.auc([], [1]) is None
    t = metrics.tpr_at_fixed_fpr([5, 9], list(range(10)), 0.1)
    assert t["reliable"] and t["tpr"] == 0.5  # threshold = 90th percentile of 0..9 = 8.1


def test_perplexity_against_manual_computation(fake_lm):
    prompt, new = "abc def", [5, 17, 2000, 42]
    context = fake_lm.tokenizer.encode(prompt)
    nll = []
    for tok in new:
        logits = fake_lm.model.logits_for(context)
        logp = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
        nll.append(-logp[tok])
        context.append(tok)
    assert fake_lm.perplexity(prompt, new) == pytest.approx(math.exp(np.mean(nll)), rel=1e-4)
    assert fake_lm.perplexity(prompt, []) is None


def test_history(data_dir):
    rec = history.new_record("test", "Próba", ["kgw"], {"x": float("nan")})
    rec["status"] = "ok"
    history.save(rec)
    assert history.load(rec["id"])["params"]["x"] is None  # NaN -> null, valid JSON
    (data_dir / "experiments" / "broken.json").write_text("{not json", encoding="utf-8")
    records = history.list_records()
    assert any(r.get("broken") for r in records) and any(r["id"] == rec["id"] for r in records)
    history.delete(rec["id"])
    assert all(r["id"] != rec["id"] for r in history.list_records())
