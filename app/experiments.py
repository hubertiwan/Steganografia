"""Experiments: demo (one prompt), detection of any text, test series and comparison.

Long functions take progress(fraction, text) and should_stop() callbacks, so the GUI can run
them in a worker thread, show progress and stop them between two generated texts.
Everything returned is plain data (dicts, lists, numbers), ready to be saved as JSON.
"""
import time
from pathlib import Path

import numpy as np

from watermarks.unigram import random_edits

from . import algorithms, metrics
from .errors import AppError, Cancelled

PLAIN_SEED_OFFSET = 10_000  # as in the demos: plain texts use seed 10000 + i
EDIT_SEED = 123             # as in the demos

PROMPTS = [  # the same 20 prompts as in the demos
    "Why is the sky blue?", "How do airplanes stay in the air?", "Why do we need to sleep?",
    "How does the internet work?", "Why do cats purr?", "What causes earthquakes?",
    "How do vaccines work?", "Why is the ocean salty?", "How do magnets work?",
    "Why do leaves change color in autumn?", "How does a refrigerator keep food cold?",
    "Why do we have different time zones?", "How do bees make honey?", "What is inflation?",
    "Why do stars twinkle?", "How does a credit card work?", "Why do onions make you cry?",
    "How do computers store information?", "Why does ice float on water?", "How do rainbows form?",
]


# ---------------------------------------------------------------- files
def read_text_file(path):
    p = Path(str(path).strip())
    if not str(path).strip() or not p.is_file():
        raise AppError(f"Nie ma takiego pliku: {path}", "Wybierz plik tekstowy (.txt).")
    if p.stat().st_size > 5_000_000:
        raise AppError(f"Plik {p.name} jest za duży.", "Maksymalnie 5 MB.")
    raw = p.read_bytes()
    if b"\x00" in raw[:4096]:
        raise AppError(f"{p.name} nie jest plikiem tekstowym.", "Wybierz plik .txt zapisany w UTF-8.")
    for encoding in ("utf-8-sig", "cp1250"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise AppError(f"Nie udało się odczytać {p.name}.", "Zapisz plik w kodowaniu UTF-8.")


def read_prompts_file(path):
    prompts = [line.strip() for line in read_text_file(path).splitlines() if line.strip()]
    if not prompts:
        raise AppError("Plik z poleceniami jest pusty.", "Wpisz jedno polecenie w każdej linii.")
    return prompts


# ---------------------------------------------------------------- helpers
def _stop_if_asked(should_stop, done=None):
    if should_stop is not None and should_stop():
        raise Cancelled(done)


def _check_length(max_new_tokens):
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 2000:
        raise AppError(f"Niepoprawna liczba tokenów: {max_new_tokens}.", "Podaj liczbę od 1 do 2000.")


def _timed_generate(algo, lm, prompt, wm, params, max_new_tokens, seed):
    start = time.perf_counter()
    tokens, text = algo.generate(lm, prompt, wm, params, max_new_tokens, seed)
    return [int(t) for t in tokens], text, time.perf_counter() - start


def _timed_detect(algo, wm, tokens, params):
    start = time.perf_counter()
    d = algo.detect(wm, tokens, params)
    d["time_s"] = time.perf_counter() - start
    return d


# ---------------------------------------------------------------- demo
def run_demo(lm, algo_id, params, prompt, max_new_tokens=120, seed=1, progress=None, should_stop=None):
    """One prompt: text with and without the watermark, detection of both and of the
    watermarked text with a wrong key. Detection uses the generated tokens."""
    algo = algorithms.get(algo_id)
    params = algo.check(params)
    _check_length(max_new_tokens)
    if not prompt or not prompt.strip():
        raise AppError("Polecenie jest puste.", "Wpisz polecenie dla modelu.")
    progress = progress or (lambda *a: None)
    wm = algo.build(params, lm.vocab_size)
    other_params = algo.other_key(params)
    wm_other = algo.build(other_params, lm.vocab_size)

    result = {"kind": "demo", "algorithm": algo_id, "params": params, "model": lm.describe(),
              "prompt": prompt, "max_new_tokens": max_new_tokens, "seed": seed}
    for role, w, s, frac in (("watermarked", wm, seed, 0.0), ("plain", None, seed + PLAIN_SEED_OFFSET, 0.45)):
        _stop_if_asked(should_stop)
        progress(frac, "Generowanie tekstu ze znakiem…" if w else "Generowanie tekstu bez znaku…")
        tokens, text, gen_time = _timed_generate(algo, lm, prompt, w, params, max_new_tokens, s)
        result[role] = {
            "text": text, "tokens": tokens, "pieces": lm.pieces(tokens), "gen_time_s": gen_time,
            "detection": _timed_detect(algo, wm, tokens, params),
            "other_key": algo.detect(wm_other, tokens, other_params),
            "annotation": algo.annotate(wm, tokens),
        }
    progress(0.9, "Liczenie perplexity…")
    for role in ("watermarked", "plain"):
        result[role]["perplexity"] = lm.perplexity(prompt, result[role]["tokens"])
    progress(1.0, "Gotowe")
    return result


def detect_text(lm, algo_id, params, text):
    """Detection in any text (pasted by the user or changed by hand)."""
    algo = algorithms.get(algo_id)
    params = algo.check(params)
    if not text or not text.strip():
        raise AppError("Tekst jest pusty.", "Wklej tekst albo wczytaj go z pliku.")
    wm = algo.build(params, lm.vocab_size)
    tokens = lm.tokenize(text)
    return {"kind": "detect", "algorithm": algo_id, "params": params, "model": lm.describe(), "text": text,
            "tokens": tokens, "pieces": lm.pieces(tokens), "detection": _timed_detect(algo, wm, tokens, params),
            "annotation": algo.annotate(wm, tokens)}


def edit_text(lm, text, fraction, seed=0):
    """Random insertions / deletions / replacements of a fraction of tokens (random_edits from
    watermarks/unigram, the same as in the demos)."""
    if not 0 <= fraction <= 1:
        raise AppError("Odsetek zmian musi być z przedziału 0–100%.")
    tokens = lm.tokenize(text)
    if not tokens:
        raise AppError("Brak tekstu do zmiany.")
    return lm.decode(random_edits(tokens, fraction, lm.vocab_size, np.random.default_rng(seed)))


# ---------------------------------------------------------------- test series
def check_settings(settings):
    s = dict(settings)
    s["prompts"] = [p for p in s.get("prompts", []) if str(p).strip()]
    if not s["prompts"]:
        raise AppError("Brak poleceń.", "Wybierz wbudowane polecenia albo plik .txt.")
    s["max_new_tokens"] = int(s.get("max_new_tokens", 100))
    _check_length(s["max_new_tokens"])
    s["seed"] = int(s.get("seed", 0))
    s["detect_on"] = s.get("detect_on", "tokens")
    if s["detect_on"] not in ("tokens", "text"):
        raise AppError("Nieznany tryb detekcji.")
    fractions = sorted({float(f) for f in s.get("edit_fractions", [0.0, 0.1, 0.2, 0.3, 0.5])})
    if any(not 0 <= f <= 1 for f in fractions):
        raise AppError("Odsetki zmian muszą być z przedziału 0–1.", "Np. 0, 0.1, 0.2, 0.3")
    s["edit_fractions"] = fractions
    s["perplexity"] = bool(s.get("perplexity", True))
    s["other_key"] = bool(s.get("other_key", True))
    return s


def run_test(lm, algo_id, params, settings, progress=None, should_stop=None, part=(0.0, 1.0)):
    """For every prompt: one text with and one without the watermark (same sampling), then
    detection, wrong key check, perplexity, times and robustness to random edits."""
    algo = algorithms.get(algo_id)
    params = algo.check(params)
    s = check_settings(settings)
    progress = progress or (lambda *a: None)
    wm = algo.build(params, lm.vocab_size)
    other_params = algo.other_key(params)
    wm_other = algo.build(other_params, lm.vocab_size) if s["other_key"] else None
    start = time.perf_counter()
    steps = 2 * len(s["prompts"]) + 1
    lo, hi = part

    samples = []
    for i, prompt in enumerate(s["prompts"]):
        sample = {"prompt": prompt}
        for role, w, seed in (("watermarked", wm, s["seed"] + i), ("plain", None, s["seed"] + PLAIN_SEED_OFFSET + i)):
            _stop_if_asked(should_stop, i)
            step = 2 * i + (0 if w else 1)
            progress(lo + (hi - lo) * step / steps,
                     f"{algo.name}: tekst {i + 1} z {len(s['prompts'])} ({'ze znakiem' if w else 'bez znaku'})")
            tokens, text, gen_time = _timed_generate(algo, lm, prompt, w, params, s["max_new_tokens"], seed)
            scored = tokens if s["detect_on"] == "tokens" else lm.tokenize(text)
            rec = {"seed": seed, "text": text, "tokens": tokens, "n_tokens": len(tokens), "gen_time_s": gen_time,
                   "detection": _timed_detect(algo, wm, scored, params)}
            if s["perplexity"]:
                rec["perplexity"] = lm.perplexity(prompt, tokens)
            if w is not None and wm_other is not None:
                rec["other_key"] = algo.detect(wm_other, scored, other_params)
            sample[role] = rec
        samples.append(sample)

    progress(lo + (hi - lo) * (steps - 1) / steps, f"{algo.name}: test odporności na zmiany…")
    result = {"kind": "test", "algorithm": algo_id, "params": params, "model": lm.describe(), "settings": s,
              "samples": samples, "summary": summarize(algo, samples, s),
              "edits": edit_robustness(algo, wm, params, samples, s, lm.vocab_size)}
    result["duration_s"] = time.perf_counter() - start
    progress(hi, "Gotowe")
    return result


def summarize(algo, samples, s):
    wm_recs = [x["watermarked"] for x in samples]
    plain_recs = [x["plain"] for x in samples]
    score_wm = [r["detection"]["score"] for r in wm_recs]
    score_plain = [r["detection"]["score"] for r in plain_recs]
    time_wm = sum(r["gen_time_s"] for r in wm_recs)
    time_plain = sum(r["gen_time_s"] for r in plain_recs)
    tokens_wm = sum(r["n_tokens"] for r in wm_recs)
    tokens_plain = sum(r["n_tokens"] for r in plain_recs)
    out = {
        "n": len(samples),
        "threshold": wm_recs[0]["detection"]["threshold"],
        "tpr": metrics.rate(r["detection"]["detected"] for r in wm_recs),
        "fpr": metrics.rate(r["detection"]["detected"] for r in plain_recs),
        "auc": metrics.auc(score_wm, score_plain),
        "tpr_at_fpr": [metrics.tpr_at_fixed_fpr(score_wm, score_plain, f) for f in (0.01, 0.1)],
        "score_wm_mean": metrics.mean(score_wm), "score_wm_std": metrics.std(score_wm),
        "score_plain_mean": metrics.mean(score_plain), "score_plain_std": metrics.std(score_plain),
        "p_value_wm_median": metrics.median([r["detection"]["p_value"] for r in wm_recs]),
        "tokens_mean": metrics.mean([r["n_tokens"] for r in wm_recs]),
        "ms_per_token_wm": 1000 * time_wm / tokens_wm if tokens_wm else None,
        "ms_per_token_plain": 1000 * time_plain / tokens_plain if tokens_plain else None,
        "detect_ms": metrics.mean([1000 * r["detection"]["time_s"] for r in wm_recs + plain_recs]),
    }
    if out["ms_per_token_wm"] and out["ms_per_token_plain"]:
        out["slowdown"] = out["ms_per_token_wm"] / out["ms_per_token_plain"]
    if s["perplexity"]:
        out["ppl_wm"] = metrics.median([r.get("perplexity") for r in wm_recs])
        out["ppl_plain"] = metrics.median([r.get("perplexity") for r in plain_recs])
    if s["other_key"]:
        out["other_key"] = metrics.rate(r["other_key"]["detected"] for r in wm_recs)
        out["score_other_key_mean"] = metrics.mean([r["other_key"]["score"] for r in wm_recs])
    if algo.multibit:
        acc = [r["detection"]["bit_accuracy"] for r in wm_recs]
        out["bit_accuracy"] = metrics.mean(acc)
        out["message_correct"] = metrics.rate(a == 1.0 for a in acc)
    return out


def edit_robustness(algo, wm, params, samples, s, vocab_size):
    """Random edits of the watermarked token sequences (as in the demos)."""
    rng = np.random.default_rng(EDIT_SEED)
    rows = []
    for frac in s["edit_fractions"]:
        dets = [algo.detect(wm, random_edits(x["watermarked"]["tokens"], frac, vocab_size, rng), params)
                for x in samples]
        row = {"fraction": frac, "score_mean": metrics.mean([d["score"] for d in dets]),
               "detected": metrics.rate(d["detected"] for d in dets)}
        if algo.multibit:
            row["bit_accuracy"] = metrics.mean([d["bit_accuracy"] for d in dets])
        rows.append(row)
    return rows


# ---------------------------------------------------------------- comparison
def run_comparison(lm, configs, settings, progress=None, should_stop=None):
    """configs = [(algo_id, params), ...]. All algorithms get the same model, prompts, seeds,
    text length and detection mode. Each one is compared with its own plain baseline."""
    if len(configs) < 2:
        raise AppError("Do porównania potrzebne są co najmniej dwa algorytmy.")
    s = check_settings(settings)
    runs = []
    for j, (algo_id, params) in enumerate(configs):
        part = (j / len(configs), (j + 1) / len(configs))
        runs.append(run_test(lm, algo_id, params, s, progress, should_stop, part))
    return {"kind": "comparison", "model": lm.describe(), "settings": s, "runs": runs,
            "notes": comparison_notes(runs)}


def comparison_notes(runs):
    notes = ["Każdy algorytm jest porównywany z własnymi tekstami bez znaku, wygenerowanymi tak samo "
             "(te same polecenia, ziarna i ustawienia próbkowania co tekst ze znakiem)."]
    sampling = set()
    for r in runs:
        algo = algorithms.get(r["algorithm"])
        sampling.add(tuple((p.name, r["params"][p.name]) for p in algo.params if p.group == "sampling"))
    if len(sampling) > 1:
        notes.append("Algorytmy mają różne ustawienia próbkowania (domyślne z ich prac), więc perplexity "
                     "i czas porównuj jako stosunek do własnych tekstów bez znaku, a nie wprost.")
    notes.append("Progi detekcji są różne (każdy z własnej pracy), więc skuteczność najlepiej porównywać "
                 "przez AUC i TPR przy tym samym FPR.")
    return notes
