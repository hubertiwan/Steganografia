"""The algorithms available in the application.

Every algorithm here is a thin wrapper around a package from watermarks/. The wrapper only
describes the algorithm, checks the parameters and gives all algorithms the same interface:

    build(params, vocab_size) -> watermark object
    generate(lm, prompt, wm, params, max_new_tokens, seed) -> (tokens, text); wm=None -> no watermark
    detect(wm, tokens, params) -> dict with score, threshold, p_value, n, detected, details
    annotate(wm, tokens) -> one value per token for coloring (1 good, 0 bad, None = not scored)
"""
from dataclasses import dataclass, field
from functools import lru_cache
from math import log, sqrt

from watermarks import dipmark, kgw, mpac, synthid, unigram

from .errors import AppError


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str          # "int", "float", "text" or "bits"
    default: object
    help: str
    low: float = None
    high: float = None
    step: float = None
    decimals: int = 3
    group: str = "watermark"   # "watermark", "detection" or "sampling"
    low_open: bool = False     # True -> value must be > low

    def check(self, value):
        try:
            if self.kind == "int":
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError
                value = int(value)
            elif self.kind == "float":
                value = float(value)
            else:
                value = str(value).strip()
        except (TypeError, ValueError):
            raise AppError(f"Parametr „{self.label}” ma niepoprawną wartość: {value!r}.", self.help)
        if self.kind in ("int", "float"):
            too_low = self.low is not None and (value <= self.low if self.low_open else value < self.low)
            too_high = self.high is not None and value > self.high
            if value != value or too_low or too_high:
                raise AppError(f"Parametr „{self.label}” = {value} jest poza zakresem.", self.help)
        elif self.kind == "bits":
            if not 1 <= len(value) <= 64 or set(value) - {"0", "1"}:
                raise AppError(f"Wiadomość „{value}” jest niepoprawna.", self.help)
        elif not value:
            raise AppError(f"Parametr „{self.label}” nie może być pusty.", self.help)
        return value


@dataclass
class Algorithm:
    id: str
    name: str
    paper: str
    summary: str
    steps: list
    params: list
    color: str
    score_name: str = "z"
    multibit: bool = False
    doc: str = ""
    limitations: list = field(default_factory=list)

    def defaults(self):
        return {p.name: p.default for p in self.params}

    def check(self, params):
        unknown = set(params) - {p.name for p in self.params}
        if unknown:
            raise AppError(f"Nieznane parametry: {', '.join(sorted(unknown))}.")
        return {p.name: p.check(params.get(p.name, p.default)) for p in self.params}

    def param(self, name):
        return next(p for p in self.params if p.name == name)

    def other_key(self, params):
        """The same parameters with a different key, for the wrong key check."""
        p = dict(params)
        if isinstance(p["key"], int):
            p["key"] = p["key"] + 1
        else:
            p["key"] = p["key"] + "-inny"
        return p

    # implemented by the subclasses
    def build(self, params, vocab_size):
        raise NotImplementedError

    def generate(self, lm, prompt, wm, params, max_new_tokens, seed):
        raise NotImplementedError

    def detect(self, wm, tokens, params):
        raise NotImplementedError

    def annotate(self, wm, tokens):
        raise NotImplementedError


def _key_param(default=15485863):
    return Param("key", "Klucz", "int", default, "Tajny klucz, liczba całkowita.", low=0, high=2**62)


def _sampling(temperature, top_p=None, top_k=None):
    out = [Param("temperature", "Temperatura", "float", temperature, "Temperatura próbkowania, większa od 0.",
                 low=0, high=5, step=0.1, group="sampling", low_open=True)]
    if top_p is not None:
        out.append(Param("top_p", "top-p", "float", top_p, "Próbkowanie nucleus, wartość z (0, 1]. 1 = wyłączone.",
                         low=0, high=1, step=0.05, group="sampling", low_open=True))
    if top_k is not None:
        out.append(Param("top_k", "top-k", "int", top_k, "Liczba najbardziej prawdopodobnych tokenów do losowania.",
                         low=1, high=100000, step=10, group="sampling"))
    return out


def _cached_green_mask(wm):
    """KGW and DiPmark build a permutation of the whole vocabulary for every token. With a
    150k vocabulary that is a few ms each, and the same contexts come back many times
    (detection, coloring, edits), so the result is cached on the object."""
    wm.green_mask = lru_cache(maxsize=256)(wm.green_mask)
    return wm


# ---------------------------------------------------------------- Unigram-Watermark
class Unigram(Algorithm):
    def build(self, params, vocab_size):
        wm = unigram.UnigramWatermark(params["key"], vocab_size, params["gamma"], params["delta"])
        if wm.green_size in (0, vocab_size):
            raise AppError("Przy tym γ lista zielona jest pusta albo obejmuje cały słownik.", "Zmień γ.")
        return wm

    def generate(self, lm, prompt, wm, params, max_new_tokens, seed):
        return unigram.generate(lm.model, lm.tokenizer, prompt, wm=wm, max_new_tokens=max_new_tokens,
                                temperature=params["temperature"], top_p=params["top_p"], seed=seed,
                                use_chat_template=lm.use_chat_template)

    def detect(self, wm, tokens, params):
        d = wm.detect(tokens, tau=params["tau"])
        u = wm.detect_unique(tokens, alpha=0.01)
        return {"score": d["z"], "threshold": params["tau"], "p_value": d["p_value"], "n": d["n"],
                "detected": bool(d["watermarked"]),
                "details": {"zielone tokeny": f"{d['green']} z {d['n']} ({d['green_frac']:.0%}, "
                                              f"bez znaku ok. {wm.gamma_eff:.0%})",
                            "Alg. 4 (unikalne tokeny)": f"z = {u['z']:.2f}, próg {u['tau']:.2f}, "
                                                        f"{'wykryty' if u['watermarked'] else 'nie wykryty'}"}}

    def annotate(self, wm, tokens):
        return [1 if wm.green_mask[int(t)] else 0 for t in tokens]


# ---------------------------------------------------------------- SynthID
class SynthID(Algorithm):
    def build(self, params, vocab_size):
        return synthid.TournamentWatermark(params["key"], params["context_size"], params["num_layers"])

    def generate(self, lm, prompt, wm, params, max_new_tokens, seed):
        return synthid.generate(lm.model, lm.tokenizer, prompt, wm=wm, max_new_tokens=max_new_tokens,
                                temperature=params["temperature"], top_k=params["top_k"], seed=seed,
                                use_chat_template=lm.use_chat_template)

    def detect(self, wm, tokens, params):
        d = wm.detect(tokens, threshold=params["threshold"])
        details = {}
        if d["n"]:
            # from z = (sum g - nm/2) / sqrt(nm/4)
            details["średnia g-wartość"] = f"{0.5 + d['z'] / (2 * sqrt(d['n'] * wm.num_layers)):.3f} (bez znaku 0.5)"
        return {"score": d["z"], "threshold": params["threshold"], "p_value": d["p_value"], "n": d["n"],
                "detected": bool(d["watermarked"]), "details": details}

    def annotate(self, wm, tokens):
        """Mean g-value of every token, with the same rules as z_score()."""
        tokens = [int(t) for t in tokens]
        out = [None] * len(tokens)
        seen = set()
        for t in range(wm.context_size, len(tokens)):
            context = tuple(tokens[t - wm.context_size:t])
            if context in seen:
                continue
            seen.add(context)
            seed = wm.seed(context)
            g = sum(wm.g_value(tokens[t], layer, seed) for layer in range(1, wm.num_layers + 1))
            out[t] = g / wm.num_layers
        return out


# ---------------------------------------------------------------- KGW
class KGW(Algorithm):
    def build(self, params, vocab_size):
        return _cached_green_mask(kgw.KGWWatermark(params["key"], vocab_size, params["gamma"], params["delta"]))

    def generate(self, lm, prompt, wm, params, max_new_tokens, seed):
        return kgw.generate(lm.model, lm.tokenizer, prompt, wm=wm, max_new_tokens=max_new_tokens,
                            temperature=params["temperature"], top_p=params["top_p"], seed=seed,
                            use_chat_template=lm.use_chat_template)

    def detect(self, wm, tokens, params):
        d = wm.detect(tokens, z_threshold=params["z_threshold"])
        return {"score": d["z"], "threshold": params["z_threshold"], "p_value": d["p_value"], "n": d["n"],
                "detected": bool(d["watermarked"]),
                "details": {"zielone tokeny": f"{d['green']} z {d['n']} ({d['green_frac']:.0%}, "
                                              f"bez znaku ok. {wm.gamma_eff:.0%})"}}

    def annotate(self, wm, tokens):
        tokens = [int(t) for t in tokens]
        return [None] + [1 if wm.green_mask(tokens[t - 1])[tokens[t]] else 0 for t in range(1, len(tokens))]


# ---------------------------------------------------------------- DiPmark
class DiPmark(Algorithm):
    def build(self, params, vocab_size):
        return _cached_green_mask(dipmark.DiPmarkWatermark(params["key"], vocab_size, params["alpha"],
                                                           params["gamma"], params["context_size"]))

    def generate(self, lm, prompt, wm, params, max_new_tokens, seed):
        return dipmark.generate(lm.model, lm.tokenizer, prompt, wm=wm, max_new_tokens=max_new_tokens,
                                temperature=params["temperature"], top_p=params["top_p"], seed=seed,
                                use_chat_template=lm.use_chat_template)

    def detect(self, wm, tokens, params):
        d = wm.detect(tokens, fpr=params["fpr"])
        # the demo reports Phi*sqrt(n): then one threshold works for any text length
        score = d["phi"] * sqrt(d["n"]) if d["n"] else 0.0
        return {"score": score, "threshold": sqrt(log(1 / params["fpr"]) / 2), "p_value": d["p_value"],
                "n": d["n"], "detected": bool(d["watermarked"]),
                "details": {"zielone tokeny": f"{d['green']} z {d['n']} ({d['green_frac']:.0%}, "
                                              f"bez znaku ok. {1 - wm.gamma:.0%})",
                            "Φ": f"{d['phi']:.3f}"}}

    def annotate(self, wm, tokens):
        """Same rules as score(): first a tokens and repeated texture keys are not scored."""
        tokens = [int(t) for t in tokens]
        out = [None] * len(tokens)
        seen = set()
        for t in range(wm.context_size, len(tokens)):
            context = tuple(tokens[t - wm.context_size:t])
            if context in seen:
                continue
            seen.add(context)
            out[t] = 1 if wm.green_mask(context)[tokens[t]] else 0
        return out


# ---------------------------------------------------------------- MPAC
class MPAC(Algorithm):
    def build(self, params, vocab_size):
        return mpac.MPACWatermark(params["key"], vocab_size, len(params["message"]), params["gamma"],
                                  params["delta"], params["context_size"])

    def generate(self, lm, prompt, wm, params, max_new_tokens, seed):
        return mpac.generate(lm.model, lm.tokenizer, prompt, wm=wm, message=params["message"],
                             max_new_tokens=max_new_tokens, temperature=params["temperature"],
                             top_p=params["top_p"], seed=seed, use_chat_template=lm.use_chat_template)

    def detect(self, wm, tokens, params):
        d = wm.detect(tokens, alpha=params["alpha"])
        details = {"odczytana wiadomość": d["message"],
                   "tokeny w wybranych listach": f"{d['w']} z {d['n']}"}
        if d["empty_positions"]:
            details["pozycje bez tokenów"] = f"{d['empty_positions']} z {wm.positions} (odczytane jako 0)"
        return {"score": d["z"], "threshold": None, "p_value": d["p_value"], "n": d["n"],
                "detected": bool(d["watermarked"]), "message": d["message"],
                "bit_accuracy": mpac.bit_accuracy(params["message"], d["message"]), "details": details}

    def annotate(self, wm, tokens):
        """1 = token is in the colorlist of the decoded digit of its position."""
        tokens = [int(t) for t in tokens]
        digits = wm.count(tokens)[0].argmax(axis=1)  # the same argmax as in decode()
        out = [None] * len(tokens)
        for t in range(wm.context_size, len(tokens)):
            position, color = wm.locate(tokens[t - wm.context_size:t], tokens[t])
            out[t] = 1 if color == digits[position] else 0
        return out


# ---------------------------------------------------------------- list
ALGORITHMS = {a.id: a for a in [
    Unigram(
        id="unigram", name="Unigram-Watermark", color="#2a78d6", doc="unigram.md",
        paper="X. Zhao, P. Ananth, L. Li, Y.-X. Wang, Provable Robust Watermarking for AI-Generated Text, ICLR 2024",
        summary="Klucz dzieli słownik na listę zieloną i czerwoną. Podział jest stały dla całego tekstu. "
                "Model dostaje bonus do zielonych tokenów, a detektor liczy, ile ich jest w tekście.",
        steps=["Z klucza powstaje losowa permutacja słownika, pierwsze γ|V| tokenów to lista zielona.",
               "Przy generowaniu do logitów zielonych tokenów dodawane jest δ.",
               "Detekcja: z = (liczba zielonych − γn) / √(nγ(1−γ)). Znak jest wykryty, gdy z > τ."],
        limitations=["Stałą listę zieloną da się odtworzyć statystycznie z wielu tekstów."],
        params=[_key_param(),
                Param("gamma", "γ (część zielona)", "float", 0.5, "Ułamek słownika na liście zielonej, z (0, 1).",
                      low=0, high=0.95, step=0.05, low_open=True),
                Param("delta", "δ (siła znaku)", "float", 2.0, "Bonus do logitów zielonych tokenów.",
                      low=0, high=20, step=0.5),
                Param("tau", "Próg z (τ)", "float", 6.0, "Próg z-score, w pracy 6.", low=0, high=50, step=0.5,
                      group="detection")] + _sampling(1.0, top_p=0.9)),
    SynthID(
        id="synthid", name="SynthID-Text", color="#eb6834", doc="synthid.md",
        paper="S. Dathathri i in., Scalable watermarking for identifying large language model outputs, Nature 2024",
        summary="Znak wodny wprowadzany sposobem losowania tokenu (Tournament sampling). Model losuje kilku "
                "kandydatów, a zwycięzcę wybiera turniej rozstrzygany pseudolosowymi g-wartościami z klucza.",
        steps=["Ziarno z klucza i H poprzednich tokenów.",
               "Losowanie 2^m kandydatów z rozkładu modelu i m rund turnieju: wygrywa token z większą g-wartością.",
               "Detekcja: średnia g-wartość tokenów, bez znaku wynosi 0.5. Test z, próg 2.326 (FPR 1%)."],
        limitations=["Koszt rośnie z liczbą kandydatów 2^m."],
        params=[Param("key", "Klucz", "text", "secret-watermark-key", "Tajny klucz, dowolny tekst."),
                Param("context_size", "H (kontekst)", "int", 4, "Liczba poprzednich tokenów w ziarnie.",
                      low=1, high=16),
                Param("num_layers", "m (rundy turnieju)", "int", 8, "Turniej ma 2^m kandydatów, maksymalnie 10.",
                      low=1, high=10),
                Param("threshold", "Próg z", "float", 2.326, "2.326 odpowiada FPR = 1%.", low=0, high=50,
                      step=0.1, group="detection")] + _sampling(0.7, top_k=100)),
    KGW(
        id="kgw", name="KGW", color="#1baf7a", doc="kgw.md",
        paper="J. Kirchenbauer i in., A Watermark for Large Language Models, ICML 2023",
        summary="Lista zielona jest losowana dla każdego tokenu na nowo, z klucza i poprzedniego tokenu. "
                "Zielone tokeny dostają bonus δ, detektor liczy zielone tokeny i liczy z-score.",
        steps=["Ziarno z klucza i poprzedniego tokenu, z niego permutacja słownika i lista zielona γ|V|.",
               "Do logitów zielonych tokenów dodawane jest δ.",
               "Detekcja: dla każdego tokenu odtwarzana jest jego lista zielona, z-score jak w Unigram, próg 4."],
        params=[_key_param(),
                Param("gamma", "γ (część zielona)", "float", 0.25, "Ułamek słownika na liście zielonej, z (0, 1).",
                      low=0, high=0.95, step=0.05, low_open=True),
                Param("delta", "δ (siła znaku)", "float", 2.0, "Bonus do logitów zielonych tokenów.",
                      low=0, high=20, step=0.5),
                Param("z_threshold", "Próg z", "float", 4.0, "Próg z-score, w pracy 4.", low=0, high=50,
                      step=0.5, group="detection")] + _sampling(1.0, top_p=0.9)),
    DiPmark(
        id="dipmark", name="DiPmark", color="#eda100", doc="dipmark.md", score_name="Φ·√n",
        paper="Y. Wu i in., A Resilient and Accessible Distribution-Preserving Watermark for LLMs, ICML 2024",
        summary="Znak, który nie zmienia rozkładu modelu (uśredniając po kluczach). Część prawdopodobieństwa "
                "z początku permutacji słownika jest przenoszona na jej koniec.",
        steps=["Permutacja słownika z klucza i poprzedniego tokenu.",
               "DiP-reweight: masa prawdopodobieństwa z przedziału [0, α] trafia na koniec permutacji.",
               "Detekcja: lista zielona to koniec permutacji, Φ = L_G/n − (1 − γ). Wykryty, gdy Φ·√n ≥ 1.517 (FPR 1%)."],
        limitations=["Sygnał słabszy niż w KGW przy tej samej długości tekstu."],
        params=[_key_param(),
                Param("alpha", "α (siła przesunięcia)", "float", 0.45, "Wartość z [0, 0.5], w pracy 0.45.",
                      low=0, high=0.5, step=0.05),
                Param("gamma", "γ (podział do detekcji)", "float", 0.5, "Wartość z (0, 1), w pracy 0.5.",
                      low=0, high=0.95, step=0.05, low_open=True),
                Param("context_size", "a (kontekst)", "int", 1, "Liczba poprzednich tokenów w kluczu tekstury.",
                      low=1, high=8),
                Param("fpr", "FPR detektora", "float", 0.01, "Dopuszczalny odsetek fałszywych alarmów.",
                      low=0, high=0.5, step=0.01, decimals=4, group="detection", low_open=True)]
        + _sampling(1.0, top_p=0.9)),
    MPAC(
        id="mpac", name="MPAC (wielobitowy)", color="#e87ba4", doc="mpac.md", score_name="z", multibit=True,
        paper="K. Yoo, W. Ahn, N. Kwak, Advancing Beyond Identification: Multi-bit Watermark for LLMs "
              "via Position Allocation, NAACL 2024",
        summary="Znak wodny, który przenosi wiadomość (np. ID użytkownika). Każdy token jest przypisany do "
                "jednej pozycji wiadomości i wzmacnia listę kolorów odpowiadającą cyfrze na tej pozycji.",
        steps=["Wiadomość zapisana w systemie o podstawie r = ⌊1/γ⌋, słownik dzielony na r list kolorów.",
               "Ziarno z klucza i poprzedniego tokenu wybiera pozycję p wiadomości i permutację słownika.",
               "Do logitów listy o numerze równym cyfrze m[p] dodawane jest δ.",
               "Odczyt: liczniki W[p][kolor] i głosowanie większościowe na każdej pozycji.",
               "Detekcja: dokładna p-wartość dla sumy maksimów liczników (szczegóły w docs/mpac.md)."],
        limitations=["Im dłuższa wiadomość, tym mniej tokenów na pozycję i więcej błędów bitowych."],
        params=[_key_param(),
                Param("message", "Wiadomość (bity)", "bits", "10110010", "Od 1 do 64 znaków 0 lub 1."),
                Param("gamma", "γ (rozmiar listy)", "float", 0.25, "Wartość z (0, 0.5], liczba list r = ⌊1/γ⌋.",
                      low=0.05, high=0.5, step=0.05),
                Param("delta", "δ (siła znaku)", "float", 2.0, "Bonus do logitów wybranej listy.",
                      low=0, high=20, step=0.5),
                Param("context_size", "h (kontekst)", "int", 1, "Liczba poprzednich tokenów w ziarnie.",
                      low=1, high=8),
                Param("alpha", "α (FPR detektora)", "float", 0.001, "Wykryty, gdy p-wartość ≤ α.",
                      low=0, high=0.5, step=0.001, decimals=4, group="detection", low_open=True)]
        + _sampling(0.7, top_p=1.0)),
]}


def get(algo_id):
    if algo_id not in ALGORITHMS:
        raise AppError(f"Nieznany algorytm: {algo_id}.")
    return ALGORITHMS[algo_id]
