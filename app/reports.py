"""Result tables and export to JSON, CSV and PDF. The same tables are shown in the GUI."""
import csv
import io
import json
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from . import APP_NAME, VERSION, algorithms, charts
from .errors import AppError
from .history import KINDS, STATUS, clean


# ---------------------------------------------------------------- formatting
def num(v, digits=2):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "tak" if v else "nie"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def pval(p):
    if p is None:
        return "–"
    if p < 1e-12:
        return "< 1e-12"
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def proportion(r):
    """'0.95 (19/20, 95% CI 0.76–0.99)'"""
    if not r or r["rate"] is None:
        return "–"
    return f"{r['rate']:.2f} ({r['k']}/{r['n']}, 95% CI {r['low']:.2f}–{r['high']:.2f})"


def threshold_text(algo, detection, params):
    if detection["threshold"] is not None:
        return f"{algo.score_name} > {detection['threshold']:g}"
    return f"p ≤ {params['alpha']:g}"


# ---------------------------------------------------------------- tables
def demo_rows(result):
    """Header and rows of the demo table: text with watermark, without, and with a wrong key."""
    algo = algorithms.get(result["algorithm"])
    header = ["Tekst", "Ocenione tokeny", algo.score_name, "p-wartość", "Wynik"]
    if algo.multibit:
        header += ["Odczytana wiadomość", "Zgodność bitów"]
    header.append("Perplexity")
    rows = []
    cases = [("ze znakiem", result["watermarked"], "detection"), ("bez znaku", result["plain"], "detection"),
             ("ze znakiem, inny klucz", result["watermarked"], "other_key")]
    for name, gen, key in cases:
        d = gen[key]
        row = [name, str(d["n"]), num(d["score"]), pval(d["p_value"]), "wykryty" if d["detected"] else "nie wykryty"]
        if algo.multibit:
            row += [d["message"], f"{d['bit_accuracy']:.0%}"]
        row.append(num(gen.get("perplexity")) if key == "detection" else "")
        rows.append(row)
    return header, rows


def test_rows(result):
    algo = algorithms.get(result["algorithm"])
    s = result["summary"]
    params = result["params"]
    rule = f"{algo.score_name} > {s['threshold']:g}" if s["threshold"] is not None else f"p ≤ {params['alpha']:g}"
    rows = [("Liczba poleceń", str(s["n"])),
            ("Warunek wykrycia", rule),
            ("TPR (teksty ze znakiem)", proportion(s["tpr"])),
            ("FPR (teksty bez znaku)", proportion(s["fpr"]))]
    if "other_key" in s:
        rows.append(("Wykrycia z innym kluczem", proportion(s["other_key"])))
    if algo.multibit:
        rows.append(("Dokładność bitowa (1 − BER)", num(s["bit_accuracy"], 3)))
        rows.append(("Cała wiadomość poprawna", proportion(s["message_correct"])))
    rows.append(("AUC", num(s["auc"], 3)))
    for t in s["tpr_at_fpr"]:
        if t:
            note = "" if t["reliable"] else " (za mało tekstów, orientacyjnie)"
            rows.append((f"TPR przy FPR {t['fpr']:.0%}", f"{t['tpr']:.2f}{note}"))
    rows += [(f"Średnie {algo.score_name}: ze znakiem / bez znaku",
              f"{num(s['score_wm_mean'])} / {num(s['score_plain_mean'])}"),
             ("Mediana p-wartości (ze znakiem)", pval(s["p_value_wm_median"])),
             ("Średnia liczba tokenów", num(s["tokens_mean"], 1))]
    if "ppl_wm" in s:
        rows.append(("Perplexity (mediana): ze znakiem / bez znaku", f"{num(s['ppl_wm'])} / {num(s['ppl_plain'])}"))
    rows.append(("Czas generacji na token [ms]: ze znakiem / bez znaku",
                 f"{num(s['ms_per_token_wm'], 1)} / {num(s['ms_per_token_plain'], 1)}"))
    rows.append(("Czas detekcji jednego tekstu [ms]", num(s["detect_ms"], 1)))
    return rows


def edit_rows(result):
    multibit = algorithms.get(result["algorithm"]).multibit
    rows = []
    for e in result["edits"]:
        row = [f"{100 * e['fraction']:.0f}%", num(e["score_mean"]), proportion(e["detected"])]
        if multibit:
            row.append(num(e["bit_accuracy"], 3))
        rows.append(row)
    header = ["Zmienione tokeny", "Średnia statystyka", "Odsetek wykryć"] + (["Dokładność bitowa"] if multibit else [])
    return header, rows


def comparison_rows(result):
    runs = result["runs"]
    header = ["Metryka"] + [algorithms.get(r["algorithm"]).name for r in runs]

    def row(label, fn):
        return [label] + [fn(r["summary"], r) for r in runs]

    rows = [row("AUC", lambda s, r: num(s["auc"], 3)),
            row("TPR przy FPR 10%", lambda s, r: num((s["tpr_at_fpr"][1] or {}).get("tpr"))),
            row("TPR przy własnym progu", lambda s, r: proportion(s["tpr"])),
            row("FPR przy własnym progu", lambda s, r: proportion(s["fpr"])),
            row("Dokładność bitowa", lambda s, r: num(s.get("bit_accuracy"), 3)),
            row("Perplexity: ze znakiem / bez", lambda s, r: f"{num(s.get('ppl_wm'))} / {num(s.get('ppl_plain'))}"),
            row("Spowolnienie generacji", lambda s, r: f"{num(s.get('slowdown'))}×"),
            row("Czas detekcji [ms]", lambda s, r: num(s["detect_ms"], 1))]
    for k, e in enumerate(runs[0]["edits"]):
        if e["fraction"] > 0:
            rows.append(row(f"Wykrycia po {100 * e['fraction']:.0f}% zmian",
                            lambda s, r, k=k: num(r["edits"][k]["detected"]["rate"])))
    return header, rows


def summary_lines(result):
    """A few sentences with the main numbers (only facts from this run)."""
    if result["kind"] == "test":
        s = result["summary"]
        lines = [f"Wykryto znak w {s['tpr']['k']} z {s['n']} tekstów ze znakiem i w {s['fpr']['k']} z {s['n']} "
                 f"tekstów bez znaku. AUC = {num(s['auc'], 3)}."]
        if "bit_accuracy" in s:
            lines.append(f"Średnia dokładność odczytu wiadomości: {s['bit_accuracy']:.1%}.")
        if s["n"] < 30:
            lines.append(f"Próba ma tylko {s['n']} poleceń, więc przedziały ufności są szerokie.")
        return lines
    if result["kind"] == "comparison":
        return [f"{algorithms.get(r['algorithm']).name}: {summary_lines(r)[0]}" for r in result["runs"]] + result["notes"]
    if result["kind"] == "demo":
        return ["Jedno polecenie – pojedynczy przykład, nie pomiar skuteczności."]
    return []


# ---------------------------------------------------------------- JSON / CSV
def export_json(record, path):
    data = {"app": f"{APP_NAME} {VERSION}", "exported": datetime.now().isoformat(timespec="seconds"),
            "experiment": record}
    Path(path).write_text(json.dumps(clean(data), ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)


def _text_rows(run):
    algo = algorithms.get(run["algorithm"])
    rows = []
    for i, sample in enumerate(run["samples"]):
        for role in ("watermarked", "plain"):
            rec = sample[role]
            d = rec["detection"]
            row = {"algorithm": algo.id, "prompt_no": i + 1, "watermark": role == "watermarked", "seed": rec["seed"],
                   "prompt": sample["prompt"], "tokens": rec["n_tokens"], "score": d["score"],
                   "p_value": d["p_value"], "detected": d["detected"], "perplexity": rec.get("perplexity"),
                   "gen_time_s": rec["gen_time_s"]}
            if algo.multibit:
                row["decoded_message"] = d["message"]
                row["bit_accuracy"] = d["bit_accuracy"] if role == "watermarked" else None
            row["text"] = rec["text"]
            rows.append(row)
    return rows


def export_csv(record, path):
    """One row per generated text (demo: one row per table line)."""
    result = record.get("result")
    if not result:
        raise AppError("Ten eksperyment nie ma wyników do zapisania.")
    if result["kind"] == "test":
        rows = _text_rows(result)
    elif result["kind"] == "comparison":
        rows = [r for run in result["runs"] for r in _text_rows(run)]
    elif result["kind"] == "demo":
        header, table = demo_rows(result)
        rows = [dict(zip(header, r)) for r in table]
        rows[0]["text"] = result["watermarked"]["text"]
        rows[1]["text"] = result["plain"]["text"]
    else:
        d = result["detection"]
        rows = [{"algorithm": result["algorithm"], "tokens": d["n"], "score": d["score"], "p_value": d["p_value"],
                 "detected": d["detected"], "text": result["text"]}]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel reads Polish letters correctly
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: "" if v is None else v for k, v in clean(r).items()})
    return Path(path)


# ---------------------------------------------------------------- PDF
def _fonts():
    """DejaVu Sans (shipped with matplotlib) has Polish letters."""
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu", str(folder / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(folder / "DejaVuSans-Bold.ttf")))


def export_pdf(record, path):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise AppError("Brak biblioteki reportlab.", "Zainstaluj zależności: pip install -r requirements.txt") from exc
    result = record.get("result")
    if not result:
        raise AppError("Ten eksperyment nie ma wyników do zapisania.")
    _fonts()
    styles = {"h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=15, leading=19, spaceAfter=6),
              "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11, leading=14, spaceBefore=10, spaceAfter=4),
              "p": ParagraphStyle("p", fontName="DejaVu", fontSize=9, leading=12),
              "cell": ParagraphStyle("c", fontName="DejaVu", fontSize=8, leading=10),
              "small": ParagraphStyle("s", fontName="DejaVu", fontSize=8, leading=10, textColor=colors.grey)}

    def para(text, style="p"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def table(rows, widths, header=True):
        t = Table([[para(c, "cell") for c in r] for r in rows], colWidths=widths, repeatRows=1 if header else 0)
        style = [("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "TOP")]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")))
        t.setStyle(TableStyle(style))
        return t

    def image(fig, width=16 * cm):
        w, h = fig.get_size_inches()
        return Image(io.BytesIO(charts.to_png(fig)), width=width, height=width * h / w)

    def params_table(run):
        algo = algorithms.get(run["algorithm"])
        rows = [["Parametr", "Wartość"]] + [[p.label, str(run["params"][p.name])] for p in algo.params]
        return [para(f"Parametry: {algo.name}", "h2"), para(algo.paper, "small"), table(rows, [6 * cm, 10 * cm])]

    def settings_table(s):
        rows = [["Ustawienie", "Wartość"], ["Liczba poleceń", str(len(s["prompts"]))],
                ["Maks. tokenów na tekst", str(s["max_new_tokens"])],
                ["Ziarna", f"ze znakiem: {s['seed']} + i, bez znaku: {s['seed'] + 10000} + i"],
                ["Detekcja na", "tokenach z generacji" if s["detect_on"] == "tokens" else "tekście po tokenizacji"],
                ["Zmiany tokenów", ", ".join(f"{100 * f:.0f}%" for f in s["edit_fractions"])]]
        return [para("Ustawienia", "h2"), table(rows, [6 * cm, 10 * cm])]

    model = result.get("model", {})
    story = [para(f"{KINDS.get(record['kind'], record['kind'])}: {record['title']}", "h1"),
             para(f"Data: {record['created'].replace('T', ' ')} · status: {STATUS.get(record['status'])} · "
                  f"model: {model.get('name', '–')} ({model.get('device', '–')})", "small")]
    if record.get("files"):
        story.append(para("Pliki: " + ", ".join(record["files"]), "small"))
    story.append(Spacer(1, 6))

    kind = result["kind"]
    if kind == "test":
        story += params_table(result) + settings_table(result["settings"])
        story += [para("Wyniki", "h2"), table([["Metryka", "Wartość"]] + [list(r) for r in test_rows(result)],
                                              [8 * cm, 8 * cm])]
        header, rows = edit_rows(result)
        story += [para("Losowe zmiany tokenów", "h2"), table([header] + rows, [16 * cm / len(header)] * len(header))]
        story += [para("Wykresy", "h2")] + [image(f) for f in charts.test_figures(result)]
    elif kind == "comparison":
        story += settings_table(result["settings"])
        for run in result["runs"]:
            story += params_table(run)
        header, rows = comparison_rows(result)
        first = 5 * cm
        rest = (16 * cm - first) / (len(header) - 1)
        story += [para("Porównanie", "h2"), table([header] + rows, [first] + [rest] * (len(header) - 1))]
        story += [para("Wykresy", "h2")] + [image(f) for f in charts.comparison_figures(result)]
    elif kind == "demo":
        story += params_table(result)
        story += [para("Polecenie", "h2"), para(result["prompt"])]
        header, rows = demo_rows(result)
        story += [para("Wyniki", "h2"), table([header] + rows, [16 * cm / len(header)] * len(header))]
        story += [para("Tekst ze znakiem", "h2"), para(result["watermarked"]["text"]),
                  para("Tekst bez znaku", "h2"), para(result["plain"]["text"])]
    else:
        story += params_table(result)
        d = result["detection"]
        algo = algorithms.get(result["algorithm"])
        rows = [["Tokeny", str(d["n"])], [algo.score_name, num(d["score"])], ["p-wartość", pval(d["p_value"])],
                ["Wynik", "wykryty" if d["detected"] else "nie wykryty"]] + [[k, v] for k, v in d["details"].items()]
        story += [para("Wynik", "h2"), table(rows, [6 * cm, 10 * cm], header=False),
                  para("Tekst", "h2"), para(result["text"][:5000])]

    lines = summary_lines(result)
    if lines:
        story += [para("Podsumowanie", "h2")] + [para("– " + line) for line in lines]
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title=record["title"]).build(story)
    return Path(path)
