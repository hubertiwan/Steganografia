"""Charts for the GUI and the PDF report (matplotlib Figure objects, no pyplot).
Color = algorithm; texts without a watermark are grey and hatched."""
import io

import numpy as np
from matplotlib.figure import Figure

from . import algorithms

GREY = "#8a8984"


def _figure(width=6.0, height=3.4):
    return Figure(figsize=(width, height), dpi=100, layout="constrained")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def score_histogram(result):
    algo = algorithms.get(result["algorithm"])
    fig = _figure()
    ax = fig.add_subplot()
    wm = [s["watermarked"]["detection"]["score"] for s in result["samples"]]
    plain = [s["plain"]["detection"]["score"] for s in result["samples"]]
    threshold = result["summary"]["threshold"]
    values = wm + plain + ([threshold] if threshold is not None else [])
    bins = np.linspace(min(values) - 0.5, max(values) + 0.5, 20)
    ax.hist(plain, bins=bins, color=GREY, alpha=0.8, hatch="//", edgecolor="white", label="bez znaku")
    ax.hist(wm, bins=bins, color=algo.color, alpha=0.85, edgecolor="white", label="ze znakiem")
    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=1, label=f"próg = {threshold:g}")
    _style(ax, f"Rozkład statystyki {algo.score_name}", algo.score_name, "liczba tekstów")
    ax.legend(fontsize=8, frameon=False)
    return fig


def edits_chart(result):
    algo = algorithms.get(result["algorithm"])
    fig = _figure()
    ax = fig.add_subplot()
    x = [100 * e["fraction"] for e in result["edits"]]
    ax.plot(x, [e["detected"]["rate"] for e in result["edits"]], "o-", color=algo.color, label="odsetek wykryć")
    if algo.multibit:
        ax.plot(x, [e["bit_accuracy"] for e in result["edits"]], "s--", color="black", label="dokładność bitowa")
    ax.set_ylim(-0.03, 1.05)
    _style(ax, "Odporność na losowe zmiany tokenów", "zmienione tokeny [%]", "wartość")
    ax.legend(fontsize=8, frameon=False)
    return fig


def comparison_chart(result):
    fig = _figure(6.4, 3.4)
    ax = fig.add_subplot()
    names = ["AUC", "TPR (próg)", "FPR (próg)", "TPR przy FPR 10%"]
    x = np.arange(len(names))
    width = 0.8 / len(result["runs"])
    for j, run in enumerate(result["runs"]):
        algo = algorithms.get(run["algorithm"])
        s = run["summary"]
        values = [s["auc"], s["tpr"]["rate"], s["fpr"]["rate"], (s["tpr_at_fpr"][1] or {}).get("tpr")]
        values = [v if v is not None else 0 for v in values]
        ax.bar(x - 0.4 + width * (j + 0.5), values, width * 0.9, color=algo.color, label=algo.name)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    _style(ax, "Skuteczność detekcji", "", "wartość")
    ax.legend(fontsize=8, frameon=False, ncol=min(3, len(result["runs"])), loc="upper center")
    return fig


def comparison_edits_chart(result):
    fig = _figure(6.4, 3.4)
    ax = fig.add_subplot()
    for run in result["runs"]:
        algo = algorithms.get(run["algorithm"])
        ax.plot([100 * e["fraction"] for e in run["edits"]], [e["detected"]["rate"] for e in run["edits"]],
                "o-", color=algo.color, label=algo.name)
    ax.set_ylim(-0.03, 1.05)
    _style(ax, "Odsetek wykryć po losowych zmianach", "zmienione tokeny [%]", "odsetek wykryć")
    ax.legend(fontsize=8, frameon=False)
    return fig


def test_figures(result):
    return [score_histogram(result), edits_chart(result)]


def comparison_figures(result):
    return [comparison_chart(result), comparison_edits_chart(result)]


def to_png(fig, dpi=150):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    return buf.getvalue()
