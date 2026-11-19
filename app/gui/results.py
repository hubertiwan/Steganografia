"""Views of experiment results. Used right after a run and again in the history tab."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QSplitter, QTabWidget,
                               QVBoxLayout, QWidget)

from .. import algorithms, charts, reports
from .widgets import ChartView, TokenView, fill_table, fit_table_height, legend_text, make_table


MPAC_NOTE = ("Dla MPAC o wyniku decyduje p-wartość. Wartość z ze wzoru (1) z pracy jest zawyżona także dla tekstu bez znaku (zob. docs/mpac.md), więc sama w sobie nie świadczy o wykryciu.")


def _details_text(detection):
    return "   ".join(f"{k}: {v}" for k, v in detection.get("details", {}).items())


class DemoResultView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = make_table()
        lay.addWidget(self.table)
        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.details)
        row = QHBoxLayout()
        self.legend = QLabel()
        row.addWidget(self.legend)
        row.addStretch(1)
        self.colored = QCheckBox("Koloruj tokeny")
        self.colored.setChecked(True)
        row.addWidget(self.colored)
        lay.addLayout(row)
        split = QSplitter(Qt.Horizontal)
        self.views = []
        for title in ("Tekst ze znakiem wodnym", "Tekst bez znaku wodnego"):
            box = QWidget()
            col = QVBoxLayout(box)
            col.setContentsMargins(0, 0, 0, 0)
            col.addWidget(QLabel(f"<b>{title}</b>"))
            view = TokenView()
            col.addWidget(view)
            self.views.append(view)
            split.addWidget(box)
        split.setMinimumHeight(260)
        lay.addWidget(split, 1)
        self.colored.toggled.connect(lambda on: [v.set_colored(on) for v in self.views])

    def set_result(self, result):
        header, rows = reports.demo_rows(result)
        fill_table(self.table, header, rows)
        fit_table_height(self.table)
        text = "Tekst ze znakiem – " + _details_text(result["watermarked"]["detection"])
        if algorithms.get(result["algorithm"]).multibit:
            text += "\n" + MPAC_NOTE
        self.details.setText(text)
        self.legend.setText(legend_text(result["algorithm"]))
        for view, role in zip(self.views, ("watermarked", "plain")):
            view.show_tokens(result[role]["pieces"], result[role]["annotation"])


class DetectResultView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.verdict = QLabel()
        lay.addWidget(self.verdict)
        self.table = make_table()
        lay.addWidget(self.table)
        self.note = QLabel(MPAC_NOTE)
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        self.legend = QLabel()
        lay.addWidget(self.legend)
        self.view = TokenView()
        lay.addWidget(self.view, 1)

    def set_result(self, result):
        algo = algorithms.get(result["algorithm"])
        self.note.setVisible(algo.multibit)
        d = result["detection"]
        color = "#1b7a3a" if d["detected"] else "#444444"
        self.verdict.setText(f"<span style='font-size:12pt; color:{color}'><b>"
                             f"{'Znak wodny wykryty' if d['detected'] else 'Znak wodny nie wykryty'}</b></span>")
        rows = [["Tokeny", str(d["n"])], [algo.score_name, reports.num(d["score"])],
                ["p-wartość", reports.pval(d["p_value"])],
                ["Warunek wykrycia", reports.threshold_text(algo, d, result["params"])]]
        rows += [[k, v] for k, v in d["details"].items()]
        fill_table(self.table, ["", ""], rows)
        self.table.horizontalHeader().setVisible(False)
        fit_table_height(self.table)
        self.legend.setText(legend_text(result["algorithm"]))
        self.view.show_tokens(result["pieces"], result["annotation"])


class TestResultView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        lay.addWidget(self.summary)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)

        page = QWidget()
        col = QVBoxLayout(page)
        self.table = make_table()
        self.edits = make_table()
        col.addWidget(self.table)
        col.addWidget(QLabel("<b>Losowe zmiany tokenów w tekstach ze znakiem</b>"))
        col.addWidget(self.edits)
        col.addStretch(1)
        self.tabs.addTab(page, "Wyniki")

        page = QWidget()
        grid = QGridLayout(page)
        self.charts = [ChartView(), ChartView()]
        grid.addWidget(self.charts[0], 0, 0)
        grid.addWidget(self.charts[1], 0, 1)
        self.tabs.addTab(page, "Wykresy")

        split = QSplitter(Qt.Vertical)
        self.samples = make_table()
        self.samples.itemSelectionChanged.connect(self._show_sample)
        split.addWidget(self.samples)
        pair = QWidget()
        row = QHBoxLayout(pair)
        row.setContentsMargins(0, 0, 0, 0)
        self.texts = [QPlainTextEdit(readOnly=True), QPlainTextEdit(readOnly=True)]
        for title, w in zip(("Ze znakiem", "Bez znaku"), self.texts):
            c = QVBoxLayout()
            c.addWidget(QLabel(f"<b>{title}</b>"))
            c.addWidget(w)
            row.addLayout(c)
        split.addWidget(pair)
        self.tabs.addTab(split, "Teksty")
        self.result = None

    def set_result(self, result):
        self.result = result
        algo = algorithms.get(result["algorithm"])
        self.summary.setText(" ".join(reports.summary_lines(result)))
        fill_table(self.table, ["Metryka", "Wartość"], reports.test_rows(result))
        fit_table_height(self.table)
        header, rows = reports.edit_rows(result)
        fill_table(self.edits, header, rows)
        fit_table_height(self.edits)
        for view, fig in zip(self.charts, charts.test_figures(result)):
            view.show_figure(fig)
        header = ["#", "Polecenie", f"{algo.score_name} ze znakiem", "wykryty", f"{algo.score_name} bez znaku",
                  "fałszywy alarm"]
        rows = []
        for i, s in enumerate(result["samples"]):
            w, p = s["watermarked"]["detection"], s["plain"]["detection"]
            rows.append([i + 1, s["prompt"], reports.num(w["score"]), "tak" if w["detected"] else "NIE",
                         reports.num(p["score"]), "TAK" if p["detected"] else "nie"])
        if algo.multibit:
            header.append("odczytana wiadomość")
            for row, s in zip(rows, result["samples"]):
                row.append(s["watermarked"]["detection"]["message"])
        fill_table(self.samples, header, rows)
        if rows:
            self.samples.selectRow(0)

    def _show_sample(self):
        selected = self.samples.selectionModel().selectedRows()
        if selected and self.result:
            s = self.result["samples"][selected[0].row()]
            self.texts[0].setPlainText(s["watermarked"]["text"])
            self.texts[1].setPlainText(s["plain"]["text"])


class ComparisonResultView(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        lay.addWidget(self.notes)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        page = QWidget()
        col = QVBoxLayout(page)
        self.table = make_table()
        col.addWidget(self.table)
        col.addStretch(1)
        self.tabs.addTab(page, "Tabela")
        page = QWidget()
        grid = QGridLayout(page)
        self.charts = [ChartView(), ChartView()]
        grid.addWidget(self.charts[0], 0, 0)
        grid.addWidget(self.charts[1], 0, 1)
        self.tabs.addTab(page, "Wykresy")
        self.details = QTabWidget()
        self.tabs.addTab(self.details, "Szczegóły")

    def set_result(self, result):
        s = result["settings"]
        self.notes.setText(f"Warunki: {len(s['prompts'])} poleceń, maks. {s['max_new_tokens']} tokenów, ziarno "
                           f"{s['seed']}, model {result['model']['name']}.\n" + "\n".join(result["notes"]))
        header, rows = reports.comparison_rows(result)
        fill_table(self.table, header, rows)
        fit_table_height(self.table)
        for view, fig in zip(self.charts, charts.comparison_figures(result)):
            view.show_figure(fig)
        self.details.clear()
        for run in result["runs"]:
            view = TestResultView()
            view.set_result(run)
            self.details.addTab(view, algorithms.get(run["algorithm"]).name)


def view_for(result):
    view = {"demo": DemoResultView, "detect": DetectResultView, "test": TestResultView,
            "comparison": ComparisonResultView}[result["kind"]]()
    view.set_result(result)
    return view
