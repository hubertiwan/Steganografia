"""Small widgets used by several tabs."""
import html
import random
import traceback
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QProgressBar, QPushButton, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import algorithms, experiments, history, reports
from ..errors import AppError

# token highlight colors (Demonstracja, Sprawdź tekst, Pomoc); change them here
GREEN_BG = "#a6e3b4"     # background of a "good" token (green list / g > 0.5 / list of the decoded digit)
RED_BG = "#f5b1a9"       # background of a "bad" token
TOKEN_TEXT = "#111111"   # text color on the colored tokens, always dark so it is readable on both backgrounds
NOT_SCORED = "#7a7a7a"   # text color of the tokens the detector skips


def show_error(parent, exc, tb=None):
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Warning)
    box.setWindowTitle("Błąd")
    if isinstance(exc, AppError):
        box.setText(exc.message)
        if exc.hint:
            box.setInformativeText(exc.hint)
    else:
        box.setText(f"Nieoczekiwany błąd: {type(exc).__name__}: {exc}")
        box.setInformativeText("Operacja została przerwana, program działa dalej.")
        box.setDetailedText(tb or "".join(traceback.format_exception(exc)))
    box.exec()


def fill_table(table, header, rows):
    table.clear()
    table.setColumnCount(len(header))
    table.setHorizontalHeaderLabels(header)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(str(value)))
    table.resizeColumnsToContents()
    table.horizontalHeader().setStretchLastSection(True)
    table.resizeRowsToContents()


def make_table(min_height=0):
    t = QTableWidget()
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.setWordWrap(True)
    if min_height:
        t.setMinimumHeight(min_height)
    return t


def fit_table_height(table):
    """Make a short table exactly as tall as its rows, so no scrollbar appears."""
    height = table.horizontalHeader().height() + 4
    for r in range(table.rowCount()):
        height += table.rowHeight(r)
    table.setFixedHeight(height)


def algorithm_combo():
    combo = QComboBox()
    for algo in algorithms.ALGORITHMS.values():
        combo.addItem(algo.name, algo.id)
    return combo


# ---------------------------------------------------------------- parameters
class ParamForm(QWidget):
    """Form built from the parameter list of an algorithm."""
    GROUPS = [("watermark", "Znak wodny"), ("detection", "Detekcja"), ("sampling", "Próbkowanie")]

    def __init__(self, algo):
        super().__init__()
        self.algo = algo
        self.inputs = {}
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        for group, title in self.GROUPS:
            params = [p for p in algo.params if p.group == group]
            if not params:
                continue
            box = QGroupBox(title)
            form = QFormLayout(box)
            for p in params:
                widget = self._input(p)
                widget.setToolTip(p.help)
                form.addRow(p.label, widget)
            row.addWidget(box)
        reset = QPushButton("Domyślne")
        reset.setToolTip("Wartości z publikacji i skryptów demo")
        reset.clicked.connect(lambda: self.set_values(algo.defaults()))
        col = QVBoxLayout()
        col.addStretch(1)
        col.addWidget(reset)
        row.addLayout(col)

    def _input(self, p):
        if p.kind == "int" and p.high <= 2**31 - 1:
            w = QSpinBox()
            w.setRange(int(p.low), int(p.high))
            w.setValue(p.default)
        elif p.kind == "float":
            w = QDoubleSpinBox()
            w.setDecimals(p.decimals)
            low = p.low + 10 ** -p.decimals if p.low_open else p.low
            w.setRange(low, p.high)
            w.setSingleStep(p.step or 0.1)
            w.setValue(p.default)
        elif p.kind == "bits":
            w = QWidget()
            lay = QHBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(p.default)
            button = QPushButton("Losuj")
            button.clicked.connect(lambda: edit.setText("".join(random.choice("01") for _ in range(max(len(edit.text()), 8)))))
            lay.addWidget(edit)
            lay.addWidget(button)
            self.inputs[p.name] = edit
            return w
        else:  # text, or a key larger than a spin box can hold
            w = QLineEdit(str(p.default))
        self.inputs[p.name] = w
        return w

    def values(self):
        return {name: w.text() if isinstance(w, QLineEdit) else w.value() for name, w in self.inputs.items()}

    def set_values(self, values):
        for name, value in values.items():
            w = self.inputs.get(name)
            if isinstance(w, QLineEdit):
                w.setText(str(value))
            elif w is not None:
                w.setValue(value)

    def checked(self):
        return self.algo.check(self.values())


class ParamStack(QWidget):
    """One ParamForm per algorithm, the one of the selected algorithm is visible."""

    def __init__(self, combo):
        super().__init__()
        self.combo = combo
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.forms = {}
        for algo in algorithms.ALGORITHMS.values():
            self.forms[algo.id] = ParamForm(algo)
            lay.addWidget(self.forms[algo.id])
        combo.currentIndexChanged.connect(self._show)
        self._show()

    def _show(self):
        for algo_id, form in self.forms.items():
            form.setVisible(algo_id == self.combo.currentData())

    def current(self):
        return self.combo.currentData(), self.forms[self.combo.currentData()].checked()


# ---------------------------------------------------------------- colored tokens
def legend_text(algo_id):
    green, red = {"synthid": ("średnia g > 0.5", "średnia g < 0.5"),
                  "mpac": ("lista odczytanej cyfry", "inna lista"),
                  "mptour": ("średnia g > 0.5", "średnia g < 0.5")}.get(algo_id, ("lista zielona", "lista czerwona"))
    green, red = html.escape(green), html.escape(red)  # "<" in the label would cut the rest of the text
    return (f"<span style='background:{GREEN_BG}; color:{TOKEN_TEXT}'>&nbsp;{green}&nbsp;</span>&nbsp; "
            f"<span style='background:{RED_BG}; color:{TOKEN_TEXT}'>&nbsp;{red}&nbsp;</span>&nbsp; "
            f"<span style='color:{NOT_SCORED}'>szary – token nieoceniany</span>")


class TokenView(QTextBrowser):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(150)
        self.setStyleSheet(f"QTextBrowser {{ background: white; color: {TOKEN_TEXT}; }}")  # also in dark mode
        self._data = None
        self.colored = True

    def show_tokens(self, pieces, annotation):
        self._data = (pieces, annotation)
        self._render()

    def set_colored(self, on):
        self.colored = on
        self._render()

    def _render(self):
        if self._data is None:
            self.clear()
            return
        pieces, annotation = self._data
        parts = []
        for piece, value in zip(pieces, annotation):
            text = html.escape(piece)
            if not self.colored:
                parts.append(text)
            elif value is None:
                parts.append(f"<span style='color:{NOT_SCORED}'>{text}</span>")
            elif value > 0.5:
                parts.append(f"<span style='background:{GREEN_BG}; color:{TOKEN_TEXT}'>{text}</span>")
            elif value < 0.5:
                parts.append(f"<span style='background:{RED_BG}; color:{TOKEN_TEXT}'>{text}</span>")
            else:
                parts.append(text)
        self.setHtml(f"<div style='white-space: pre-wrap; font-size: 10.5pt'>{''.join(parts)}</div>")


# ---------------------------------------------------------------- charts
class ChartView(QWidget):
    def __init__(self, min_height=280):
        super().__init__()
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.canvas = None
        self.min_height = min_height

    def show_figure(self, fig):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        if self.canvas is not None:
            self.lay.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumHeight(self.min_height)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.lay.addWidget(self.canvas)


# ---------------------------------------------------------------- progress
class ProgressRow(QWidget):
    stop_clicked = Signal()

    def __init__(self):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setMaximumHeight(14)
        self.label = QLabel()
        self.stop = QPushButton("Przerwij")
        self.stop.clicked.connect(self._stop)
        lay.addWidget(self.bar, 1)
        lay.addWidget(self.label, 2)
        lay.addWidget(self.stop)
        self.hide()

    def start(self, text):
        self.bar.setValue(0)
        self.label.setText(text)
        self.stop.setEnabled(True)
        self.show()

    def update_progress(self, fraction, text):
        self.bar.setValue(int(1000 * max(0.0, min(1.0, fraction))))
        self.label.setText(text)

    def _stop(self):
        self.stop.setEnabled(False)
        self.label.setText(self.label.text() + " – przerywanie po bieżącym tekście…")
        self.stop_clicked.emit()


# ---------------------------------------------------------------- export
class ExportButtons(QWidget):
    """Save the shown experiment as PDF / CSV / JSON."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.record = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel("Zapisz wyniki:"))
        self.buttons = []
        for ext in ("pdf", "csv", "json"):
            b = QPushButton(ext.upper())
            b.clicked.connect(lambda _=False, e=ext: self.export(e))
            lay.addWidget(b)
            self.buttons.append(b)
        self.status = QLabel()
        lay.addWidget(self.status, 1)
        self.set_record(None)

    def set_record(self, record):
        self.record = record
        for b in self.buttons:
            b.setEnabled(bool(record and record.get("result")))
        self.status.setText("")

    def export(self, ext):
        name = f"{self.record['kind']}_{self.record['id']}.{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Zapisz wyniki", str(Path.home() / name), f"{ext.upper()} (*.{ext})")
        if not path:
            return
        if not path.lower().endswith("." + ext):
            path += "." + ext
        fn = {"pdf": reports.export_pdf, "csv": reports.export_csv, "json": reports.export_json}[ext]
        record = self.record
        job = self.state.start_job(lambda progress, stop: fn(record, path))
        if job is None:
            self.status.setText("Poczekaj na koniec bieżącego zadania.")
            return
        job.succeeded.connect(lambda p: self.status.setText(f"Zapisano: {p}"))
        job.failed.connect(lambda exc, tb: show_error(self, exc, tb))


# ---------------------------------------------------------------- test settings
class SettingsBox(QGroupBox):
    """Settings shared by the test and comparison tabs."""

    def __init__(self, prompts=20):
        super().__init__("Ustawienia")
        grid = QGridLayout(self)
        self.source = QComboBox()
        self.source.addItems(["20 wbudowanych poleceń (jak w demach)", "z pliku .txt (jedno polecenie w linii)"])
        self.file = QLineEdit()
        self.file.setPlaceholderText("plik z poleceniami")
        browse = QPushButton("Wybierz…")
        browse.clicked.connect(self._browse)
        self.count = QSpinBox()
        self.count.setRange(1, 10000)
        self.count.setValue(prompts)
        self.tokens = QSpinBox()
        self.tokens.setRange(10, 2000)
        self.tokens.setValue(150)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.detect_on = QComboBox()
        self.detect_on.addItem("tokenach z generacji", "tokens")
        self.detect_on.addItem("tekście po ponownej tokenizacji", "text")
        self.edits = QLineEdit("0, 0.1, 0.2, 0.3, 0.5")
        self.edits.setToolTip("Odsetki tokenów zmienianych losowo (wstawienie, usunięcie, zamiana)")
        self.perplexity = QCheckBox("licz perplexity")
        self.perplexity.setChecked(True)
        self.other_key = QCheckBox("sprawdź też innym kluczem")
        self.other_key.setChecked(True)
        grid.addWidget(QLabel("Polecenia:"), 0, 0)
        grid.addWidget(self.source, 0, 1)
        grid.addWidget(self.file, 0, 2)
        grid.addWidget(browse, 0, 3)
        grid.addWidget(QLabel("Liczba poleceń:"), 1, 0)
        grid.addWidget(self.count, 1, 1)
        grid.addWidget(QLabel("Maks. tokenów:"), 1, 2)
        grid.addWidget(self.tokens, 1, 3)
        grid.addWidget(QLabel("Ziarno:"), 2, 0)
        grid.addWidget(self.seed, 2, 1)
        grid.addWidget(QLabel("Detekcja na:"), 2, 2)
        grid.addWidget(self.detect_on, 2, 3)
        grid.addWidget(QLabel("Losowe zmiany:"), 3, 0)
        grid.addWidget(self.edits, 3, 1)
        grid.addWidget(self.perplexity, 3, 2)
        grid.addWidget(self.other_key, 3, 3)
        self.source.currentIndexChanged.connect(lambda i: (self.file.setEnabled(i == 1), browse.setEnabled(i == 1)))
        self.file.setEnabled(False)
        browse.setEnabled(False)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Plik z poleceniami", "", "Tekst (*.txt);;Wszystkie pliki (*)")
        if path:
            self.file.setText(path)

    def collect(self):
        """(settings, files used). Raises AppError on bad input."""
        files = []
        if self.source.currentIndex() == 1:
            prompts = experiments.read_prompts_file(self.file.text())
            files.append(self.file.text())
        else:
            prompts = experiments.PROMPTS
        try:
            fractions = [float(x) for x in self.edits.text().replace(";", ",").split(",") if x.strip()]
        except ValueError:
            raise AppError("Niepoprawna lista losowych zmian.", "Wpisz liczby od 0 do 1 oddzielone przecinkami.")
        settings = {"prompts": prompts[:self.count.value()], "max_new_tokens": self.tokens.value(),
                    "seed": self.seed.value(), "detect_on": self.detect_on.currentData(),
                    "edit_fractions": fractions, "perplexity": self.perplexity.isChecked(),
                    "other_key": self.other_key.isChecked()}
        return experiments.check_settings(settings), files


# ---------------------------------------------------------------- base class of experiment tabs
class ExperimentTab(QWidget):
    """Runs an experiment in the background, shows progress and saves it in the history."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.run_buttons = []
        state.busy_changed.connect(self.update_buttons)
        state.model_changed.connect(self.update_buttons)

    def update_buttons(self, *args):
        for b in self.run_buttons:
            b.setEnabled(self.state.ready)
            b.setToolTip("" if self.state.model else "Najpierw wczytaj model (Plik → Wczytaj model)")

    def run(self, kind, title, algos, params, files, fn, on_result, progress_row):
        lm = self.state.model
        record = history.new_record(kind, title, algos, params, files, lm.describe())
        job = self.state.start_job(fn)
        if job is None:
            return
        progress_row.start(title + "…")
        if getattr(progress_row, "connection", None) is not None:
            progress_row.stop_clicked.disconnect(progress_row.connection)
        progress_row.connection = progress_row.stop_clicked.connect(job.cancel)
        job.progress.connect(progress_row.update_progress)

        def finish(status, **fields):
            record.update(status=status, duration_s=job.elapsed(), **fields)
            try:
                history.save(record)
                self.state.history_changed.emit()
            except OSError as exc:
                show_error(self, AppError(f"Nie udało się zapisać historii: {exc}"))

        def ok(result):
            finish("ok", result=result)
            on_result(result, record)

        def failed(exc, tb):
            finish("error", error={"message": str(exc), "traceback": tb})
            show_error(self, exc, tb)

        def cancelled(exc):
            done = f" po {exc.done} poleceniach" if exc.done is not None else ""
            finish("cancelled", error={"message": f"Przerwano{done}."})
            progress_row.label.setText(f"Przerwano{done}. Wpis jest w historii.")

        job.succeeded.connect(ok)
        job.failed.connect(failed)
        job.cancelled.connect(cancelled)
        job.finished.connect(lambda: progress_row.setVisible(progress_row.label.text().startswith("Przerwano")))
