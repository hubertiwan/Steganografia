"""Tabs that run experiments: demo, check a text, test series, comparison."""
import difflib
import time

from PySide6.QtWidgets import (QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
                               QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .. import algorithms, experiments
from ..errors import AppError
from .results import ComparisonResultView, DemoResultView, DetectResultView, TestResultView
from .widgets import (ExperimentTab, ExportButtons, ParamForm, ParamStack, ProgressRow, SettingsBox,
                      algorithm_combo, show_error)


def scrolled(widget):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(widget)
    return area


def algorithm_row(combo):
    row = QHBoxLayout()
    row.addWidget(QLabel("Algorytm:"))
    row.addWidget(combo)
    row.addStretch(1)
    return row


def load_text(parent, edit):
    path, _ = QFileDialog.getOpenFileName(parent, "Wczytaj tekst", "", "Tekst (*.txt);;Wszystkie pliki (*)")
    if not path:
        return None
    try:
        edit.setPlainText(experiments.read_text_file(path).strip())
        return path
    except AppError as exc:
        show_error(parent, exc)
        return None


# ---------------------------------------------------------------- demo
class DemoTab(ExperimentTab):
    def __init__(self, state):
        super().__init__(state)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        self.algo = algorithm_combo()
        lay.addLayout(algorithm_row(self.algo))
        self.params = ParamStack(self.algo)
        lay.addWidget(self.params)

        box = QGroupBox("Polecenie dla modelu")
        col = QVBoxLayout(box)
        self.prompt = QPlainTextEdit("Is it possible to fail several times and succeed once?")
        self.prompt.setMaximumHeight(70)
        col.addWidget(self.prompt)
        row = QHBoxLayout()
        row.addWidget(QLabel("Maks. tokenów:"))
        self.tokens = QSpinBox()
        self.tokens.setRange(10, 2000)
        self.tokens.setValue(150)
        row.addWidget(self.tokens)
        row.addWidget(QLabel("Ziarno:"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.seed.setValue(1)
        row.addWidget(self.seed)
        row.addStretch(1)
        from_file = QPushButton("Wczytaj z pliku…")
        from_file.clicked.connect(lambda: load_text(self, self.prompt))
        self.go = QPushButton("Generuj i sprawdź")
        self.go.setObjectName("start")
        self.go.clicked.connect(self.start)
        self.run_buttons.append(self.go)
        row.addWidget(from_file)
        row.addWidget(self.go)
        col.addLayout(row)
        lay.addWidget(box)
        self.progress = ProgressRow()
        lay.addWidget(self.progress)

        self.result_box = QGroupBox("Wynik")
        col = QVBoxLayout(self.result_box)
        self.view = DemoResultView()
        col.addWidget(self.view)
        self.export = ExportButtons(state)
        col.addWidget(self.export)
        self.result_box.hide()
        lay.addWidget(self.result_box)

        self.edit_box = QGroupBox("Zmień tekst ze znakiem i sprawdź jeszcze raz")
        col = QVBoxLayout(self.edit_box)
        col.addWidget(QLabel("Możesz poprawić tekst ręcznie albo losowo zmienić część tokenów."))
        self.edit_text = QPlainTextEdit()
        self.edit_text.setMinimumHeight(110)
        col.addWidget(self.edit_text)
        row = QHBoxLayout()
        row.addWidget(QLabel("Losowo zmień"))
        self.edit_percent = QSpinBox()
        self.edit_percent.setRange(0, 100)
        self.edit_percent.setValue(20)
        self.edit_percent.setSuffix(" %")
        row.addWidget(self.edit_percent)
        random_button = QPushButton("tokenów")
        random_button.clicked.connect(self.random_edit)
        row.addWidget(random_button)
        restore = QPushButton("Przywróć oryginał")
        restore.clicked.connect(lambda: self.edit_text.setPlainText(self.last["watermarked"]["text"]))
        row.addWidget(restore)
        row.addStretch(1)
        self.recheck = QPushButton("Sprawdź")
        self.recheck.clicked.connect(self.check_edited)
        self.run_buttons.append(self.recheck)
        row.addWidget(self.recheck)
        col.addLayout(row)
        self.edit_result = QLabel()
        self.edit_result.setWordWrap(True)
        col.addWidget(self.edit_result)
        self.edit_box.hide()
        lay.addWidget(self.edit_box)
        lay.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(inner))
        self.last = None
        self.update_buttons()

    def select(self, algo_id):
        self.algo.setCurrentIndex(self.algo.findData(algo_id))

    def start(self):
        try:
            algo_id, params = self.params.current()
            prompt = self.prompt.toPlainText().strip()
            if not prompt:
                raise AppError("Polecenie jest puste.", "Wpisz polecenie dla modelu.")
        except AppError as exc:
            show_error(self, exc)
            return
        lm, n, seed = self.state.model, self.tokens.value(), self.seed.value()
        title = f"{algorithms.get(algo_id).name}: {prompt[:60]}"
        self.run("demo", title, [algo_id], params, [],
                 lambda progress, stop: experiments.run_demo(lm, algo_id, params, prompt, n, seed, progress, stop),
                 self.show_result, self.progress)

    def show_result(self, result, record):
        self.last = result
        self.view.set_result(result)
        self.export.set_record(record)
        self.result_box.show()
        self.edit_text.setPlainText(result["watermarked"]["text"])
        self.edit_result.setText("")
        self.edit_box.show()

    def random_edit(self):
        if self.last:
            try:
                self.edit_text.setPlainText(experiments.edit_text(self.state.model, self.edit_text.toPlainText(),
                                                                  self.edit_percent.value() / 100,
                                                                  seed=int(time.time())))
            except AppError as exc:
                show_error(self, exc)

    def check_edited(self):
        if not self.last:
            return
        if self.state.model.name != self.last["model"]["name"]:
            show_error(self, AppError("Model został zmieniony.", "Wygeneruj tekst jeszcze raz tym modelem."))
            return
        lm, last = self.state.model, self.last
        original, edited = last["watermarked"]["text"], self.edit_text.toPlainText()

        def work(progress, stop):
            return (experiments.detect_text(lm, last["algorithm"], last["params"], original),
                    experiments.detect_text(lm, last["algorithm"], last["params"], edited))

        job = self.state.start_job(work)
        if job is None:
            return
        job.succeeded.connect(self._show_edited)
        job.failed.connect(lambda exc, tb: show_error(self, exc, tb))

    def _show_edited(self, pair):
        before, after = pair
        algo = algorithms.get(before["algorithm"])
        a, b = before["tokens"], after["tokens"]
        kept = sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
        d0, d1 = before["detection"], after["detection"]
        text = (f"Przed zmianą: {algo.score_name} = {d0['score']:.2f}, {'wykryty' if d0['detected'] else 'nie wykryty'}. "
                f"Po zmianie: {algo.score_name} = {d1['score']:.2f}, "
                f"<b>{'wykryty' if d1['detected'] else 'nie wykryty'}</b>. "
                f"Niezmienionych tokenów: {kept} z {len(a)}.")
        if algo.multibit:
            text += f" Odczytana wiadomość: {d1['message']} (zgodność {d1['bit_accuracy']:.0%})."
        self.edit_result.setText(text)


# ---------------------------------------------------------------- check any text
class DetectTab(ExperimentTab):
    def __init__(self, state):
        super().__init__(state)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        self.algo = algorithm_combo()
        lay.addLayout(algorithm_row(self.algo))
        self.params = ParamStack(self.algo)
        lay.addWidget(self.params)
        box = QGroupBox("Tekst do sprawdzenia")
        col = QVBoxLayout(box)
        col.addWidget(QLabel("Klucz i parametry muszą być takie same jak przy generowaniu, model (tokenizer) też."))
        self.text = QPlainTextEdit()
        self.text.setMinimumHeight(150)
        col.addWidget(self.text)
        row = QHBoxLayout()
        from_file = QPushButton("Wczytaj z pliku…")
        from_file.clicked.connect(self._load)
        row.addWidget(from_file)
        row.addStretch(1)
        self.go = QPushButton("Sprawdź")
        self.go.setObjectName("start")
        self.go.clicked.connect(self.start)
        self.run_buttons.append(self.go)
        row.addWidget(self.go)
        col.addLayout(row)
        lay.addWidget(box)
        self.progress = ProgressRow()
        lay.addWidget(self.progress)
        self.result_box = QGroupBox("Wynik")
        col = QVBoxLayout(self.result_box)
        self.view = DetectResultView()
        self.view.setMinimumHeight(380)
        col.addWidget(self.view)
        self.export = ExportButtons(state)
        col.addWidget(self.export)
        self.result_box.hide()
        lay.addWidget(self.result_box)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(inner))
        self.file = None
        self.update_buttons()

    def _load(self):
        self.file = load_text(self, self.text) or self.file

    def start(self):
        try:
            algo_id, params = self.params.current()
        except AppError as exc:
            show_error(self, exc)
            return
        lm, text = self.state.model, self.text.toPlainText()
        if not text.strip():
            show_error(self, AppError("Tekst jest pusty.", "Wklej tekst albo wczytaj go z pliku."))
            return
        self.run("detect", f"{algorithms.get(algo_id).name}: sprawdzenie tekstu", [algo_id], params,
                 [self.file] if self.file else [],
                 lambda progress, stop: experiments.detect_text(lm, algo_id, params, text),
                 self.show_result, self.progress)

    def show_result(self, result, record):
        self.view.set_result(result)
        self.export.set_record(record)
        self.result_box.show()


# ---------------------------------------------------------------- test series
class TestTab(ExperimentTab):
    def __init__(self, state):
        super().__init__(state)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        self.algo = algorithm_combo()
        lay.addLayout(algorithm_row(self.algo))
        self.params = ParamStack(self.algo)
        lay.addWidget(self.params)
        self.settings = SettingsBox()
        lay.addWidget(self.settings)
        row = QHBoxLayout()
        self.info = QLabel()
        row.addWidget(self.info, 1)
        self.go = QPushButton("Uruchom test")
        self.go.setObjectName("start")
        self.go.clicked.connect(self.start)
        self.run_buttons.append(self.go)
        row.addWidget(self.go)
        lay.addLayout(row)
        self.progress = ProgressRow()
        lay.addWidget(self.progress)
        self.result_box = QGroupBox("Wynik")
        col = QVBoxLayout(self.result_box)
        self.view = TestResultView()
        self.view.setMinimumHeight(620)
        col.addWidget(self.view)
        self.export = ExportButtons(state)
        col.addWidget(self.export)
        self.result_box.hide()
        lay.addWidget(self.result_box)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(inner))
        self.settings.count.valueChanged.connect(self._info)
        self.settings.tokens.valueChanged.connect(self._info)
        state.model_changed.connect(self._info)
        self._info()
        self.update_buttons()

    def _info(self, *args):
        n = self.settings.count.value()
        text = f"Zostanie wygenerowanych {2 * n} tekstów (ze znakiem i bez)."
        if self.state.model is not None and self.state.model.device == "cpu":
            text += " Na procesorze może to potrwać kilka minut."
        self.info.setText(text)

    def select(self, algo_id):
        self.algo.setCurrentIndex(self.algo.findData(algo_id))

    def start(self):
        try:
            algo_id, params = self.params.current()
            settings, files = self.settings.collect()
        except AppError as exc:
            show_error(self, exc)
            return
        lm = self.state.model
        title = f"{algorithms.get(algo_id).name}: {len(settings['prompts'])} poleceń, {settings['max_new_tokens']} tokenów"
        self.run("test", title, [algo_id], params, files,
                 lambda progress, stop: experiments.run_test(lm, algo_id, params, settings, progress, stop),
                 self.show_result, self.progress)

    def show_result(self, result, record):
        self.view.set_result(result)
        self.export.set_record(record)
        self.result_box.show()


# ---------------------------------------------------------------- comparison
class CompareTab(ExperimentTab):
    def __init__(self, state):
        super().__init__(state)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.addWidget(QLabel("Zaznacz algorytmy do porównania. Wszystkie dostaną ten sam model, te same polecenia, "
                             "ziarna i długość tekstu."))
        self.pages = QTabWidget()
        self.include = {}
        self.forms = {}
        for algo in algorithms.ALGORITHMS.values():
            page = QWidget()
            col = QVBoxLayout(page)
            self.include[algo.id] = QCheckBox(f"Porównuj {algo.name}")
            self.include[algo.id].setChecked(True)
            self.forms[algo.id] = ParamForm(algo)
            col.addWidget(self.include[algo.id])
            col.addWidget(self.forms[algo.id])
            self.pages.addTab(page, algo.name)
        lay.addWidget(self.pages)
        self.settings = SettingsBox(prompts=10)
        lay.addWidget(self.settings)
        row = QHBoxLayout()
        row.addStretch(1)
        self.go = QPushButton("Uruchom porównanie")
        self.go.setObjectName("start")
        self.go.clicked.connect(self.start)
        self.run_buttons.append(self.go)
        row.addWidget(self.go)
        lay.addLayout(row)
        self.progress = ProgressRow()
        lay.addWidget(self.progress)
        self.result_box = QGroupBox("Wynik")
        col = QVBoxLayout(self.result_box)
        self.view = ComparisonResultView()
        self.view.setMinimumHeight(620)
        col.addWidget(self.view)
        self.export = ExportButtons(state)
        col.addWidget(self.export)
        self.result_box.hide()
        lay.addWidget(self.result_box)
        lay.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(inner))
        self.update_buttons()

    def select(self, algo_id):
        self.pages.setCurrentIndex(list(algorithms.ALGORITHMS).index(algo_id))

    def start(self):
        try:
            configs = [(a, self.forms[a].checked()) for a, box in self.include.items() if box.isChecked()]
            if len(configs) < 2:
                raise AppError("Zaznacz co najmniej dwa algorytmy.")
            settings, files = self.settings.collect()
        except AppError as exc:
            show_error(self, exc)
            return
        lm = self.state.model
        names = ", ".join(algorithms.get(a).name for a, _ in configs)
        self.run("comparison", f"Porównanie: {names}", [a for a, _ in configs], dict(configs), files,
                 lambda progress, stop: experiments.run_comparison(lm, configs, settings, progress, stop),
                 self.show_result, self.progress)

    def show_result(self, result, record):
        self.view.set_result(result)
        self.export.set_record(record)
        self.result_box.show()
