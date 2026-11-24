"""Main window: tabs, menu, model loading."""
import sys
import traceback

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QTabWidget,
                               QVBoxLayout)

from .. import APP_NAME, VERSION
from ..model import DEFAULT_MODEL, SUGGESTED_MODELS, load_model
from .tab_experiments import CompareTab, DemoTab, DetectTab, TestTab
from .tab_info import AlgorithmsTab, HelpTab, HistoryTab
from .widgets import show_error
from .worker import AppState

STYLE = """
QGroupBox { font-weight: bold; margin-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QPushButton { padding: 4px 12px; }
QPushButton#start { font-weight: bold; padding: 5px 18px; }
"""


class ModelDialog(QDialog):
    def __init__(self, parent, current):
        super().__init__(parent)
        self.setWindowTitle("Wczytaj model")
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Nazwa modelu z huggingface.co albo folder z pobranym modelem:"))
        row = QHBoxLayout()
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.addItems(SUGGESTED_MODELS)
        self.name.setEditText(current)
        self.name.setMinimumWidth(360)
        folder = QPushButton("Folder…")
        folder.clicked.connect(self._folder)
        row.addWidget(self.name, 1)
        row.addWidget(folder)
        lay.addLayout(row)
        self.chat = QCheckBox("Szablon czatu (wyłącz dla modeli bazowych, np. OPT, GPT-2)")
        self.chat.setChecked(True)
        lay.addWidget(self.chat)
        lay.addWidget(QLabel("Za pierwszym razem model jest pobierany z internetu (Qwen2.5-0.5B: ok. 1 GB)."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _folder(self):
        path = QFileDialog.getExistingDirectory(self, "Folder z modelem")
        if path:
            self.name.setEditText(path)


class MainWindow(QMainWindow):
    def __init__(self, state=None):
        super().__init__()
        self.state = state or AppState()
        self.settings = QSettings("Steganografia", "watermark-app")
        self.setWindowTitle(f"{APP_NAME} {VERSION}")
        self.resize(1280, 860)

        self.tabs = QTabWidget()
        self.pages = {"algorithms": AlgorithmsTab(self.state), "demo": DemoTab(self.state),
                      "detect": DetectTab(self.state), "test": TestTab(self.state),
                      "compare": CompareTab(self.state), "history": HistoryTab(self.state),
                      "help": HelpTab(self.state)}
        titles = {"algorithms": "Algorytmy", "demo": "Demonstracja", "detect": "Sprawdź tekst", "test": "Test",
                  "compare": "Porównanie", "history": "Historia", "help": "Pomoc"}
        for key, page in self.pages.items():
            self.tabs.addTab(page, titles[key])
        self.setCentralWidget(self.tabs)

        menu = self.menuBar().addMenu("Plik")
        menu.addAction("Wczytaj model…", self.choose_model)
        menu.addSeparator()
        menu.addAction("Zakończ", self.close)
        help_menu = self.menuBar().addMenu("Pomoc")
        help_menu.addAction("Jak korzystać z aplikacji", lambda: self.show_tab("help"))
        help_menu.addAction("O programie", self.about)

        self.model_label = QLabel("Model: nie wczytano")
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(120)
        self.busy.setMaximumHeight(12)
        self.busy.hide()
        self.statusBar().addWidget(self.model_label, 1)
        self.statusBar().addPermanentWidget(self.busy)
        self.state.model_changed.connect(self._model_changed)
        self.state.busy_changed.connect(self.busy.setVisible)
        if self.state.model is not None:
            self._model_changed(self.state.model)

    def show_tab(self, key):
        self.tabs.setCurrentWidget(self.pages[key])

    def choose_model(self):
        current = self.state.model.name if self.state.model else self.settings.value("model", DEFAULT_MODEL)
        dialog = ModelDialog(self, current)
        if dialog.exec() == QDialog.Accepted:
            self.load(dialog.name.currentText().strip(), dialog.chat.isChecked())

    def load(self, name, chat_template=True):
        job = self.state.start_job(lambda progress, stop: load_model(name, chat_template))
        if job is None:
            QMessageBox.information(self, APP_NAME, "Poczekaj, aż skończy się bieżące zadanie.")
            return
        self.model_label.setText(f"Ładowanie modelu {name}… (za pierwszym razem pobieranie może potrwać kilka minut)")
        job.succeeded.connect(lambda lm: (self.settings.setValue("model", name), self.state.set_model(lm)))
        job.failed.connect(self._load_failed)

    def _load_failed(self, exc, tb):
        self._model_changed(self.state.model)
        show_error(self, exc, tb)

    def _model_changed(self, lm):
        if lm is None:
            self.model_label.setText("Model: nie wczytano (Plik → Wczytaj model)")
        else:
            self.model_label.setText(f"Model: {lm.name} · {lm.device} · słownik {lm.vocab_size} tokenów")

    def about(self):
        QMessageBox.about(self, APP_NAME,
                          f"<b>{APP_NAME} {VERSION}</b><br>Aplikacja do demonstracji i testowania algorytmów "
                          "znakowania wodnego tekstu z repozytorium Steganografia (Unigram-Watermark, SynthID-Text, "
                          "KGW, DiPmark, MPAC, MPTour).<br>Projekt inżynierski.")

    def closeEvent(self, event):
        if self.state.job is not None:
            self.state.job.cancel()
            self.state.job.wait(15000)
        super().closeEvent(event)


def run():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    # always light: since Qt 6.5 Fusion follows the dark theme of Windows and then the text on the colored
    # tokens was light on a light background
    if hasattr(app.styleHints(), "setColorScheme"):  # Qt >= 6.8
        app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    app.setPalette(app.style().standardPalette())
    app.setStyleSheet(STYLE)

    def excepthook(kind, exc, tb):  # last resort: an error in the GUI thread does not close the app
        sys.stderr.write("".join(traceback.format_exception(kind, exc, tb)))
        show_error(None, exc, "".join(traceback.format_exception(kind, exc, tb)))

    sys.excepthook = excepthook
    window = MainWindow()
    window.show()
    QTimer.singleShot(0, lambda: window.load(window.settings.value("model", DEFAULT_MODEL)))
    return app.exec()
