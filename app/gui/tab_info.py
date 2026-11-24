"""Tabs without experiments: algorithm descriptions, history, help."""
import html
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTabWidget,
                               QTextBrowser, QVBoxLayout, QWidget)

from .. import algorithms, history
from ..errors import AppError
from ..paths import DOCS_DIR
from .results import view_for
from .widgets import GREEN_BG, NOT_SCORED, RED_BG, TOKEN_TEXT, ExportButtons, fill_table, fit_table_height, make_table, show_error


def params_text(params):
    """{'key': 1, 'gamma': 0.5} -> 'key = 1, gamma = 0.5'; for a comparison one line per algorithm."""
    if params and all(isinstance(v, dict) for v in params.values()):
        return "\n".join(f"{algorithms.get(a).name if a in algorithms.ALGORITHMS else a}: {params_text(p)}"
                         for a, p in params.items())
    return ", ".join(f"{k} = {v}" for k, v in params.items())


def color_icon(color):
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


# ---------------------------------------------------------------- algorithms
class AlgorithmsTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        lay = QHBoxLayout(self)
        self.list = QListWidget()
        self.list.setMaximumWidth(230)
        for algo in algorithms.ALGORITHMS.values():
            item = QListWidgetItem(color_icon(algo.color), algo.name)
            item.setData(Qt.UserRole, algo.id)
            self.list.addItem(item)
        lay.addWidget(self.list)
        self.tabs = QTabWidget()
        self.description = QTextBrowser()
        self.description.setOpenExternalLinks(True)
        self.note = QTextBrowser()
        self.tabs.addTab(self.description, "Opis")
        self.tabs.addTab(self.note, "Notatka z docs/")
        lay.addWidget(self.tabs, 1)
        self.list.currentItemChanged.connect(self.show_algorithm)
        self.list.setCurrentRow(0)

    def select(self, algo_id):
        self.list.setCurrentRow(list(algorithms.ALGORITHMS).index(algo_id))

    def show_algorithm(self, item, previous=None):
        if item is None:
            return
        algo = algorithms.get(item.data(Qt.UserRole))
        e = html.escape
        parts = [f"<h2>{e(algo.name)}</h2>", f"<p><i>{e(algo.paper)}</i></p>", f"<p>{e(algo.summary)}</p>",
                 "<h3>Jak działa</h3><ol>"] + [f"<li>{e(s)}</li>" for s in algo.steps] + ["</ol>"]
        parts.append("<h3>Parametry (domyślne wartości z pracy)</h3>"
                     "<table border='1' cellspacing='0' cellpadding='4'><tr><th>Parametr</th><th>Domyślnie</th>"
                     "<th>Opis</th></tr>")
        for p in algo.params:
            parts.append(f"<tr><td>{e(p.label)}</td><td>{e(str(p.default))}</td><td>{e(p.help)}</td></tr>")
        parts.append("</table>")
        if algo.limitations:
            parts.append("<h3>Uwagi</h3><ul>" + "".join(f"<li>{e(x)}</li>" for x in algo.limitations) + "</ul>")
        self.description.setHtml("".join(parts))
        path = DOCS_DIR / algo.doc
        self.note.setMarkdown(path.read_text(encoding="utf-8") if path.exists() else f"Brak pliku {path}")


# ---------------------------------------------------------------- history
class HistoryTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.current = None
        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItem("Wszystkie", None)
        for key, label in history.KINDS.items():
            self.kind.addItem(label, key)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Szukaj…")
        row.addWidget(QLabel("Rodzaj:"))
        row.addWidget(self.kind)
        row.addWidget(self.search, 1)
        lay.addLayout(row)
        split = QSplitter(Qt.Vertical)
        self.table = make_table()
        self.table.itemSelectionChanged.connect(self._selected)
        split.addWidget(self.table)
        detail = QWidget()
        col = QVBoxLayout(detail)
        col.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.export = ExportButtons(state)
        row.addWidget(self.export, 1)
        self.delete_button = QPushButton("Usuń wpis")
        self.delete_button.clicked.connect(self._delete)
        row.addWidget(self.delete_button)
        col.addLayout(row)
        self.info = make_table()
        col.addWidget(self.info)
        self.holder = QVBoxLayout()
        col.addLayout(self.holder)
        col.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(detail)
        split.addWidget(area)
        split.setSizes([220, 600])
        lay.addWidget(split, 1)
        self.kind.currentIndexChanged.connect(self.refresh)
        self.search.textChanged.connect(self.refresh)
        state.history_changed.connect(self.refresh)
        self.refresh()
        self._show(None)

    def refresh(self):
        query = self.search.text().strip().lower()
        self.records = [r for r in history.list_records()
                        if (not self.kind.currentData() or r.get("kind") == self.kind.currentData())
                        and (not query or query in json.dumps(r, ensure_ascii=False).lower())]
        rows = [[r.get("created", "").replace("T", " "), history.KINDS.get(r.get("kind"), "?"), r.get("title", ""),
                 history.STATUS.get(r.get("status"), r.get("status"))] for r in self.records]
        self.table.blockSignals(True)
        fill_table(self.table, ["Data", "Rodzaj", "Opis", "Status"], rows)
        self.table.blockSignals(False)
        keep = self.current["id"] if self.current else None
        for i, r in enumerate(self.records):
            if r["id"] == keep:
                self.table.selectRow(i)

    def open(self, record_id):
        for i, r in enumerate(self.records):
            if r["id"] == record_id:
                self.table.selectRow(i)

    def _selected(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        rec = self.records[rows[0].row()]
        if self.current and self.current["id"] == rec["id"]:
            return
        try:
            self._show(history.load(rec["id"]))
        except (OSError, json.JSONDecodeError) as exc:
            self._show(None)
            show_error(self, AppError(f"Nie można odczytać wpisu: {exc}"))

    def _show(self, rec):
        self.current = rec
        while self.holder.count():
            w = self.holder.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.export.set_record(rec)
        self.delete_button.setEnabled(rec is not None)
        self.info.setVisible(rec is not None)
        if rec is None:
            return
        rows = [["Data", rec["created"].replace("T", " ")],
                ["Algorytmy", ", ".join(algorithms.get(a).name for a in rec["algorithms"] if a in algorithms.ALGORITHMS)],
                ["Parametry", params_text(rec.get("params", {}))],
                ["Model", (rec.get("model") or {}).get("name", "–")],
                ["Pliki", ", ".join(rec.get("files") or []) or "–"],
                ["Status", history.STATUS.get(rec["status"], rec["status"])],
                ["Czas", f"{rec['duration_s']:.1f} s" if rec.get("duration_s") else "–"]]
        if rec.get("error"):
            rows.append(["Komunikat", rec["error"].get("message", "")])
        fill_table(self.info, ["", ""], rows)
        self.info.horizontalHeader().setVisible(False)
        self.info.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.info.resizeRowsToContents()
        fit_table_height(self.info)
        if rec.get("result"):
            try:
                view = view_for(rec["result"])
                view.setMinimumHeight(560)
                self.holder.addWidget(view)
            except Exception as exc:  # e.g. a file saved by an older version
                self.holder.addWidget(QLabel(f"Nie można wyświetlić wyników: {type(exc).__name__}: {exc}"))
        if rec.get("error", {}) and rec["error"].get("traceback"):
            self.holder.addWidget(QPlainTextEdit(rec["error"]["traceback"], readOnly=True))

    def _delete(self):
        if self.current and QMessageBox.question(self, "Usuń wpis", f"Usunąć „{self.current['title']}”?") == QMessageBox.Yes:
            history.delete(self.current["id"])
            self._show(None)
            self.state.history_changed.emit()


# ---------------------------------------------------------------- help
HELP_INTRO = """
<h2>Pomoc</h2>
<p>Aplikacja pokazuje, jak działają znaki wodne w tekstach generowanych przez model językowy.
Znak jest dodawany <b>w trakcie generowania</b> (algorytm zmienia logity albo sposób losowania tokenów),
więc nie widać go w treści. Wykrywa go test statystyczny, który zna tajny klucz, ale nie potrzebuje modelu.</p>

<h3>Pierwsze kroki</h3>
<ol>
<li>Po starcie aplikacja w tle wczytuje model <i>Qwen/Qwen2.5-0.5B-Instruct</i>. Za pierwszym razem jest on
pobierany z internetu (ok. 1 GB), potem jest już na dysku.</li>
<li>Gdy pasek stanu na dole pokaże nazwę modelu, przyciski uruchamiania stają się aktywne.</li>
<li>Inny model albo folder z pobranym modelem wybierzesz w menu <b>Plik → Wczytaj model…</b></li>
<li>Najprościej zacząć od zakładki <b>Demonstracja</b>: wybierz algorytm i kliknij <b>Generuj i sprawdź</b>.</li>
</ol>

<h3>Zakładki</h3>
<table border='1' cellspacing='0' cellpadding='5'>
<tr><th>Zakładka</th><th>Co można zrobić</th></tr>
<tr><td><b>Algorytmy</b></td><td>Opis każdego algorytmu: skąd pochodzi, jak działa krok po kroku, parametry
z domyślnymi wartościami i uwagi. Podzakładka <i>Notatka z docs/</i> pokazuje dłuższy opis z repozytorium.</td></tr>
<tr><td><b>Demonstracja</b></td><td>Jedno polecenie dla modelu daje dwa teksty: ze znakiem i bez znaku. Tabela
pokazuje wynik detekcji obu tekstów i tekstu ze znakiem sprawdzonego innym kluczem, a tokeny są pokolorowane
według detektora. Dla algorytmów wielobitowych widać odczytaną wiadomość. Pod wynikiem tekst ze znakiem można
poprawić ręcznie albo kliknąć <b>Losowo zmień … tokenów</b>, a potem <b>Sprawdź</b>, żeby zobaczyć,
czy znak przetrwał zmiany.</td></tr>
<tr><td><b>Sprawdź tekst</b></td><td>Detekcja w dowolnym wklejonym albo wczytanym z pliku tekście. Klucz i
parametry muszą być takie same jak przy generowaniu, a model (tokenizer) ten sam.</td></tr>
<tr><td><b>Test</b></td><td>Seria poleceń (20 wbudowanych albo z pliku .txt, jedno w linii) dla jednego
algorytmu. Wynik: tabela metryk z 95% przedziałami ufności, odporność na losowe zmiany, wykresy (podzakładka
<i>Wykresy</i>) i wszystkie wygenerowane teksty (<i>Teksty</i>).</td></tr>
<tr><td><b>Porównanie</b></td><td>Kilka algorytmów na tym samym modelu, tych samych poleceniach, ziarnach i
długości tekstu. Zaznacz <b>Porównuj …</b> przy wybranych algorytmach (co najmniej dwóch). Wynik: wspólna tabela,
wykresy i szczegóły każdego algorytmu.</td></tr>
<tr><td><b>Historia</b></td><td>Każde uruchomienie, także przerwane albo zakończone błędem, z datą, parametrami
i wynikami. Stąd zapiszesz wynik jako <b>PDF</b> (raport z tabelami i wykresami), <b>CSV</b> albo <b>JSON</b>.</td></tr>
<tr><td><b>Pomoc</b></td><td>Ta strona.</td></tr>
</table>
"""

HELP_METRICS = """
<h3>Jak czytać wyniki</h3>
<table border='1' cellspacing='0' cellpadding='5'>
<tr><th>Pojęcie</th><th>Znaczenie</th></tr>
<tr><td>z</td><td>o ile odchyleń standardowych wynik tekstu przekracza to, czego można się spodziewać bez znaku;
bez znaku z jest zwykle blisko 0</td></tr>
<tr><td>p-wartość</td><td>prawdopodobieństwo takiego wyniku w tekście bez znaku; mała p-wartość oznacza znak</td></tr>
<tr><td>TPR</td><td>odsetek tekstów ze znakiem, które detektor wykrył</td></tr>
<tr><td>FPR</td><td>odsetek tekstów bez znaku uznanych za oznaczone (fałszywe alarmy)</td></tr>
<tr><td>AUC</td><td>szansa, że tekst ze znakiem dostanie wyższy wynik niż tekst bez znaku (1.0 = idealnie,
0.5 = zgadywanie); nie zależy od progu</td></tr>
<tr><td>TPR przy ustalonym FPR</td><td>skuteczność przy progu dobranym tak, by fałszywych alarmów było np. 10%;
pozwala porównać algorytmy o różnych progach</td></tr>
<tr><td>perplexity</td><td>jak bardzo model jest zaskoczony tekstem; wzrost względem tekstu bez znaku oznacza
gorszą jakość</td></tr>
<tr><td>dokładność bitowa</td><td>odsetek poprawnie odczytanych bitów wiadomości (tylko algorytmy wielobitowe)</td></tr>
</table>
<p>Algorytmy mają domyślne parametry ze swoich publikacji, więc progi i sposób próbkowania się różnią.
W porównaniu skuteczność najlepiej oceniać przez AUC i TPR przy ustalonym FPR.</p>
"""

HELP_TIPS = """
<h3>Wskazówki</h3>
<ul>
<li><b>Czas.</b> Na procesorze demonstracja trwa kilkanaście sekund, test na 20 poleceniach kilka minut,
porównanie wszystkich algorytmów nawet godzinę. Test i porównanie można przerwać przyciskiem <b>Przerwij</b>,
a okno cały czas działa.</li>
<li><b>Długość tekstu.</b> Sygnał rośnie z długością tekstu. Przy mniej niż ~50 tokenach znak bywa niewykrywalny,
a wiadomość algorytmu wielobitowego odczytana z błędami.</li>
<li><b>Powtarzalność.</b> To samo polecenie, ziarno i parametry dają ten sam tekst, więc każdy wynik można odtworzyć.</li>
<li><b>Inny klucz.</b> Wiersz „ze znakiem, inny klucz” pokazuje, że bez właściwego klucza znaku nie da się wykryć.</li>
<li><b>Szare przyciski.</b> Model jeszcze się wczytuje albo trwa inne zadanie (pasek w prawym dolnym rogu).</li>
<li><b>Model się nie wczytuje.</b> Sprawdź połączenie z internetem przy pierwszym uruchomieniu albo wskaż folder
z pobranym modelem w menu Plik.</li>
<li><b>Dane.</b> Historia jest zapisywana w folderze <code>app_data/</code> obok repozytorium.</li>
</ul>
"""


class HelpTab(QTextBrowser):
    """What the application can do. The list of algorithms comes from the registry, so it is always up to date."""

    def __init__(self, state=None):
        super().__init__()
        self.setOpenExternalLinks(True)
        self.setHtml(self.text())

    @staticmethod
    def text():
        e = html.escape
        rows = "".join(f"<tr><td><span style='color:{a.color}'>■</span> {e(a.name)}</td>"
                       f"<td>{'wiadomość b-bitowa' if a.multibit else '1 bit (czy jest znak)'}</td>"
                       f"<td>{e(a.paper)}</td></tr>" for a in algorithms.ALGORITHMS.values())
        algos = ("<h3>Algorytmy</h3><table border='1' cellspacing='0' cellpadding='5'>"
                 "<tr><th>Algorytm</th><th>Co ukrywa</th><th>Źródło</th></tr>" + rows + "</table>")
        colors = (
            "<h3>Kolory tokenów</h3><p>W wynikach demonstracji i sprawdzenia tekstu każdy token ma kolor według "
            "detektora:</p><p>"
            f"<span style='background:{GREEN_BG}; color:{TOKEN_TEXT}'>&nbsp;zielony&nbsp;</span> token „dobry” według "
            "klucza (lista zielona, g-wartość powyżej 0.5 albo lista odczytanej cyfry),<br>"
            f"<span style='background:{RED_BG}; color:{TOKEN_TEXT}'>&nbsp;czerwony&nbsp;</span> pozostałe tokeny,<br>"
            f"<span style='color:{NOT_SCORED}'>szary</span> token pomijany przez detektor (np. pierwsze tokeny albo "
            "powtórzony kontekst).</p><p>W tekście ze znakiem przeważa zielony, w tekście bez znaku kolory są mniej "
            "więcej po równo. Kolorowanie wyłącza pole <b>Koloruj tokeny</b>.</p>")
        return HELP_INTRO + algos + colors + HELP_METRICS + HELP_TIPS
