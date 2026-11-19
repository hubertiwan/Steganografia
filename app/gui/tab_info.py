"""Tabs without experiments: algorithm descriptions, history, PDF documents."""
import html
import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea, QSpinBox, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

from .. import algorithms, documents, history
from ..errors import AppError
from ..paths import DOCS_DIR
from .results import view_for
from .widgets import ExportButtons, fill_table, fit_table_height, make_table, show_error


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
        state.documents_changed.connect(lambda: self.show_algorithm(self.list.currentItem()))
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
        docs = documents.for_algorithm(algo.id)
        parts.append("<h3>Dokumenty źródłowe</h3>")
        if docs:
            parts.append("<ul>" + "".join(
                f"<li>{e(d['title'])} ({e(a['role'])}{', str. ' + e(a['pages']) if a['pages'] else ''})</li>"
                for d, a in docs) + "</ul>")
        else:
            parts.append("<p>Brak. Dokument PDF można dodać i przypisać w zakładce „Dokumenty PDF”.</p>")
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


# ---------------------------------------------------------------- PDF documents
class DocumentsTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.pdf = None
        self.doc = None
        self.page = 0
        self.hits = []
        lay = QHBoxLayout(self)

        left = QVBoxLayout()
        row = QHBoxLayout()
        add = QPushButton("Dodaj PDF…")
        add.clicked.connect(self.add)
        self.remove_button = QPushButton("Usuń")
        self.remove_button.clicked.connect(self.remove)
        row.addWidget(add)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        left.addLayout(row)
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._select)
        left.addWidget(self.list, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        left.addWidget(self.info)

        self.assign_box = QGroupBox("Przypisz do algorytmu")
        form = QFormLayout(self.assign_box)
        self.title = QLineEdit()
        self.title.editingFinished.connect(self._rename)
        self.assign_algo = QComboBox()
        for algo in algorithms.ALGORITHMS.values():
            self.assign_algo.addItem(algo.name, algo.id)
        self.role = QComboBox()
        self.role.addItems(documents.ROLES)
        self.pages_edit = QLineEdit()
        self.pages_edit.setPlaceholderText("np. 3–5")
        self.note = QLineEdit()
        form.addRow("Tytuł:", self.title)
        form.addRow("Algorytm:", self.assign_algo)
        form.addRow("Rola:", self.role)
        form.addRow("Strony:", self.pages_edit)
        form.addRow("Notatka:", self.note)
        row = QHBoxLayout()
        assign = QPushButton("Przypisz")
        assign.clicked.connect(self.assign)
        unassign = QPushButton("Usuń przypisanie")
        unassign.clicked.connect(self.unassign)
        row.addWidget(assign)
        row.addWidget(unassign)
        form.addRow(row)
        self.assigned = QLabel()
        self.assigned.setWordWrap(True)
        form.addRow(self.assigned)
        left.addWidget(self.assign_box)
        box = QWidget()
        box.setLayout(left)
        box.setMaximumWidth(380)
        lay.addWidget(box)

        right = QVBoxLayout()
        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("Szukaj w dokumencie…")
        self.query.returnPressed.connect(self.search)
        find = QPushButton("Szukaj")
        find.clicked.connect(self.search)
        self.prev = QPushButton("◀")
        self.prev.clicked.connect(lambda: self.goto(self.page - 1))
        self.page_spin = QSpinBox()
        self.page_spin.setMinimum(1)
        self.page_spin.valueChanged.connect(lambda v: self.goto(v - 1))
        self.page_count = QLabel("/ –")
        self.next = QPushButton("▶")
        self.next.clicked.connect(lambda: self.goto(self.page + 1))
        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(0.5, 3.0)
        self.zoom.setSingleStep(0.25)
        self.zoom.setValue(1.0)
        self.zoom.setSuffix("×")
        self.zoom.valueChanged.connect(lambda _: self.render())
        for w in (self.query, find, self.prev, self.page_spin, self.page_count, self.next, QLabel("Powiększenie:"),
                  self.zoom):
            row.addWidget(w, 1 if w is self.query else 0)
        right.addLayout(row)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet("color: #8a5a00")
        right.addWidget(self.warning)
        split = QSplitter(Qt.Horizontal)
        self.view_tabs = QTabWidget()
        self.page_label = QLabel()
        self.page_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.page_label)
        self.view_tabs.addTab(area, "Strona")
        self.text = QPlainTextEdit(readOnly=True)
        self.view_tabs.addTab(self.text, "Tekst")
        split.addWidget(self.view_tabs)
        self.hit_list = QListWidget()
        self.hit_list.setWordWrap(True)
        self.hit_list.itemClicked.connect(self._hit)
        split.addWidget(self.hit_list)
        split.setSizes([800, 250])
        right.addWidget(split, 1)
        lay.addLayout(right, 1)
        self.refresh()

    def _enable(self, on):
        for w in (self.remove_button, self.assign_box, self.view_tabs, self.query, self.prev, self.next, self.page_spin):
            w.setEnabled(on)

    def refresh(self, select=None):
        keep = select or (self.doc["id"] if self.doc else None)
        self.list.blockSignals(True)
        self.list.clear()
        row = None
        for i, d in enumerate(documents.load_library()):
            names = ", ".join(algorithms.get(a["algorithm"]).name for a in d["assignments"]
                              if a["algorithm"] in algorithms.ALGORITHMS)
            item = QListWidgetItem(f"{d['title']}\n{d['pages']} str. · {names or 'nieprzypisany'}")
            item.setData(Qt.UserRole, d["id"])
            self.list.addItem(item)
            if d["id"] == keep:
                row = i
        self.list.blockSignals(False)
        if row is not None:
            self.list.setCurrentRow(row)
        elif self.list.count() == 0:
            self._close()
            self.info.setText("Brak dokumentów. Dodaj publikację, na podstawie której powstał algorytm.")

    def _close(self):
        if self.pdf is not None:
            self.pdf.close()
        self.pdf, self.doc, self.hits = None, None, []
        self.page_label.clear()
        self.text.clear()
        self.hit_list.clear()
        self.warning.clear()
        self.assigned.clear()
        self.title.clear()
        self._enable(False)

    def add(self):
        path, _ = QFileDialog.getOpenFileName(self, "Dodaj dokument PDF", "", "PDF (*.pdf)")
        if not path:
            return
        try:
            doc = documents.import_pdf(path)
        except AppError as exc:
            show_error(self, exc)
            return
        self.state.documents_changed.emit()
        self.refresh(select=doc["id"])

    def remove(self):
        if self.doc and QMessageBox.question(self, "Usuń dokument", f"Usunąć „{self.doc['title']}” z listy? "
                                             "Oryginalny plik na dysku zostaje.") == QMessageBox.Yes:
            doc_id = self.doc["id"]
            self._close()
            documents.remove(doc_id)
            self.state.documents_changed.emit()
            self.refresh()

    def open_doc(self, doc_id):
        self.refresh(select=doc_id)

    def _select(self, item, previous=None):
        if item is None:
            return
        self._close()
        try:
            self.doc = documents.get(item.data(Qt.UserRole))
            self.pdf = documents.open_document(self.doc)
        except AppError as exc:
            self.warning.setText(f"{exc.message} {exc.hint}")
            return
        self._enable(True)
        info = self.pdf.info()
        lines = [f"<b>{html.escape(self.doc['file'])}</b>, {info['pages']} str., {info['size_bytes'] // 1024} kB"]
        if info["author"]:
            lines.append("Autor: " + html.escape(info["author"]))
        lines.append(f"Strony z tekstem: {info['pages_with_text']} z {info['pages']}")
        self.info.setText("<br>".join(lines))
        self.title.setText(self.doc["title"])
        if not info["has_text"]:
            self.warning.setText("Ten PDF nie ma warstwy tekstowej (to prawdopodobnie skan). Strony można oglądać, "
                                 "ale wyszukiwanie nie zadziała. Pomoże OCR, np. programem ocrmypdf.")
        elif info["mostly_scanned"]:
            self.warning.setText("Część stron nie ma warstwy tekstowej, wyszukiwanie obejmie tylko pozostałe.")
        self.text.setPlainText(self.pdf.full_text() if info["has_text"] else "(brak tekstu)")
        self.page_spin.blockSignals(True)
        self.page_spin.setMaximum(self.pdf.page_count)
        self.page_spin.blockSignals(False)
        self.page_count.setText(f"/ {self.pdf.page_count}")
        self._show_assigned()
        self.page = -1
        self.goto(0)

    def _rename(self):
        if self.doc and self.title.text().strip() and self.title.text() != self.doc["title"]:
            self.doc = documents.rename(self.doc["id"], self.title.text())
            self.state.documents_changed.emit()
            self.refresh()

    def _show_assigned(self):
        lines = [f"{algorithms.get(a['algorithm']).name} – {a['role']}" + (f", str. {a['pages']}" if a["pages"] else "")
                 for a in self.doc["assignments"] if a["algorithm"] in algorithms.ALGORITHMS]
        self.assigned.setText("Przypisany do: " + "; ".join(lines) if lines else "Nie jest przypisany.")

    def assign(self):
        if self.doc:
            self.doc = documents.assign(self.doc["id"], self.assign_algo.currentData(), self.role.currentText(),
                                        self.pages_edit.text().strip(), self.note.text().strip())
            self._show_assigned()
            self.state.documents_changed.emit()
            self.refresh()

    def unassign(self):
        if self.doc:
            self.doc = documents.unassign(self.doc["id"], self.assign_algo.currentData())
            self._show_assigned()
            self.state.documents_changed.emit()
            self.refresh()

    def goto(self, index):
        if self.pdf is None or not 0 <= index < self.pdf.page_count or index == self.page:
            return
        self.page = index
        self.page_spin.blockSignals(True)
        self.page_spin.setValue(index + 1)
        self.page_spin.blockSignals(False)
        self.prev.setEnabled(index > 0)
        self.next.setEnabled(index < self.pdf.page_count - 1)
        self.render()

    def render(self):
        if self.pdf is None or self.page < 0:
            return
        rects = [r for page, rs, _ in self.hits if page == self.page for r in rs]
        w, h, data = self.pdf.render(self.page, self.zoom.value(), rects)
        self.page_label.setPixmap(QPixmap.fromImage(QImage(data, w, h, 3 * w, QImage.Format_RGB888).copy()))

    def search(self):
        if self.pdf is None:
            return
        self.hits = self.pdf.search(self.query.text())
        self.hit_list.clear()
        for page, rects, snippet in self.hits:
            item = QListWidgetItem(f"str. {page + 1} ({len(rects)}×): {snippet}")
            item.setData(Qt.UserRole, page)
            self.hit_list.addItem(item)
        if self.query.text().strip() and not self.hits:
            self.hit_list.addItem("Nie znaleziono.")
        if self.hits:
            self.view_tabs.setCurrentIndex(0)
            self.page = -1
            self.goto(self.hits[0][0])
        else:
            self.render()

    def _hit(self, item):
        page = item.data(Qt.UserRole)
        if page is not None:
            self.view_tabs.setCurrentIndex(0)
            self.goto(page)
