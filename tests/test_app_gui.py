"""GUI tests (pytest-qt, offscreen). Experiments run in a worker thread and the event loop of
the GUI keeps running meanwhile; errors end with a message, not with a crash."""
import time

import pytest

pytest.importorskip("pytestqt")
from PySide6.QtCore import QTimer  # noqa: E402

from app import history  # noqa: E402
from app.gui.main_window import MainWindow  # noqa: E402
from app.gui.worker import AppState  # noqa: E402


@pytest.fixture
def errors(monkeypatch):
    """Collect error dialogs instead of showing them."""
    seen = []
    import app.gui.main_window as mw
    import app.gui.tab_experiments as te
    import app.gui.tab_info as ti
    import app.gui.widgets as w
    for module in (w, te, ti, mw):
        monkeypatch.setattr(module, "show_error", lambda parent, exc, tb=None: seen.append(exc))
    return seen


@pytest.fixture
def window(qtbot, data_dir, fake_lm):
    state = AppState()
    state.set_model(fake_lm)
    win = MainWindow(state)
    qtbot.addWidget(win)
    win.show()
    return win


def wait(qtbot, win, timeout=120_000):
    qtbot.waitUntil(lambda: win.state.job is None, timeout=timeout)


def test_tabs_and_buttons_without_model(qtbot, data_dir):
    win = MainWindow(AppState())
    qtbot.addWidget(win)
    for key in win.pages:
        win.show_tab(key)
    assert not win.pages["demo"].go.isEnabled()  # nothing to run without a model
    assert "nie wczytano" in win.model_label.text()


def test_demo(window, qtbot, errors):
    demo = window.pages["demo"]
    demo.select("mpac")
    demo.tokens.setValue(60)
    demo.start()
    assert window.state.job is not None and not demo.go.isEnabled()
    wait(qtbot, window)
    assert not errors and not demo.result_box.isHidden() and demo.go.isEnabled()
    assert demo.view.table.item(0, 4).text() == "wykryty"
    assert history.list_records()[0]["status"] == "ok"
    demo.edit_percent.setValue(30)
    demo.random_edit()
    demo.check_edited()
    wait(qtbot, window)
    assert "Po zmianie" in demo.edit_result.text()


def test_gui_is_not_blocked_during_a_test(window, qtbot, errors):
    tab = window.pages["test"]
    tab.select("synthid")
    tab.settings.count.setValue(10)
    tab.settings.tokens.setValue(150)
    ticks = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    timer.start(20)
    tab.start()
    wait(qtbot, window)
    timer.stop()
    assert not errors and not tab.result_box.isHidden()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) > 10 and max(gaps) < 1.0, f"GUI blocked for {max(gaps):.2f} s"


def test_cancel(window, qtbot, errors):
    tab = window.pages["test"]
    tab.select("synthid")
    tab.settings.count.setValue(20)
    tab.settings.tokens.setValue(200)
    tab.start()
    qtbot.waitUntil(lambda: tab.progress.bar.value() > 0, timeout=30_000)
    tab.progress.stop.click()
    wait(qtbot, window)
    assert history.list_records()[0]["status"] == "cancelled"
    assert "Przerwano" in tab.progress.label.text()


def test_bad_input_shows_message(window, errors, tmp_path):
    tab = window.pages["test"]
    tab.settings.source.setCurrentIndex(1)
    tab.settings.file.setText(str(tmp_path / "missing.txt"))
    tab.start()
    assert errors and window.state.job is None
    demo = window.pages["demo"]
    demo.select("mpac")
    demo.params.forms["mpac"].inputs["message"].setText("12")
    demo.start()
    assert "Wiadomość" in errors[-1].message


def test_comparison_and_history(window, qtbot, errors):
    tab = window.pages["compare"]
    for algo_id, box in tab.include.items():
        box.setChecked(algo_id in ("kgw", "mpac"))
    tab.settings.count.setValue(3)
    tab.settings.tokens.setValue(40)
    tab.start()
    wait(qtbot, window)
    assert not errors and not tab.result_box.isHidden()
    hist = window.pages["history"]
    hist.table.selectRow(0)
    assert hist.current["kind"] == "comparison" and hist.holder.count() == 1


def test_export(window, qtbot, errors, tmp_path, monkeypatch):
    demo = window.pages["demo"]
    demo.tokens.setValue(30)
    demo.start()
    wait(qtbot, window)
    target = tmp_path / "report.pdf"
    import app.gui.widgets as w
    monkeypatch.setattr(w.QFileDialog, "getSaveFileName", lambda *a, **k: (str(target), ""))
    demo.export.export("pdf")
    wait(qtbot, window)
    assert not errors and target.stat().st_size > 5000


def test_documents_tab(window, errors, text_pdf, scanned_pdf):
    from app import documents
    tab = window.pages["documents"]
    doc = documents.import_pdf(text_pdf)
    tab.refresh(select=doc["id"])
    assert tab.page_label.pixmap() is not None
    tab.query.setText("z-score")
    tab.search()
    assert len(tab.hits) == 3
    tab.assign_algo.setCurrentIndex(tab.assign_algo.findData("mpac"))
    tab.assign()
    assert documents.for_algorithm("mpac")[0][0]["id"] == doc["id"]
    scan = documents.import_pdf(scanned_pdf)
    tab.refresh(select=scan["id"])
    assert "warstwy tekstowej" in tab.warning.text()
    assert not errors
