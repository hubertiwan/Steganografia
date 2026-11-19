"""Running long operations outside the GUI thread, and the state shared by the tabs."""
import threading
import time
import traceback

from PySide6.QtCore import QObject, QThread, Signal

from ..errors import Cancelled


class _Runner(QObject):
    progress = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(object, str)
    cancelled = Signal(object)
    done = Signal()

    def __init__(self, fn, stop_flag):
        super().__init__()
        self.fn = fn
        self.stop_flag = stop_flag

    def run(self):
        try:
            result = self.fn(lambda f, text="": self.progress.emit(float(f), str(text)), self.stop_flag.is_set)
            self.succeeded.emit(result)
        except Cancelled as exc:
            self.cancelled.emit(exc)
        except Exception as exc:  # shown to the user, the app keeps running
            self.failed.emit(exc, traceback.format_exc())
        finally:
            self.done.emit()


class Job(QObject):
    """fn(progress, should_stop) run in a QThread. Results come back as Qt signals."""
    progress = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(object, str)
    cancelled = Signal(object)
    finished = Signal()

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.started = None
        self._stop = threading.Event()
        self._thread = QThread()
        self._runner = _Runner(fn, self._stop)
        self._runner.moveToThread(self._thread)
        self._thread.started.connect(self._runner.run)
        self._runner.progress.connect(self.progress)
        self._runner.succeeded.connect(self.succeeded)
        self._runner.failed.connect(self.failed)
        self._runner.cancelled.connect(self.cancelled)
        self._runner.done.connect(self._thread.quit)
        self._thread.finished.connect(self.finished)

    def start(self):
        self.started = time.perf_counter()
        self._thread.start()

    def cancel(self):
        self._stop.set()

    def elapsed(self):
        return time.perf_counter() - self.started if self.started else 0.0

    def wait(self, ms=30000):
        return self._thread.wait(ms)


class AppState(QObject):
    """The loaded model and the running job. Only one job runs at a time, so the model is
    never used from two threads."""
    model_changed = Signal(object)
    busy_changed = Signal(bool)
    history_changed = Signal()
    documents_changed = Signal()

    def __init__(self):
        super().__init__()
        self.model = None
        self.job = None

    def set_model(self, lm):
        self.model = lm
        self.model_changed.emit(lm)

    @property
    def ready(self):
        return self.model is not None and self.job is None

    def start_job(self, fn):
        if self.job is not None:
            return None
        self.job = Job(fn, self)
        self.job.finished.connect(self._finished)
        self.busy_changed.emit(True)
        self.job.start()
        return self.job

    def _finished(self):
        self.job = None
        self.busy_changed.emit(False)
