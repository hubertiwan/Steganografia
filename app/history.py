"""Experiment history: one JSON file per experiment in app_data/experiments/.
Cancelled and failed runs are saved too, with their status and error message."""
import json
import math
import uuid
from datetime import datetime

from .paths import experiments_dir

STATUS = {"ok": "zakończony", "error": "błąd", "cancelled": "przerwany", "running": "w toku"}
KINDS = {"demo": "Demonstracja", "detect": "Sprawdzenie tekstu", "test": "Test", "comparison": "Porównanie"}


def clean(obj):
    """JSON has no NaN/inf and no numpy types."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def new_record(kind, title, algorithms, params=None, files=None, model=None):
    now = datetime.now()
    return {"id": now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
            "created": now.isoformat(timespec="seconds"), "kind": kind, "title": title,
            "algorithms": list(algorithms), "params": params or {}, "files": files or [],
            "model": model or {}, "status": "running", "duration_s": None, "result": None, "error": None}


def save(record):
    path = experiments_dir() / f"{record['id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(record), ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # a crash while writing does not leave a broken file
    return path


def load(record_id):
    return json.loads((experiments_dir() / f"{record_id}.json").read_text(encoding="utf-8"))


def list_records():
    """Newest first, without the (large) results. Broken files are listed as errors."""
    out = []
    for path in sorted(experiments_dir().glob("*.json"), reverse=True):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec.pop("result", None)
        except (OSError, json.JSONDecodeError):
            rec = {"id": path.stem, "created": "", "kind": "?", "title": f"Uszkodzony plik {path.name}",
                   "algorithms": [], "status": "error", "broken": True}
        out.append(rec)
    return out


def delete(record_id):
    path = experiments_dir() / f"{record_id}.json"
    if path.exists():
        path.unlink()
