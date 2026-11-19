"""Source documents (PDF): reading with PyMuPDF and a small library with assignments to
algorithms. An imported PDF is copied to app_data/documents/, so the app still works after
the original file is moved. The content is only reference material: nothing is added to it
and assigning it to an algorithm is the user's statement, not a check."""
import hashlib
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from .errors import AppError
from .paths import documents_dir

ROLES = ["publikacja źródłowa", "materiały z zajęć", "materiał dodatkowy"]
MIN_CHARS = 25  # a page with less text than this counts as a page without a text layer


def _pymupdf():
    try:
        import pymupdf
        return pymupdf
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF
            return pymupdf
        except ImportError as exc:
            raise AppError("Brak biblioteki PyMuPDF.", "Zainstaluj zależności: pip install -r requirements.txt") from exc


class PdfDocument:
    def __init__(self, path):
        pymupdf = _pymupdf()
        self.path = Path(path)
        if not self.path.is_file():
            raise AppError(f"Nie ma takiego pliku: {self.path}")
        if self.path.stat().st_size == 0:
            raise AppError(f"Plik {self.path.name} jest pusty.")
        try:
            self.doc = pymupdf.open(str(self.path), filetype="pdf")
        except Exception as exc:
            raise AppError(f"Nie udało się otworzyć {self.path.name} jako PDF.",
                           "Plik jest uszkodzony albo to nie jest PDF.") from exc
        if self.doc.needs_pass:
            self.doc.close()
            raise AppError(f"{self.path.name} jest zabezpieczony hasłem.", "Zapisz kopię bez hasła i zaimportuj ją.")
        if self.doc.page_count == 0:
            self.doc.close()
            raise AppError(f"{self.path.name} nie ma żadnej strony.")
        self._texts = None

    def close(self):
        self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def page_count(self):
        return self.doc.page_count

    def page_text(self, index):
        if self._texts is None:
            self._texts = []
            for page in self.doc:
                try:
                    self._texts.append(page.get_text("text"))
                except Exception:  # one broken page should not block the whole document
                    self._texts.append("")
        return self._texts[index]

    def full_text(self):
        return "\n".join(f"--- strona {i + 1} ---\n{self.page_text(i)}" for i in range(self.page_count))

    def info(self):
        meta = {k: v for k, v in (self.doc.metadata or {}).items() if v}
        chars = [len(self.page_text(i).strip()) for i in range(self.page_count)]
        with_text = sum(c >= MIN_CHARS for c in chars)
        return {"file": self.path.name, "size_bytes": self.path.stat().st_size, "pages": self.page_count,
                "title": meta.get("title", ""), "author": meta.get("author", ""),
                "created": meta.get("creationDate", ""), "chars": sum(chars),
                "pages_with_text": with_text, "has_text": with_text > 0,
                "mostly_scanned": with_text < 0.5 * self.page_count}

    def render(self, index, zoom=1.0, highlights=()):
        """(width, height, RGB bytes) of a page; search hits drawn as yellow boxes on a copy."""
        pymupdf = _pymupdf()
        if not 0 <= index < self.page_count:
            raise AppError(f"Strona {index + 1} nie istnieje.")
        pix = self.doc[index].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        if not highlights:
            return pix.width, pix.height, bytes(pix.samples)
        import numpy as np
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        for x0, y0, x1, y1 in highlights:
            box = img[max(int(y0 * zoom), 0):int(y1 * zoom), max(int(x0 * zoom), 0):int(x1 * zoom)]
            box[:] = (box * 0.55 + np.array([255, 214, 0]) * 0.45).astype(np.uint8)
        return pix.width, pix.height, img.tobytes()

    def search(self, query):
        """List of (page, rectangles, snippet)."""
        query = (query or "").strip()
        hits = []
        if not query:
            return hits
        for i, page in enumerate(self.doc):
            rects = page.search_for(query)
            if not rects:
                continue
            text = self.page_text(i)
            pos = text.lower().find(query.lower())
            snippet = text[max(0, pos - 50):pos + len(query) + 50].replace("\n", " ").strip() if pos >= 0 else ""
            hits.append((i, [tuple(r) for r in rects], snippet))
        return hits


# ---------------------------------------------------------------- library
def _library_file():
    return documents_dir() / "library.json"


def load_library():
    path = _library_file()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))["documents"]
    except (json.JSONDecodeError, KeyError, TypeError):
        shutil.copy2(path, path.with_name("library.broken.json"))
        return []


def _save_library(docs):
    path = _library_file()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"documents": docs}, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def import_pdf(source):
    src = Path(source)
    if src.suffix.lower() != ".pdf":
        raise AppError(f"{src.name} nie jest plikiem PDF.", "Wybierz plik z rozszerzeniem .pdf.")
    with PdfDocument(src) as pdf:  # raises AppError for broken / encrypted files
        info = pdf.info()
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    docs = load_library()
    for d in docs:
        if d["sha256"] == digest:
            raise AppError(f"Ten dokument już jest w bibliotece („{d['title']}”).")
    doc_id = uuid.uuid4().hex[:10]
    shutil.copy2(src, documents_dir() / f"{doc_id}.pdf")
    doc = {"id": doc_id, "title": info["title"] or src.stem, "file": src.name, "sha256": digest,
           "added": datetime.now().isoformat(timespec="seconds"), "pages": info["pages"],
           "has_text": info["has_text"], "assignments": []}
    docs.append(doc)
    _save_library(docs)
    return doc


def get(doc_id):
    for d in load_library():
        if d["id"] == doc_id:
            return d
    raise AppError("Nie ma takiego dokumentu w bibliotece.")


def _change(doc_id, fn):
    docs = load_library()
    for d in docs:
        if d["id"] == doc_id:
            fn(d)
            _save_library(docs)
            return d
    raise AppError("Nie ma takiego dokumentu w bibliotece.")


def rename(doc_id, title):
    def fn(d):
        d["title"] = title.strip() or d["title"]
    return _change(doc_id, fn)


def assign(doc_id, algorithm, role=ROLES[0], pages="", note=""):
    def fn(d):
        d["assignments"] = [a for a in d["assignments"] if a["algorithm"] != algorithm]
        d["assignments"].append({"algorithm": algorithm, "role": role, "pages": pages, "note": note})
    return _change(doc_id, fn)


def unassign(doc_id, algorithm):
    def fn(d):
        d["assignments"] = [a for a in d["assignments"] if a["algorithm"] != algorithm]
    return _change(doc_id, fn)


def remove(doc_id):
    """Removes the copy in the library, never the original file."""
    _save_library([d for d in load_library() if d["id"] != doc_id])
    copy = documents_dir() / f"{doc_id}.pdf"
    if copy.exists():
        copy.unlink()


def for_algorithm(algorithm):
    return [(d, a) for d in load_library() for a in d["assignments"] if a["algorithm"] == algorithm]


def open_document(doc):
    path = documents_dir() / f"{doc['id']}.pdf"
    if not path.exists():
        raise AppError("Brakuje pliku tego dokumentu w bibliotece.", "Usuń wpis i zaimportuj PDF jeszcze raz.")
    return PdfDocument(path)
