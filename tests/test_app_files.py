"""PDF documents and export of results."""
import csv
import json

import pymupdf
import pytest

from app import algorithms, documents, experiments, history, reports
from app.documents import PdfDocument
from app.errors import AppError


def test_pdf_reading(text_pdf):
    with PdfDocument(text_pdf) as pdf:
        info = pdf.info()
        assert (info["pages"], info["title"], info["author"]) == (3, "Test paper", "Test Author")
        assert info["has_text"] and not info["mostly_scanned"]
        hits = pdf.search("z-score")
        assert [page for page, _, _ in hits] == [0, 1, 2] and "z-score" in hits[0][2]
        assert pdf.search("") == [] and pdf.search("not in the text") == []
        w, h, plain = pdf.render(0)
        assert len(plain) == w * h * 3
        assert pdf.render(0, highlights=hits[0][1])[2] != plain
        with pytest.raises(AppError):
            pdf.render(5)


def test_scanned_pdf(scanned_pdf):
    with PdfDocument(scanned_pdf) as pdf:
        assert not pdf.info()["has_text"]


@pytest.mark.parametrize("content", [b"", b"not a pdf", b"%PDF-1.4\n%broken"])
def test_broken_pdf(tmp_path, content):
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)
    with pytest.raises(AppError):
        PdfDocument(path)


def test_encrypted_pdf(tmp_path):
    path = tmp_path / "password.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    doc.save(str(path), encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="a", owner_pw="a")
    with pytest.raises(AppError) as e:
        PdfDocument(path)
    assert "hasłem" in e.value.message


def test_document_library(data_dir, text_pdf, scanned_pdf, tmp_path):
    doc = documents.import_pdf(text_pdf)
    with pytest.raises(AppError):
        documents.import_pdf(text_pdf)  # same file twice
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(AppError):
        documents.import_pdf(tmp_path / "notes.txt")
    assert not documents.import_pdf(scanned_pdf)["has_text"]
    documents.assign(doc["id"], "mpac", documents.ROLES[0], "4-5")
    documents.assign(doc["id"], "kgw", documents.ROLES[2])
    documents.assign(doc["id"], "mpac", documents.ROLES[1])  # replaces the first assignment
    assert [a["role"] for _, a in documents.for_algorithm("mpac")] == [documents.ROLES[1]]
    documents.unassign(doc["id"], "kgw")
    assert documents.for_algorithm("kgw") == []
    assert documents.rename(doc["id"], "New title")["title"] == "New title"
    with documents.open_document(documents.get(doc["id"])) as pdf:
        assert pdf.page_count == 3
    documents.remove(doc["id"])
    assert text_pdf.exists() and len(documents.load_library()) == 1  # only the copy is removed


def test_broken_library_file(data_dir):
    (data_dir / "documents").mkdir(parents=True)
    (data_dir / "documents" / "library.json").write_text("{broken", encoding="utf-8")
    assert documents.load_library() == []
    assert (data_dir / "documents" / "library.broken.json").exists()


@pytest.fixture(scope="module")
def results(fake_lm):
    s = {"prompts": experiments.PROMPTS[:3], "max_new_tokens": 40}
    d = algorithms.get("mpac").defaults()
    return {"test": experiments.run_test(fake_lm, "mpac", d, s),
            "comparison": experiments.run_comparison(fake_lm, [("kgw", algorithms.get("kgw").defaults()), ("mpac", d)], s),
            "demo": experiments.run_demo(fake_lm, "dipmark", algorithms.get("dipmark").defaults(), "Q?", 40, 1),
            "detect": experiments.detect_text(fake_lm, "synthid", algorithms.get("synthid").defaults(), "w1 w2 w3 w4 w5 w6")}


@pytest.mark.parametrize("kind", ["test", "comparison", "demo", "detect"])
def test_export(kind, results, tmp_path, data_dir):
    rec = history.new_record(kind, f"Raport {kind}", ["mpac"], files=["prompts.txt"])
    rec.update(status="ok", duration_s=1.0, result=results[kind])
    history.save(rec)
    data = json.loads(reports.export_json(rec, tmp_path / "r.json").read_text(encoding="utf-8"))
    assert data["experiment"]["result"]["kind"] == kind
    rows = list(csv.DictReader(open(reports.export_csv(rec, tmp_path / "r.csv"), encoding="utf-8-sig")))
    assert len(rows) == {"test": 6, "comparison": 12, "demo": 3, "detect": 1}[kind]
    with pymupdf.open(str(reports.export_pdf(rec, tmp_path / "r.pdf"))) as pdf:
        text = "".join(page.get_text() for page in pdf)
    assert "Parametry" in text
    if kind == "test":
        assert "Dokładność bitowa" in text and "Podsumowanie" in text


def test_export_needs_results(tmp_path):
    rec = history.new_record("test", "x", ["kgw"])
    for fn in (reports.export_pdf, reports.export_csv):
        with pytest.raises(AppError):
            fn(rec, tmp_path / "x")
