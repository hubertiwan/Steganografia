"""Fixtures for the application tests. The algorithm tests (test_unigram.py etc.) do not use them."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # GUI tests without a screen


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Every test gets its own history and document library."""
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path / "app_data"))
    return tmp_path / "app_data"


@pytest.fixture(scope="session")
def fake_lm():
    from fake_lm import make_fake_lm
    return make_fake_lm()


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    pytest.importorskip("transformers")
    from tiny_model import build_tiny_model
    return build_tiny_model(tmp_path_factory.mktemp("tiny_model"))


@pytest.fixture(scope="session")
def text_pdf(tmp_path_factory):
    import pymupdf
    path = tmp_path_factory.mktemp("pdf") / "publikacja.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((72, 90), f"Page {i + 1}", fontsize=14)
        page.insert_textbox(pymupdf.Rect(72, 110, 520, 700), "The detector computes a z-score. " * 8, fontsize=11)
    doc.set_metadata({"title": "Test paper", "author": "Test Author"})
    doc.save(str(path))
    return path


@pytest.fixture(scope="session")
def scanned_pdf(tmp_path_factory):
    """A PDF with only an image on the page, like a scan."""
    import pymupdf
    path = tmp_path_factory.mktemp("pdf") / "scan.pdf"
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 100), False)
    pix.clear_with(200)
    doc = pymupdf.open()
    doc.new_page().insert_image(pymupdf.Rect(72, 72, 472, 272), pixmap=pix)
    doc.save(str(path))
    return path
