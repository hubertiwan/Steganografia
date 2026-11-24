"""Export of results to JSON, CSV and PDF."""
import csv
import json

import pymupdf  # only to read the text of the exported PDF report
import pytest

from app import algorithms, experiments, history, reports
from app.errors import AppError


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
