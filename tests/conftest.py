"""Fixtures for the application tests. The algorithm tests (test_unigram.py etc.) do not use them."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # GUI tests without a screen


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Every test gets its own history."""
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
