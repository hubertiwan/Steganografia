import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"


def data_dir():
    """History and imported PDFs. Kept next to the repository (app_data/, ignored by git).
    The APP_DATA_DIR environment variable can point somewhere else (used by the tests)."""
    path = Path(os.environ.get("APP_DATA_DIR", REPO_ROOT / "app_data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def experiments_dir():
    path = data_dir() / "experiments"
    path.mkdir(exist_ok=True)
    return path


def documents_dir():
    path = data_dir() / "documents"
    path.mkdir(exist_ok=True)
    return path
