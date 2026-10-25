import json

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _isolated_data_dir(monkeypatch, tmp_path):
    """Every test gets its own data folder and never loads the real embedding model."""
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "memory_db", tmp_path / "data" / "memory.db")
    monkeypatch.setattr(settings, "embed", "off")


@pytest.fixture
def bird_dir(tmp_path):
    """A tiny BIRD-shaped dev.json with two databases."""
    d = tmp_path / "bird"
    d.mkdir()
    rows = []
    for i in range(100):
        rows.append({
            "question_id": i,
            "db_id": "shop" if i < 80 else "other",
            "question": f"question {i}",
            "evidence": f"evidence {i}" if i % 2 else "",
            "SQL": f"SELECT {i}",
            "difficulty": "simple",
        })
    (d / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    return d
