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


@pytest.fixture(params=["config_dir_unset", "config_dir_empty"])
def no_credentials(request, tmp_path, monkeypatch):
    """Real SDK client, no key and no profile anywhere, and nowhere for a request to go."""
    from app.config import settings

    empty = tmp_path / "no-profile"
    empty.mkdir()
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    for name in ("APPDATA", "HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(empty))
    # the SDK raises a plain TypeError when no config dir is set and CredentialsError when one is, so cover both
    if request.param == "config_dir_empty":
        monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(empty))
    else:
        monkeypatch.delenv("ANTHROPIC_CONFIG_DIR", raising=False)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(settings, "anthropic_api_key", None)
