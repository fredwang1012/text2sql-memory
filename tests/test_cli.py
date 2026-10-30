import json
import sqlite3

import numpy as np

from app import cli, seed
from app.config import settings
from app.llm import Draft, LLMTransient
from app.memory import Store

YAML = """db_id: shop
items:
  - id: s-1
    kind: metric
    title: Revenue
    terms: [revenue]
    body: sum of amount
    provenance: schema
"""


class FakeEmbedder:
    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_passages(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def _bird_data():
    bird_dir = settings.data_dir / "bird"
    db_dir = bird_dir / "dev_databases" / "shop"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "shop.sqlite")
    conn.execute("CREATE TABLE sale (id INTEGER PRIMARY KEY, amount REAL)")
    conn.commit()
    conn.close()
    (bird_dir / "dev.json").write_text(json.dumps([]), encoding="utf-8")


def test_ask_without_bird_data_prints_a_hint(capsys):
    assert cli.main(["--db-id", "shop", "ask", "how many?"]) == 1
    assert "BIRD data not found" in capsys.readouterr().err


def test_ask_in_memory_mode_on_an_empty_store_hints_to_seed(capsys, monkeypatch):
    _bird_data()

    class Fake:
        def generate(self, *a):
            return Draft(sql="SELECT COUNT(*) FROM sale", items_used=[], assumptions="")

    monkeypatch.setattr("app.pipeline.make_llm", Fake)
    assert cli.main(["--db-id", "shop", "ask", "how many sales?"]) == 0
    assert "cli seed" in capsys.readouterr().err
    assert cli.main(["--db-id", "shop", "ask", "--mode", "schema", "how many sales?"]) == 0
    assert "cli seed" not in capsys.readouterr().err


def test_ask_reports_a_connection_failure_briefly(capsys, monkeypatch):
    _bird_data()

    class Down:
        def generate(self, *a):
            raise LLMTransient("Anthropic API not reachable: APITimeoutError")

    monkeypatch.setattr("app.pipeline.make_llm", Down)
    assert cli.main(["--db-id", "shop", "ask", "--mode", "schema", "how many?"]) == 1
    assert "not reachable" in capsys.readouterr().err


def test_cli_seed_writes_vectors(tmp_path, monkeypatch):
    path = tmp_path / "shop.yaml"
    path.write_text(YAML, encoding="utf-8")
    monkeypatch.setattr(cli, "get_embedder", lambda: FakeEmbedder())
    assert cli.main(["--db-id", "shop", "seed", "--file", str(path)]) == 0
    with Store(settings.memory_db) as s:
        assert s.vector("s-1") is not None


def test_seed_main_writes_vectors(tmp_path, monkeypatch):
    path = tmp_path / "shop.yaml"
    path.write_text(YAML, encoding="utf-8")
    monkeypatch.setattr(seed, "get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("sys.argv", ["seed", "import", "--db-id", "shop", "--file", str(path)])
    seed.main()
    with Store(settings.memory_db) as s:
        assert s.vector("s-1") is not None


def test_ask_without_credentials_prints_the_friendly_message(capsys, no_credentials):
    _bird_data()
    assert cli.main(["--db-id", "shop", "ask", "--mode", "schema", "how many?"]) == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "ant auth login" in err
    assert "Traceback" not in err
