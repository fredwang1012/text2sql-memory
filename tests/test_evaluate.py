import hashlib
import json
import sqlite3

import numpy as np
import pytest

from app import bird, evaluate, pipeline
from app.config import settings
from app.llm import Draft, LLMTransient, LLMUnavailable
from app.memory import Item, Store

GOLD = {"total revenue": "SELECT SUM(amount) FROM sale", "number of sales": "SELECT COUNT(*) FROM sale"}
WRONG = "SELECT 0"


@pytest.fixture
def world(tmp_path, monkeypatch):
    """data/bird with a tiny dev.json (30 questions on 'shop'), its sqlite file and a store holding one rule."""
    bird_dir = settings.data_dir / "bird"
    db_dir = bird_dir / "dev_databases" / "shop"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "shop.sqlite")
    conn.executescript("CREATE TABLE sale (id INTEGER PRIMARY KEY, amount REAL); INSERT INTO sale VALUES (1,2.5),(2,4);")
    conn.commit()
    conn.close()
    (db_dir / "database_description").mkdir()
    (db_dir / "database_description" / "sale.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "id,,sale id,integer,\namount,,amount of the sale,real,\n".encode("utf-8-sig"))
    rows = []
    for i in range(30):
        kind = "total revenue" if i % 2 else "number of sales"
        rows.append({"question_id": i, "db_id": "shop", "question": f"What is the {kind} ({i})?",
                     "evidence": f"SECRET-EVIDENCE-{i}", "SQL": GOLD[kind],
                     "difficulty": "simple" if i % 3 else "challenging"})
    (bird_dir / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    with Store(settings.memory_db) as s:
        s.add(Item(id="rev", db_id="shop", kind="metric", title="Revenue", terms=["revenue"],
                   body="total revenue is the sum of sale.amount", provenance="schema"))
    yaml_path = tmp_path / "shop.yaml"  # the eval seeds its temp store from this, never from memory.db
    yaml_path.write_text("""db_id: shop
items:
  - id: rev
    kind: metric
    title: Revenue
    terms: [revenue]
    body: total revenue is the sum of sale.amount
    provenance: schema
""", encoding="utf-8")
    monkeypatch.setattr(evaluate, "knowledge_path", lambda db_id: yaml_path)
    return yaml_path


class GatedLLM:
    """Gold SQL for revenue questions only when the revenue rule (or the oracle evidence) is present; count questions need none."""

    def __init__(self):
        self.seen = []

    def generate(self, question, schema_text, items):
        self.seen.append((question, [i.id for i in items], [i.body for i in items]))
        if "total revenue" in question:
            sql = GOLD["total revenue"] if any(i.id in ("rev", "oracle") for i in items) else WRONG
        else:
            sql = GOLD["number of sales"]
        return Draft(sql=sql, items_used=[i.id for i in items], assumptions="")

    def repair(self, question, schema_text, items, sql, error):
        return self.generate(question, schema_text, items)


def test_memory_beats_schema_and_oracle_is_upper_bound(world, tmp_path, capsys):
    out = tmp_path / "out" / "r.json"
    assert evaluate.main(["--db", "shop", "--oracle", "--yes", "--json", str(out)], llm=GatedLLM()) == 0
    data = json.loads(out.read_text(encoding="utf-8"))["results"]
    ex = {m: r["summary"]["ex"] for m, r in data.items()}
    assert ex["memory"] > ex["schema"]
    assert ex["oracle"] == 1.0
    assert data["schema"]["summary"]["mean_rules_used"] == 0
    assert data["memory"]["summary"]["mean_rules_used"] > 0
    assert set(data["memory"]["summary"]["by_difficulty"]) == {"simple", "challenging"}
    shown = capsys.readouterr().out
    assert "SECRET-EVIDENCE" not in shown and "upper-bound" in shown
    assert "SECRET-EVIDENCE" not in out.read_text(encoding="utf-8")


def test_evidence_reaches_only_the_oracle(world):
    llm = GatedLLM()
    qs = evaluate.eval_questions("shop")
    evaluate.run_mode("schema", qs, "shop", llm)
    with Store(settings.memory_db) as s:
        evaluate.run_mode("memory", qs, "shop", llm, s)
    assert not any("SECRET" in b for _, _, bodies in llm.seen for b in bodies)
    llm.seen.clear()
    evaluate.run_mode("oracle", qs, "shop", llm)
    assert all(ids == ["oracle"] and "SECRET" in bodies[0] for _, ids, bodies in llm.seen)
    with Store(settings.memory_db) as s:
        assert [i.id for i in s.list("shop")] == ["rev"]  # oracle added nothing


def test_only_eval_split_questions_run(world):
    seed_ids = {q.question_id for q in bird.split(bird.load_questions("shop"))[0]}
    assert seed_ids and not seed_ids & {q.question_id for q in evaluate.eval_questions("shop")}
    assert len(evaluate.eval_questions("shop", limit=3)) == 3


def test_prompt_declined_without_yes(world, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    llm = GatedLLM()
    assert evaluate.main(["--db", "shop", "--limit", "2"], llm=llm) == 1
    out = capsys.readouterr().out
    assert "2 questions x 3 modes" in out and "6 to 12 LLM calls" in out and settings.llm_model in out
    assert llm.seen == []


def test_eval_always_seeds_a_temp_store_from_the_yaml(world, capsys):
    """Even with a populated memory.db, the run seeds from the YAML and leaves the real store alone."""
    before = settings.memory_db.read_bytes()
    assert evaluate.main(["--db", "shop", "--yes"], llm=GatedLLM()) == 0
    assert settings.memory_db.read_bytes() == before
    with Store(settings.memory_db) as s:
        assert s.get("rev").uses == 0
    assert "seeded from" in capsys.readouterr().out


def test_memory_db_content_is_not_used(world, capsys):
    """A rule that exists only in memory.db must not reach the run: the YAML is the source."""
    with Store(settings.memory_db) as s:
        s.add(Item(id="stray", db_id="shop", kind="metric", title="Stray", terms=["revenue"], body="x",
                   provenance="schema"))
    llm = GatedLLM()
    assert evaluate.main(["--db", "shop", "--modes", "memory", "--yes"], llm=llm) == 0
    assert not any("stray" in ids for _, ids, _ in llm.seen)


class FakeEmbedder:
    def embed_query(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_passages(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_json_records_how_the_run_was_made(world, tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    assert evaluate.main(["--db", "shop", "--yes", "--limit", "3", "--json", str(out)], llm=GatedLLM()) == 0
    run_meta = json.loads(out.read_text(encoding="utf-8"))["run"]
    assert run_meta["retrieval_mode"] == "bm25" and run_meta["k"] == 8
    assert run_meta["knowledge_sha256"] == hashlib.sha256(world.read_bytes()).hexdigest()
    assert "git_commit" in run_meta
    monkeypatch.setattr(evaluate, "get_embedder", lambda: FakeEmbedder())
    assert evaluate.main(["--db", "shop", "--yes", "--limit", "3", "--json", str(out)], llm=GatedLLM()) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["run"]["retrieval_mode"] == "hybrid"


class FlakyLLM(GatedLLM):
    fail_on = ""

    def generate(self, question, schema_text, items):
        if self.fail_on in question:
            raise RuntimeError("overloaded")
        return super().generate(question, schema_text, items)


def test_one_failing_question_does_not_lose_the_run(world):
    qs = evaluate.eval_questions("shop")
    llm = FlakyLLM()
    llm.fail_on = f"({qs[2].question_id})"
    out = evaluate.run_mode("schema", qs, "shop", llm)
    bad = [r for r in out["questions"] if r["error"]]
    assert len(out["questions"]) == len(qs)
    assert [r["question_id"] for r in bad] == [qs[2].question_id]
    assert bad[0]["error"] == "RuntimeError: overloaded"
    assert out["summary"]["correct"] == len(qs) - 1 - sum("total revenue" in q.question for q in qs if q is not qs[2])


class DownLLM(GatedLLM):
    def generate(self, question, schema_text, items):
        if items:
            raise LLMUnavailable("no credentials")
        return super().generate(question, schema_text, items)


def test_unavailable_is_fatal_but_finished_modes_are_saved(world, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert evaluate.main(["--db", "shop", "--yes", "--json", str(out)], llm=DownLLM()) == 1
    assert list(json.loads(out.read_text(encoding="utf-8"))["results"]) == ["schema"]
    assert "not available" in capsys.readouterr().err


def test_json_metadata_and_gold_failures(world, tmp_path):
    bird_dir = settings.data_dir / "bird"
    rows = json.loads((bird_dir / "dev.json").read_text(encoding="utf-8"))
    rows[0]["SQL"] = "SELECT * FROM missing_table"
    (bird_dir / "dev.json").write_text(json.dumps(rows), encoding="utf-8")
    out = tmp_path / "r.json"
    assert evaluate.main(["--db", "shop", "--oracle", "--yes", "--json", str(out)], llm=GatedLLM()) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    n = len(evaluate.eval_questions("shop"))
    assert data["oracle"] is True and data["questions_per_mode"] == n and len(data["eval_question_ids"]) == n
    assert data["model"] and data["effort"] and data["timestamp"] and data["split_file"].endswith("split-shop.json")
    gold_bad = 1 if 0 in data["eval_question_ids"] else 0
    assert data["results"]["schema"]["summary"]["n_gold_error"] == gold_bad


def test_limit_must_be_positive(world, capsys):
    with pytest.raises(SystemExit):
        evaluate.main(["--db", "shop", "--limit", "0", "--yes"], llm=GatedLLM())
    assert "at least 1" in capsys.readouterr().err


def test_missing_bird_data_gets_a_hint_not_a_traceback(tmp_path, capsys):
    assert evaluate.main(["--db", "shop", "--yes"], llm=GatedLLM()) == 1
    assert "BIRD data not found" in capsys.readouterr().err


def test_oracle_row_reports_rules_as_na(world, capsys):
    assert evaluate.main(["--db", "shop", "--oracle", "--yes", "--limit", "4"], llm=GatedLLM()) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("oracle*"))
    assert line.split()[4:6] == ["n/a", "n/a"]  # columns: mode n EX errors rules/q coverage warned


class TimeoutLLM(GatedLLM):
    """Times out on chosen questions."""

    def __init__(self, failing=()):
        super().__init__()
        self.failing = failing

    def generate(self, question, schema_text, items):
        if any(f"({i})" in question for i in self.failing):
            raise LLMTransient("APITimeoutError")
        return super().generate(question, schema_text, items)


def test_one_timeout_does_not_abort(world):
    qs = evaluate.eval_questions("shop")
    llm = TimeoutLLM(failing=[qs[1].question_id])
    out = evaluate.run_mode("schema", qs, "shop", llm)
    assert len(out["questions"]) == len(qs)
    assert [r["error"] for r in out["questions"] if r["error"]] == ["LLMTransient: APITimeoutError"]


def test_five_failures_in_a_row_abort_with_a_partial_report(world, tmp_path, capsys):
    qs = evaluate.eval_questions("shop")
    llm = TimeoutLLM(failing=[q.question_id for q in qs[2:]])
    out = tmp_path / "r.json"
    assert evaluate.main(["--db", "shop", "--yes", "--json", str(out)], llm=llm) == 1
    captured = capsys.readouterr()
    assert "Aborted" in captured.err and "schema" in captured.out  # the partial report is still printed
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["aborted"]
    assert len(data["results"]["schema"]["questions"]) == 2 + evaluate.MAX_FAILURES
    assert "memory" not in data["results"]


def test_a_success_resets_the_failure_streak(world):
    qs = evaluate.eval_questions("shop")
    failing = [q.question_id for q in qs[:4]] + [q.question_id for q in qs[5:9]]
    out = evaluate.run_mode("schema", qs, "shop", TimeoutLLM(failing=failing))
    assert len(out["questions"]) == len(qs)


def test_json_is_written_after_every_question(world, tmp_path, monkeypatch):
    out = tmp_path / "r.json"
    sizes = []
    real = evaluate._write_json

    def spy(*a, **k):
        real(*a, **k)
        sizes.append(len(json.loads(out.read_text(encoding="utf-8"))["results"]["schema"]["questions"]))

    monkeypatch.setattr(evaluate, "_write_json", spy)
    assert evaluate.main(["--db", "shop", "--modes", "schema", "--yes", "--limit", "3", "--json", str(out)],
                         llm=GatedLLM()) == 0
    assert sizes[:3] == [1, 2, 3]


class RowsLLM(GatedLLM):
    def __init__(self, sql):
        super().__init__()
        self.sql = sql

    def generate(self, question, schema_text, items):
        return Draft(sql=self.sql, items_used=[], assumptions="")


def _with_gold(sql, n=3):
    qs = evaluate.eval_questions("shop")[:n]
    return [bird.Question(q.question_id, q.db_id, q.question, q.evidence, sql, q.difficulty) for q in qs]


def test_row_cap_does_not_distort_scoring(world, monkeypatch):
    """The pipeline caps rows at 1 here; scoring must still compare the full results."""
    real_run = pipeline.run
    monkeypatch.setattr(pipeline, "run", lambda db, sql, **k: real_run(db, sql, max_rows=1))
    sql = "SELECT id, amount FROM sale"
    out = evaluate.run_mode("schema", _with_gold(sql), "shop", RowsLLM(sql))
    assert out["summary"]["correct"] == 3 and out["summary"]["n_truncated"] == 0


def test_truncated_scoring_is_wrong_and_counted(world, monkeypatch):
    monkeypatch.setattr(evaluate, "MAX_ROWS", 1)
    sql = "SELECT id FROM sale"
    out = evaluate.run_mode("schema", _with_gold(sql), "shop", RowsLLM(sql))
    assert out["summary"]["correct"] == 0 and out["summary"]["n_truncated"] == 3
    assert "cut off" in evaluate.format_report({"schema": out})


def test_docs_baseline_and_paired_tests_are_reported(world, tmp_path, capsys):
    out = tmp_path / "r.json"
    assert evaluate.main(["--db", "shop", "--yes", "--json", str(out)], llm=GatedLLM()) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    ex = {m: r["summary"]["ex"] for m, r in data["results"].items()}
    assert ex["docs"] == ex["schema"] < ex["memory"]  # the fake needs the typed rule, not the column description
    assert data["results"]["docs"]["summary"]["mean_rules_used"] > 0
    assert data["run"]["docs_chunks"] == 2 and data["run"]["docs_retrieval_mode"] == "bm25"
    tests = data["comparisons"]
    assert set(tests) == {"memory vs schema", "docs vs schema", "memory vs docs"}
    assert tests["memory vs schema"]["only_b"] == 0 and tests["memory vs schema"]["only_a"] > 0
    assert tests["docs vs schema"]["p_mcnemar"] == 1.0
    shown = capsys.readouterr().out
    assert "paired test" in shown and "memory vs docs" in shown and "McNemar" in shown


def test_docs_mode_without_description_files_stops_early(world, capsys):
    import shutil

    shutil.rmtree(bird.db_path("shop").parent / "database_description")
    assert evaluate.main(["--db", "shop", "--yes"], llm=GatedLLM()) == 1
    assert "database_description" in capsys.readouterr().err
    assert evaluate.main(["--db", "shop", "--yes", "--modes", "schema", "memory"], llm=GatedLLM()) == 0


def test_exclude_kinds_ablates_the_memory_store(world, tmp_path, capsys):
    out = tmp_path / "r.json"
    args = ["--db", "shop", "--yes", "--modes", "schema", "memory", "--exclude-kinds", "metric", "--json", str(out)]
    assert evaluate.main(args, llm=GatedLLM()) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["run"]["excluded_kinds"] == ["metric"]
    ex = {m: r["summary"]["ex"] for m, r in data["results"].items()}
    assert ex["memory"] == ex["schema"]  # the only rule was a metric
    assert "without metric" in capsys.readouterr().out


def test_coverage_and_warnings_are_recorded(world, tmp_path):
    out = tmp_path / "r.json"
    assert evaluate.main(["--db", "shop", "--yes", "--json", str(out)], llm=GatedLLM()) == 0
    res = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert res["schema"]["summary"]["mean_coverage"] is None
    revenue = next(r for r in res["memory"]["questions"] if r["sql"] == GOLD["total revenue"])
    assert revenue["coverage"] == 1.0 and revenue["items_shown"] == ["rev"]  # gold needs amount; the rule names it
    count = next(r for r in res["memory"]["questions"] if r["sql"] == GOLD["number of sales"])
    assert count["coverage"] is None  # COUNT(*) needs no column
    assert all(r["warnings"] == [] for r in res["memory"]["questions"])
