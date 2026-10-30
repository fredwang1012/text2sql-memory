import sqlite3

import pytest

from app import cli, feedback
from app.config import settings
from app.llm import Draft, ProposedRule
from app.memory import Item, Store
from app.pipeline import Result
from app.retrieve import retrieve


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE sale (id INTEGER PRIMARY KEY, amount REAL, status TEXT);"
                       "INSERT INTO sale VALUES (1, 2.5, 'ok'), (2, 4, 'void');")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "mem.db") as s:
        s.add(Item(id="rev", db_id="shop", kind="metric", title="Revenue", terms=["revenue"],
                   body="revenue is the sum of sale.amount", provenance="schema", status="reviewed"))
        yield s


class Learner:
    def __init__(self, rules):
        self.rules, self.calls = rules, []

    def distill(self, question, schema_text, sql, correction, items):
        self.calls.append((question, sql, correction, [i.id for i in items]))
        return self.rules


VOID = ProposedRule(kind="convention", title="Exclude voided sales", terms=["sales", "revenue"],
                    body="Voided sales have status 'void' and are excluded from every total.", sql="status <> 'void'")


def _answer(store, sql="SELECT SUM(amount) FROM sale", used=("rev",)):
    return feedback.record(store, "shop", "What is the revenue?", "memory", Result(sql=sql, items_used=list(used)))


def test_record_and_mark_good(store):
    a = _answer(store)
    assert store.answer(a.id).verdict is None and a.items_used == ["rev"]
    assert feedback.mark_good(store, a.id) == []  # rev is already reviewed
    assert store.answer(a.id).verdict == "good"
    assert store.verdict_counts("rev") == {"good": 1, "wrong": 0}
    with pytest.raises(KeyError):
        feedback.mark_good(store, 999)


def test_correction_becomes_draft_rules_that_retrieval_finds(store, db):
    a = _answer(store)
    learner = Learner([VOID])
    added = feedback.mark_wrong(store, a.id, "voided sales must not count", learner, db)
    assert [i.id for i in added] == [f"fb{a.id}-1"]
    item = store.get(f"fb{a.id}-1")
    assert item.status == "draft" and item.provenance == f"feedback:{a.id}" and item.sql == "status <> 'void'"
    assert learner.calls[0][1] == "SELECT SUM(amount) FROM sale" and "rev" in learner.calls[0][3]
    judged = store.answer(a.id)
    assert judged.verdict == "wrong" and judged.distilled == [item.id] and judged.correction
    hits = [r.item.id for r in retrieve("total revenue of sales last year", store, "shop")]
    assert item.id in hits and hits.index("rev") < hits.index(item.id)  # reviewed before draft


def test_duplicates_blank_rules_and_extras_are_dropped(store, db):
    rules = [VOID, VOID, ProposedRule(kind="metric", title="revenue ", terms=[], body="dup of an existing title"),
             ProposedRule(kind="glossary", title=" ", terms=[], body="x"),
             ProposedRule(kind="glossary", title="Status", terms=[" Status "], body="sale.status is ok or void"),
             ProposedRule(kind="glossary", title="Fourth", terms=[], body="over the limit")]
    added = feedback.mark_wrong(store, _answer(store).id, "fix it", Learner(rules), db)
    assert [i.title for i in added] == ["Exclude voided sales"]  # only the first 3 proposals are considered
    added = feedback.mark_wrong(store, _answer(store).id, "fix it", Learner(rules[4:5]), db)
    assert [(i.title, i.terms) for i in added] == [("Status", ["status"])]


def test_fixed_sql_becomes_a_draft_example_only_if_it_runs(store, db):
    a = _answer(store)
    with pytest.raises(ValueError, match="does not run"):
        feedback.mark_wrong(store, a.id, "", Learner([]), db, fixed_sql="SELECT nope FROM sale")
    assert store.answer(a.id).verdict is None
    fixed = "SELECT SUM(amount) FROM sale WHERE status <> 'void'"
    learner = Learner([])
    added = feedback.mark_wrong(store, a.id, "", learner, db, fixed_sql=fixed)
    assert learner.calls == []  # nothing to distill without a correction
    assert [(i.kind, i.sql, i.status) for i in added] == [("example", fixed, "draft")]
    with pytest.raises(ValueError, match="already"):
        feedback.mark_wrong(store, a.id, "again", learner, db)
    with pytest.raises(ValueError, match="say what is wrong"):
        feedback.mark_wrong(store, _answer(store).id, "  ", learner, db)


def test_drafts_are_promoted_after_repeated_good_answers(store, db):
    (draft,) = feedback.mark_wrong(store, _answer(store).id, "voids", Learner([VOID]), db)
    first = _answer(store, used=("rev", draft.id))
    assert feedback.mark_good(store, first.id) == []
    second = _answer(store, used=(draft.id,))
    (promoted,) = feedback.mark_good(store, second.id)
    assert promoted.id == draft.id and promoted.status == "reviewed" and promoted.version == 2
    assert store.history(draft.id)[0].status == "draft"


def test_review_queue_and_status_changes(store, db):
    a, b = feedback.mark_wrong(store, _answer(store).id, "x", Learner([VOID, ProposedRule(
        kind="glossary", title="Status", terms=["status"], body="ok or void")]), db)
    feedback.mark_good(store, _answer(store, used=(b.id,)).id)
    assert [i.id for i, _ in feedback.review_queue(store, "shop")] == [b.id, a.id]
    assert feedback.set_status(store, a.id, "certified").status == "certified"
    store.archive(b.id)
    assert store.get(b.id) is None and store.history(b.id)[-1].title == "Status"
    assert feedback.review_queue(store, "shop") == []
    with pytest.raises(KeyError):
        feedback.set_status(store, "ghost", "reviewed")


def _bird_shop():
    db_dir = settings.data_dir / "bird" / "dev_databases" / "shop"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "shop.sqlite")
    conn.executescript("CREATE TABLE sale (id INTEGER PRIMARY KEY, amount REAL, status TEXT);"
                       "INSERT INTO sale VALUES (1, 2.5, 'ok');")
    conn.commit()
    conn.close()
    (settings.data_dir / "bird" / "dev.json").write_text("[]", encoding="utf-8")


def test_cli_feedback_round_trip(monkeypatch, capsys):
    _bird_shop()

    class Fake:
        def generate(self, *a):
            return Draft(sql="SELECT SUM(amount) FROM sale", items_used=[], assumptions="")

        def distill(self, *a):
            return [VOID]

    monkeypatch.setattr("app.pipeline.make_llm", Fake)
    monkeypatch.setattr("app.cli.make_llm", Fake)
    assert cli.main(["--db-id", "shop", "ask", "--mode", "schema", "What is the revenue?"]) == 0
    assert "Answer #1." in capsys.readouterr().out
    assert cli.main(["wrong", "1", "voided sales must not count"]) == 0
    assert "fb1-1" in capsys.readouterr().out
    assert cli.main(["wrong", "1", "again"]) == 1 and "already" in capsys.readouterr().err
    assert cli.main(["--db-id", "shop", "review"]) == 0
    assert "Exclude voided sales" in capsys.readouterr().out
    assert cli.main(["approve", "fb1-1"]) == 0 and "reviewed" in capsys.readouterr().out
    assert cli.main(["--db-id", "shop", "review"]) == 0 and "No draft" in capsys.readouterr().out
    assert cli.main(["reject", "fb1-1"]) == 0
    assert cli.main(["reject", "fb1-1"]) == 1
    assert cli.main(["good", "42"]) == 1
    assert cli.main(["--db-id", "shop", "ask", "--mode", "schema", "again?"]) == 0
    assert cli.main(["good", "2"]) == 0 and "good" in capsys.readouterr().out
