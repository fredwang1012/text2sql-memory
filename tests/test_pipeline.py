import sqlite3
from types import SimpleNamespace

import anthropic
import pytest

from app.cli import main as cli_main
from app.config import settings
from app.llm import SYSTEM_PROMPT, AnthropicLLM, Draft, LLMRefusal, LLMUnavailable
from app.memory import Item, Store
from app.pipeline import Result, answer


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE sale (id INTEGER PRIMARY KEY, amount REAL); INSERT INTO sale VALUES (1,2.5),(2,4);")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "mem.db") as s:
        s.add(Item(id="r-1", db_id="shop", kind="metric", title="Revenue", terms=["revenue"],
                   body="revenue is the sum of sale.amount", provenance="schema"))
        s.add(Item(id="r-2", db_id="shop", kind="glossary", title="Other", terms=["zzz"],
                   body="unrelated", provenance="schema"))
        yield s


class FakeLLM:
    def __init__(self, drafts, refuse=False):
        self.drafts = list(drafts)
        self.refuse = refuse
        self.calls = []

    def generate(self, question, schema_text, items):
        self.calls.append(("generate", [i.id for i in items]))
        if self.refuse:
            raise LLMRefusal("declined")
        return self.drafts.pop(0)

    def repair(self, question, schema_text, items, sql, error):
        self.calls.append(("repair", sql, error))
        return self.drafts.pop(0)


def draft(sql, used=()):
    return Draft(sql=sql, items_used=list(used), assumptions="")


def test_schema_mode_passes_no_items(db, store):
    fake = FakeLLM([draft("SELECT SUM(amount) FROM sale")])
    res = answer("what is the revenue", "shop", "schema", llm=fake, store=store, db_file=db)
    assert fake.calls == [("generate", [])]
    assert res.rows == [(6.5,)] and res.error is None


def test_memory_mode_passes_retrieved_items(db, store):
    fake = FakeLLM([draft("SELECT SUM(amount) FROM sale", ["r-1", "ghost"])])
    res = answer("what is the revenue", "shop", "memory", llm=fake, store=store, db_file=db)
    assert "r-1" in fake.calls[0][1]
    assert res.items_used == ["r-1"]  # invented id dropped
    assert res.explanation == ["Revenue (r-1)"]


def test_repair_path(db, store):
    fake = FakeLLM([draft("SELECT nope FROM sale"), draft("SELECT COUNT(*) FROM sale")])
    res = answer("how many", "shop", "schema", llm=fake, store=store, db_file=db)
    assert [c[0] for c in fake.calls] == ["generate", "repair"]
    assert "no such column" in fake.calls[1][2]
    assert res.rows == [(2,)] and res.sql == "SELECT COUNT(*) FROM sale"


def test_repair_only_once(db, store):
    fake = FakeLLM([draft("SELECT a FROM sale"), draft("SELECT b FROM sale")])
    res = answer("q", "shop", "schema", llm=fake, store=store, db_file=db)
    assert len(fake.calls) == 2
    assert res.error and "no such column" in res.error


def test_refusal_becomes_error_result(db, store):
    res = answer("q", "shop", "schema", llm=FakeLLM([], refuse=True), store=store, db_file=db)
    assert res.error == "declined" and res.sql == ""


def test_use_counting(db, store):
    ok = FakeLLM([draft("SELECT SUM(amount) FROM sale", ["r-1", "r-1"])])
    answer("revenue", "shop", "memory", llm=ok, store=store, db_file=db)
    assert store.get("r-1").uses == 1
    bad = FakeLLM([draft("SELECT x FROM sale", ["r-1"]), draft("SELECT y FROM sale", ["r-1"])])
    answer("revenue", "shop", "memory", llm=bad, store=store, db_file=db)
    assert store.get("r-1").uses == 1  # a failed query does not count


def test_unknown_mode(db):
    with pytest.raises(ValueError):
        answer("q", "shop", "other", llm=FakeLLM([]), db_file=db)


class _Parse:
    def __init__(self, response=None, exc=None):
        self.kwargs, self.response, self.exc = None, response, exc

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.response


def _patched(monkeypatch, parse):
    monkeypatch.setattr(anthropic.Anthropic, "__init__", lambda self, *a, **k: None)
    monkeypatch.setattr(anthropic.Anthropic, "messages", SimpleNamespace(parse=parse), raising=False)


def test_anthropic_request_shape(monkeypatch, store):
    parsed = draft("SELECT 1", ["r-1"])
    parse = _Parse(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed))
    _patched(monkeypatch, parse)
    out = AnthropicLLM().generate("the question", "CREATE TABLE sale (...)", [store.get("r-1")])
    assert out is parsed
    kw = parse.kwargs
    assert kw["model"] == settings.llm_model
    assert kw["max_tokens"] == 16000
    assert kw["system"] == SYSTEM_PROMPT
    assert kw["output_format"] is Draft
    assert kw["output_config"] == {"effort": settings.llm_effort}
    for banned in ("thinking", "temperature", "top_p", "top_k"):
        assert banned not in kw
    assert kw["messages"][0]["role"] == "user"
    text = kw["messages"][0]["content"]
    assert "CREATE TABLE sale" in text and "[r-1]" in text and "the question" in text

    AnthropicLLM().repair("the question", "SCHEMA", [], "SELECT bad", "boom")
    text = parse.kwargs["messages"][0]["content"]
    assert "the question" in text and "SELECT bad" in text and "boom" in text


def test_anthropic_refusal(monkeypatch):
    _patched(monkeypatch, _Parse(SimpleNamespace(stop_reason="refusal", parsed_output=None)))
    with pytest.raises(LLMRefusal):
        AnthropicLLM().generate("q", "s", [])


def test_anthropic_credentials_error_is_unavailable(monkeypatch):
    _patched(monkeypatch, _Parse(exc=anthropic.CredentialsError("no credentials")))
    with pytest.raises(LLMUnavailable):
        AnthropicLLM().generate("q", "s", [])


def test_client_is_lazy():
    assert AnthropicLLM()._client is None


def test_cli_unavailable_message(monkeypatch, capsys):
    monkeypatch.setattr("app.cli.bird.data_missing", lambda db_id: False)
    def boom(*a, **k):
        raise LLMUnavailable("x")

    monkeypatch.setattr("app.cli.answer", boom)
    assert cli_main(["ask", "q"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_ask_prints_sql_rules_rows(monkeypatch, capsys):
    monkeypatch.setattr("app.cli.bird.data_missing", lambda db_id: False)
    res = Result(sql="SELECT 1", rows=[(7,)], columns=["x"], items_used=["r-1"], explanation=["Revenue (r-1)"])
    monkeypatch.setattr("app.cli.answer", lambda *a, **k: res)
    assert cli_main(["ask", "q", "--mode", "schema"]) == 0
    out = capsys.readouterr().out
    assert "SELECT 1" in out and "Revenue (r-1)" in out and "7" in out


def test_cli_items_lists(tmp_path, monkeypatch, capsys):
    path = tmp_path / "cli.db"
    with Store(path) as s:
        s.add(Item(id="r-1", db_id="shop", kind="metric", title="Revenue", body="b", provenance="schema"))
    monkeypatch.setattr(settings, "memory_db", path)
    assert cli_main(["--db-id", "shop", "items"]) == 0
    assert "r-1" in capsys.readouterr().out


def test_schema_mode_never_counts_or_reports_items(db, store):
    fake = FakeLLM([draft("SELECT SUM(amount) FROM sale", ["r-1"])])
    res = answer("revenue", "shop", "schema", llm=fake, store=store, db_file=db)
    assert res.items_used == [] and res.explanation == []
    assert store.get("r-1").uses == 0


def test_refusal_during_repair_keeps_first_attempt(db, store):
    class RefusesRepair(FakeLLM):
        def repair(self, *a):
            raise LLMRefusal("declined")

    res = answer("q", "shop", "schema", llm=RefusesRepair([draft("SELECT nope FROM sale")]), store=store, db_file=db)
    assert res.sql == "SELECT nope FROM sale"
    assert "no such column" in res.error and "declined" in res.error


def test_anthropic_credentials_error(monkeypatch):
    _patched(monkeypatch, _Parse(exc=anthropic.CredentialsError("no credentials")))
    with pytest.raises(LLMUnavailable):
        AnthropicLLM().generate("q", "s", [])


def test_cli_without_credentials_is_friendly(monkeypatch, capsys, db):
    _patched(monkeypatch, _Parse(exc=anthropic.CredentialsError("no credentials")))
    monkeypatch.setattr("app.pipeline.bird.db_path", lambda db_id: db)
    monkeypatch.setattr("app.cli.bird.data_missing", lambda db_id: False)
    assert cli_main(["ask", "q", "--mode", "schema"]) == 1
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "ant auth login" in err and "Traceback" not in err
