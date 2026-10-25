import numpy as np
import pytest

from app import embedder
from app.config import settings
from app.memory import Item, Store
from app.retrieve import retrieve
from app.textindex import normalise


def make(id, kind="glossary", terms=(), title=None, body="", status="draft"):
    return Item(id=id, db_id="shop", kind=kind, title=title or id, body=body, provenance="schema",
                terms=list(terms), status=status)


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "m.db") as s:
        yield s


def ids(result):
    return [r.item.id for r in result]


def test_normalise_strips_plural_s_on_long_words_only():
    assert normalise("Loans, Gold accounts") == ["loan", "gold", "account"]
    assert normalise("Loans") == ["loan"]
    assert normalise("gas bus") == ["gas", "bus"]  # 3 letters or fewer are kept
    assert normalise("branches") == ["branche"]  # "es" is not special


def test_term_match_beats_search(store):
    store.add(make("g", terms=["gold card"], body="premium"))
    store.add(make("s", kind="convention", title="orders", body="orders orders orders gold gold"))
    result = retrieve("how many orders use a gold card", store, "shop")
    assert result[0].item.id == "g" and result[0].reason == "term:gold card"
    assert result[1].item.id == "s" and result[1].reason.startswith("search:")


def test_plural_and_multiword_terms(store):
    store.add(make("loan", terms=["loan"]))
    store.add(make("acct", terms=["account holder"]))
    assert ids(retrieve("Which LOANS are unpaid?", store, "shop")) == ["loan"]
    assert ids(retrieve("list account holders", store, "shop")) == ["acct"]
    # words must be consecutive
    split = retrieve("account of the holder", store, "shop")
    assert split[0].reason.startswith("search:")


def test_term_match_only_for_term_kinds(store):
    store.add(make("j", kind="join_path", terms=["loan"], body="x"))
    store.add(make("m", kind="metric", terms=["loan"], body="x"))
    result = retrieve("loan", store, "shop")
    assert result[0].item.id == "m" and result[0].reason == "term:loan"
    assert result[1].item.id == "j" and result[1].reason.startswith("search:")


def test_status_preferred_then_longer_term(store):
    store.add(make("d", terms=["card"], status="draft"))
    store.add(make("r", terms=["card"], status="reviewed"))
    store.add(make("c", terms=["card"], status="certified"))
    store.add(make("long", terms=["card", "gold card"], status="reviewed"))
    result = retrieve("a gold card", store, "shop")
    assert ids(result) == ["c", "long", "r", "d"]
    assert result[1].reason == "term:gold card"


def test_status_breaks_ties_between_search_hits(store):
    store.add(make("a-draft", kind="convention", body="refund policy", status="draft"))
    store.add(make("b-cert", kind="convention", body="refund policy", status="certified"))
    assert ids(retrieve("refund policy", store, "shop")) == ["b-cert", "a-draft"]


def test_cap_and_example_inclusion(store):
    for n in range(5):
        store.add(make(f"t{n}", terms=["order"]))
    store.add(make("ex", kind="example", body="unrelated words"))
    result = retrieve("order", store, "shop", k=3)
    assert len(result) == 3
    assert "ex" in ids(result) and result[-1].reason.startswith("search:")
    assert ids(result)[:2] == ["t0", "t1"]


def test_example_not_duplicated_when_already_picked(store):
    store.add(make("t", terms=["order"]))
    store.add(make("ex", kind="example", body="order totals"))
    assert ids(retrieve("order", store, "shop")).count("ex") == 1


def test_no_example_available_and_other_db(store):
    store.add(make("t", terms=["order"]))
    store.add(Item(id="o", db_id="other", kind="example", title="o", body="order", provenance="schema"))
    assert ids(retrieve("order", store, "shop")) == ["t"]
    assert retrieve("order", store, "empty") == []
    assert retrieve("nothing matches zzz", store, "shop") == []


class ConceptEmbedder:
    """Maps synonyms to the same axis, so paraphrases share a vector without sharing a word."""
    AXES = {"car": 0, "automobile": 0, "vehicle": 0, "fruit": 1, "apple": 1}

    def _vec(self, text):
        v = np.full(3, 0.01)
        for w in text.lower().split():
            if w in self.AXES:
                v[self.AXES[w]] += 1.0
        return v

    def embed_query(self, text):
        return self._vec(text)

    def embed_passages(self, texts):
        return np.array([self._vec(t) for t in texts])


def test_embedder_finds_paraphrase(tmp_path):
    items = [make("cars", kind="convention", title="car sales"), make("fruit", kind="convention", title="apple stock")]
    with Store(tmp_path / "plain.db") as plain:
        for i in items:
            plain.add(i)
        assert retrieve("automobile", plain, "shop") == []  # no shared word, no embedder
    with Store(tmp_path / "e.db", embedder=ConceptEmbedder()) as s:
        for i in items:
            s.add(i)
        result = retrieve("automobile", s, "shop", k=1)
        assert ids(result) == ["cars"] and result[0].reason.startswith("search:")


def test_item_without_vector_falls_back_to_bm25(tmp_path):
    path = tmp_path / "m.db"
    with Store(path) as plain:
        plain.add(make("a", kind="convention", title="car sales"))
    with Store(path, embedder=ConceptEmbedder()) as s:
        assert retrieve("automobile", s, "shop") == []
        assert ids(retrieve("car", s, "shop")) == ["a"]


def test_get_embedder_off_and_missing_model(monkeypatch):
    monkeypatch.setattr(embedder, "_loaded", None)
    monkeypatch.setattr(embedder, "_failed", False)
    assert embedder.get_embedder() is None  # conftest sets embed=off

    monkeypatch.setattr(settings, "embed", "auto")

    def boom():
        raise OSError("offline")

    monkeypatch.setattr(embedder, "E5Embedder", boom)
    assert embedder.get_embedder() is None
    assert embedder._failed
