import json

import numpy as np
import pytest

from app.bird import Question, split
from app.config import ROOT
from app.memory import Item, Store, seed_ids
from app.seed import export_yaml, import_yaml, knowledge_path, load_items


def make(id="t-1", **kw) -> Item:
    base = dict(id=id, db_id="shop", kind="glossary", title="Revenue", body="Sum of paid orders.",
                provenance="schema", terms=["revenue", "sales"], status="reviewed")
    return Item(**{**base, **kw})


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "memory.db") as s:
        yield s


def test_add_get_list_and_record_use(store):
    added = store.add(make())
    assert added.version == 1 and added.uses == 0 and added.created
    got = store.get("t-1")
    assert got.terms == ["revenue", "sales"] and got.sql is None and got.scope == "global"
    store.add(make("t-2", kind="example", sql="SELECT 1", status="draft"))
    store.add(make("o-1", db_id="other"))
    assert [i.id for i in store.list("shop")] == ["t-1", "t-2"]
    assert [i.id for i in store.list("shop", kind="example")] == ["t-2"]
    store.record_use(["t-1", "t-1", "t-2"])
    assert store.get("t-1").uses == 2 and store.get("t-2").uses == 1
    assert store.get("missing") is None


def test_status_filter(store):
    store.add(make("a", status="draft"))
    store.add(make("b", status="reviewed"))
    store.add(make("c", status="certified"))
    assert [i.id for i in store.list("shop", status="reviewed")] == ["b"]
    assert [i.id for i in store.list("shop", kind="glossary", status="draft")] == ["a"]


def test_validation_and_duplicates(store):
    store.add(make())
    with pytest.raises(ValueError):
        store.add(make())
    with pytest.raises(ValueError):
        store.add(make("x", kind="nonsense"))
    with pytest.raises(ValueError):
        store.add(make("y", status="approved"))
    with pytest.raises(ValueError):
        store.add(make("z", provenance=""))
    for bad in ("schema+", "seed:", "seed:x", "docs+web", "Schema", "seed:1 + data", "notes"):
        with pytest.raises(ValueError):
            store.add(make("p", provenance=bad))
    for good in ("schema", "docs", "data", "seed:12", "seed:131+docs+data", "data+docs"):
        make("p", provenance=good).validate()
    with pytest.raises(ValueError):
        store.update("t-1", version=9)
    with pytest.raises(KeyError):
        store.update("nope", title="x")


def test_update_keeps_history(store):
    store.add(make())
    store.record_use(["t-1"])
    v2 = store.update("t-1", body="Sum of paid order amounts.")
    v3 = store.update("t-1", status="certified", terms=["revenue"])
    assert (v2.version, v3.version) == (2, 3)
    current = store.get("t-1")
    assert current.status == "certified" and current.terms == ["revenue"] and current.uses == 1
    history = store.history("t-1")
    assert [h.version for h in history] == [1, 2]
    assert history[0].body == "Sum of paid orders." and history[1].body == "Sum of paid order amounts."
    assert history[1].status == "reviewed"
    # an update that changes nothing is not a new version
    assert store.update("t-1", status="certified").version == 3
    assert len(store.history("t-1")) == 2


class FakeEmbedder:
    def embed_query(self, text):
        return np.array([float(len(text)), 1.0, 0.0])

    def embed_passages(self, texts):
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts])


def test_vectors_only_with_embedder(tmp_path):
    with Store(tmp_path / "a.db") as plain:
        plain.add(make())
        assert plain.vector("t-1") is None
    with Store(tmp_path / "b.db", embedder=FakeEmbedder()) as s:
        s.add(make())
        first = s.vector("t-1")
        assert first.dtype == np.float32 and first.shape == (3,)
        s.update("t-1", body="A much longer body than before.")
        assert s.vector("t-1")[0] > first[0]


def test_upsert_fills_missing_vector(tmp_path):
    path = tmp_path / "m.db"
    with Store(path) as plain:
        plain.add(make())
    with Store(path, embedder=FakeEmbedder()) as s:
        assert s.vector("t-1") is None
        item, outcome = s.upsert(make())
        assert outcome == "unchanged" and item.version == 1
        assert s.vector("t-1") is not None


YAML = """\
db_id: shop
items:
  - id: shop-001
    kind: glossary
    title: Revenue
    terms: [revenue, sales]
    body: |
      Sum of paid orders.
      Refunds are excluded.
    provenance: schema
    status: reviewed
  - id: shop-002
    kind: example
    title: Count customers
    terms: [how many customers]
    body: Number of customers.
    sql: SELECT COUNT(*) FROM customer
    provenance: "seed:7"
    status: reviewed
    scope: team
"""


def test_yaml_round_trip(store, tmp_path):
    src = tmp_path / "shop.yaml"
    src.write_text(YAML, encoding="utf-8")
    assert import_yaml(store, src) == {"added": 2, "updated": 0, "unchanged": 0}
    item = store.get("shop-001")
    assert item.body == "Sum of paid orders.\nRefunds are excluded." and item.db_id == "shop"
    assert store.get("shop-002").scope == "team" and store.get("shop-002").provenance == "seed:7"
    out = tmp_path / "out.yaml"
    assert export_yaml(store, "shop", out) == 2
    assert [i.content() for i in load_items(out)] == [i.content() for i in load_items(src)]
    # re-importing an export changes nothing
    assert import_yaml(store, out) == {"added": 0, "updated": 0, "unchanged": 2}


def test_import_upsert_bumps_version(store, tmp_path):
    src = tmp_path / "shop.yaml"
    src.write_text(YAML, encoding="utf-8")
    import_yaml(store, src)
    src.write_text(YAML.replace("Refunds are excluded.", "Refunds are subtracted.")
                   + "  - {id: shop-003, kind: convention, title: T, body: B, provenance: data}\n", encoding="utf-8")
    assert import_yaml(store, src) == {"added": 1, "updated": 1, "unchanged": 1}
    assert store.get("shop-001").version == 2 and store.get("shop-002").version == 1
    assert store.history("shop-001")[0].body.endswith("Refunds are excluded.")


def test_load_rejects_bad_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("db_id: shop\nitems:\n  - {id: a, kind: glossary, title: T, body: B, provenance: schema, colour: red}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(bad)
    bad.write_text("db_id: shop\nitems:\n  - {id: a, kind: glossary, title: T, body: B, provenance: schema}\n"
                   "  - {id: a, kind: glossary, title: T, body: B, provenance: schema}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_items(bad)


def test_financial_knowledge_file_is_valid():
    items = load_items(knowledge_path("financial"))
    assert len(items) >= 25
    assert all(i.db_id == "financial" and i.terms for i in items)
    assert {i.kind for i in items} == {"glossary", "metric", "join_path", "convention", "example", "value_hint"}
    assert all(i.sql for i in items if i.kind == "example")


def test_financial_knowledge_cites_only_seed_questions():
    """Honesty guard: every seed:<id> in a provenance must be in the seed split. The split hashes each question id
    on its own, so recomputing it over the cited ids needs no BIRD download and gives the same answer as the full
    split. When the recorded split file exists locally it is checked as well."""
    cited = {q for i in load_items(knowledge_path("financial")) for q in seed_ids(i.provenance)}
    assert cited
    stubs = [Question(question_id=q, db_id="financial", question="", evidence="", sql="", difficulty="")
             for q in cited]
    seed_qs, eval_qs = split(stubs)  # the defaults are the parameters used for financial
    assert not eval_qs, f"provenance cites eval-split questions: {sorted(q.question_id for q in eval_qs)}"
    recorded = ROOT / "data" / "split-financial.json"  # conftest moves settings.data_dir, so use the repo path
    if recorded.exists():
        record = json.loads(recorded.read_text(encoding="utf-8"))
        assert (record["seed"], record["seed_fraction"]) == (13, 0.3)
        assert cited <= set(record["seed_ids"])
