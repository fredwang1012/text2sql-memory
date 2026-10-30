"""Question to answer: retrieve rules (memory mode) or doc chunks (docs mode), generate SQL, run it read-only, repair
once on error."""
from dataclasses import dataclass, field
from pathlib import Path

from app import bird
from app.config import settings
from app.docs import Docs
from app.embedder import get_embedder
from app.execute import run
from app.llm import LLM, AnthropicLLM, Draft, LLMRefusal
from app.memory import Item, Store
from app.retrieve import retrieve
from app.schema import schema_text

MODES = ("schema", "docs", "memory")


@dataclass
class Result:
    sql: str = ""
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    items_used: list[str] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False  # rows hit the row cap
    explanation: list[str] = field(default_factory=list)  # "<title> (<id>)" per rule used


def answer(
    question: str,
    db_id: str,
    mode: str,
    llm: LLM | None = None,
    store: Store | None = None,
    db_file: Path | None = None,
    docs: Docs | None = None,
) -> Result:
    """mode "schema" gives the model the schema only; "docs" adds the closest data-dictionary chunks (the flat-RAG
    baseline); "memory" adds the retrieved business rules."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    llm = llm or AnthropicLLM()
    db_file = db_file or bird.db_path(db_id)
    own_store = None
    if store is None and mode == "memory":
        store = own_store = Store(settings.memory_db, embedder=get_embedder())
    try:
        if mode == "memory":
            items: list[Item] = [r.item for r in retrieve(question, store, db_id)]
        elif mode == "docs":
            items = (docs or Docs.load(db_id, get_embedder())).retrieve(question)
        else:
            items = []
        schema = schema_text(db_file)
        try:
            draft = llm.generate(question, schema, items)
        except LLMRefusal as e:
            return Result(error=str(e))
        outcome = run(db_file, draft.sql)
        if outcome.error:
            try:
                retry = llm.repair(question, schema, items, draft.sql, outcome.error)
            except LLMRefusal as e:
                # keep the first attempt and its error, and note the refusal
                return Result(sql=draft.sql, error=f"{outcome.error} (repair refused: {e})")
            draft = retry
            outcome = run(db_file, draft.sql)
        known = _cited(items, draft)
        if mode == "memory" and not outcome.error and known:
            store.record_use([i.id for i in known])
        return Result(
            sql=draft.sql,
            rows=outcome.rows,
            columns=outcome.columns,
            items_used=[i.id for i in known],
            error=outcome.error,
            truncated=outcome.truncated,
            explanation=[f"{i.title} ({i.id})" for i in known],
        )
    finally:
        if own_store is not None:
            own_store.close()


def _cited(items: list[Item], draft: Draft) -> list[Item]:
    """Items the model cited out of those it was shown, in citation order; invented and repeated ids are dropped."""
    shown = {i.id: i for i in items}
    found, seen = [], set()
    for item_id in draft.items_used:
        if item_id in shown and item_id not in seen:
            seen.add(item_id)
            found.append(shown[item_id])
    return found
