"""Question to answer: retrieve rules (memory mode), generate SQL, run it read-only, repair once on error."""
from dataclasses import dataclass, field
from pathlib import Path

from app import bird
from app.config import settings
from app.embedder import get_embedder
from app.execute import run
from app.llm import LLM, AnthropicLLM, Draft, LLMRefusal
from app.memory import Item, Store
from app.retrieve import retrieve
from app.schema import schema_text

MODES = ("schema", "memory")


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
) -> Result:
    """mode "schema" gives the model the schema only; "memory" adds the retrieved business rules."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    llm = llm or AnthropicLLM()
    db_file = db_file or bird.db_path(db_id)
    own_store = None
    if store is None and mode == "memory":
        store = own_store = Store(settings.memory_db, embedder=get_embedder())
    try:
        items: list[Item] = [r.item for r in retrieve(question, store, db_id)] if mode == "memory" else []
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
        known = _known_items(store, draft) if mode == "memory" else []  # schema mode never reports or counts rules
        if not outcome.error and known:
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


def _known_items(store: Store | None, draft: Draft) -> list[Item]:
    """Items the model cited that exist in the store; invented and repeated ids are dropped."""
    if store is None:
        return []
    found, seen = [], set()
    for item_id in draft.items_used:
        if item_id in seen:
            continue
        seen.add(item_id)
        item = store.get(item_id)
        if item is not None:
            found.append(item)
    return found
