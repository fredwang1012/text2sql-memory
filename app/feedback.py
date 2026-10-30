"""Learning from usage: record answers, take the user's verdict, distill a correction into draft rules, and promote
drafts that keep being confirmed.

A correction never edits an existing rule. It adds drafts with provenance feedback:<answer id>, which retrieval ranks
below reviewed and certified items. A draft becomes reviewed after PROMOTE_AFTER good answers cited it, or when a
person approves it; only a person certifies.
"""
from pathlib import Path

from app.execute import run
from app.llm import Learner
from app.memory import Answer, Item, Store
from app.pipeline import Result
from app.retrieve import retrieve
from app.schema import schema_text

PROMOTE_AFTER = 2  # good answers citing a draft before it is promoted to reviewed
MAX_RULES = 3  # rules kept from one correction


def record(store: Store, db_id: str, question: str, mode: str, result: Result) -> Answer:
    return store.add_answer(db_id, question, mode, result.sql, result.items_used)


def mark_good(store: Store, answer_id: int) -> list[Item]:
    """Record a good verdict. Returns the drafts it promoted."""
    answer = store.judge(answer_id, "good")
    promoted = []
    for item_id in answer.items_used:
        item = store.get(item_id)
        if item is not None and item.status == "draft" and store.verdict_counts(item_id)["good"] >= PROMOTE_AFTER:
            promoted.append(store.update(item_id, status="reviewed"))
    return promoted


def _key(kind: str, title: str) -> tuple[str, str]:
    return kind, " ".join(title.lower().split())


def mark_wrong(store: Store, answer_id: int, correction: str, learner: Learner, db_file: Path,
               fixed_sql: str | None = None) -> list[Item]:
    """Record a wrong verdict and distill the correction into draft items. A fixed query that runs also becomes a
    draft example. Returns the items added."""
    answer = store.answer(answer_id)
    if answer is None:
        raise KeyError(answer_id)
    if answer.verdict == "wrong":
        raise ValueError(f"answer {answer_id} was already marked wrong")
    if not correction.strip() and not fixed_sql:
        raise ValueError("say what is wrong, or give the fixed SQL")
    if fixed_sql:
        check = run(db_file, fixed_sql)
        if check.error:
            raise ValueError(f"the fixed SQL does not run: {check.error}")

    added = []
    known = {_key(i.kind, i.title) for i in store.list(answer.db_id)}
    if correction.strip():
        related = [r.item for r in retrieve(f"{answer.question} {correction}", store, answer.db_id)]
        proposed = learner.distill(answer.question, schema_text(db_file), answer.sql, correction, related)
        for rule in proposed[:MAX_RULES]:
            terms = [t.strip().lower() for t in rule.terms if t.strip()]
            if not rule.title.strip() or not rule.body.strip() or _key(rule.kind, rule.title) in known:
                continue
            known.add(_key(rule.kind, rule.title))
            added.append(store.add(Item(
                id=f"fb{answer_id}-{len(added) + 1}", db_id=answer.db_id, kind=rule.kind, title=rule.title.strip(),
                terms=terms, body=rule.body.strip(), sql=(rule.sql or "").strip() or None,
                provenance=f"feedback:{answer_id}", status="draft")))
    if fixed_sql:
        added.append(store.add(Item(
            id=f"fb{answer_id}-ex", db_id=answer.db_id, kind="example", title=answer.question,
            body=f"Question: {answer.question}" + (f"\nNote: {correction.strip()}" if correction.strip() else ""),
            sql=fixed_sql.strip(), provenance=f"feedback:{answer_id}", status="draft")))
    store.judge(answer_id, "wrong", correction.strip() or None, fixed_sql, [i.id for i in added])
    return added


def review_queue(store: Store, db_id: str) -> list[tuple[Item, dict[str, int]]]:
    """Drafts with how often answers citing them were judged good or wrong, most confirmed first."""
    rows = [(i, store.verdict_counts(i.id)) for i in store.list(db_id, status="draft")]
    return sorted(rows, key=lambda r: (-r[1]["good"], r[1]["wrong"], r[0].id))


def set_status(store: Store, item_id: str, status: str) -> Item:
    if store.get(item_id) is None:
        raise KeyError(item_id)
    return store.update(item_id, status=status)
