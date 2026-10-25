"""Pick the knowledge items relevant to a question: exact term matches first, then hybrid search."""
from dataclasses import dataclass

from app.memory import Item, Store
from app.textindex import Index, normalise

DEFAULT_K = 8
TERM_KINDS = ("glossary", "metric", "value_hint", "convention")
_STATUS_RANK = {"certified": 0, "reviewed": 1, "draft": 2}


@dataclass
class Retrieved:
    item: Item
    reason: str  # "term:<matched term>" or "search:<score>"


def _contains(tokens: list[str], sub: list[str]) -> bool:
    n = len(sub)
    return n > 0 and any(tokens[i:i + n] == sub for i in range(len(tokens) - n + 1))


def _term_matches(question: str, items: list[Item]) -> list[Retrieved]:
    tokens = normalise(question)
    found = []
    for item in items:
        if item.kind not in TERM_KINDS:
            continue
        hits = [t for t in item.terms if _contains(tokens, normalise(t))]
        if hits:
            found.append((item, max(hits, key=len)))
    # more certain first, then the more specific (longer) term
    found.sort(key=lambda m: (_STATUS_RANK[m[0].status], -len(m[1])))
    return [Retrieved(item, f"term:{term}") for item, term in found]


def retrieve(question: str, store: Store, db_id: str, k: int = DEFAULT_K) -> list[Retrieved]:
    """Up to k items for a question, each with the reason it was picked. Includes an example when any exist."""
    items = store.list(db_id)
    if not items or k <= 0:
        return []
    picked = _term_matches(question, items)[:k]
    taken = {r.item.id for r in picked}

    index = Index(items, [store.vector(i.id) for i in items], store.embedder)
    hits = [h for h in index.search(question, len(items)) if h.item.id not in taken]
    hits.sort(key=lambda h: (-round(h.score, 2), _STATUS_RANK[h.item.status]))
    picked += [Retrieved(h.item, f"search:{h.score:.2f}") for h in hits[:k - len(picked)]]

    if not any(r.item.kind == "example" for r in picked):
        scores = index.scores(question)
        examples = [(-round(float(scores[n]), 2), _STATUS_RANK[i.status], n) for n, i in enumerate(items) if i.kind == "example"]
        if examples:
            n = min(examples)[2]
            if len(picked) >= k:
                picked.pop()
            picked.append(Retrieved(items[n], f"search:{scores[n]:.2f}"))
    return picked
