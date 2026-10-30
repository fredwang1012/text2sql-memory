"""Checks on a query that ran but may still answer the wrong question, and a coverage proxy for retrieval.

The warnings flag results that are rarely what a question wants: no rows, a column that is entirely NULL, or many
identical rows (often a join that fans out). They are reported, not acted on.
"""
import re

from app.execute import _TOKEN
from app.execute import Result as ExecResult
from app.memory import Item, search_text
from app.schema import Table

FANOUT_MIN_ROWS = 10  # below this a few repeated rows say little


def warnings(result: ExecResult) -> list[str]:
    if result.error:
        return []
    if not result.rows:
        return ["empty result"]
    found = []
    for n, name in enumerate(result.columns):
        if all(row[n] is None for row in result.rows):
            found.append(f"column {name} is entirely NULL")
    distinct = len(set(result.rows))
    if len(result.rows) >= FANOUT_MIN_ROWS and distinct <= len(result.rows) // 2:
        found.append(f"{len(result.rows)} rows but only {distinct} distinct: a join may be fanning out")
    return found


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _is_key(name: str) -> bool:
    low = name.lower()
    return low == "id" or low.endswith("_id")


def referenced_columns(sql: str, tables: list[Table]) -> set[str]:
    """Lowercased names of schema columns that a query mentions, leaving out id and *_id join keys.

    Names are matched by spelling, so a name shared by two tables counts once. Quoted identifiers are unwrapped;
    string literals are ignored."""
    names = {c.name.lower() for t in tables for c in t.columns if not _is_key(c.name)}

    def unwrap(m: re.Match) -> str:
        tok = m.group(0)
        if tok.startswith(("--", "/*", "'")):
            return " "
        return " " + tok[1:-1] + " "  # "name", `name` or [name]

    words = {w.lower() for w in _IDENT.findall(_TOKEN.sub(unwrap, sql))}
    return names & words


def coverage(needed: set[str], items: list[Item]) -> float | None:
    """Share of the needed column names that appear as a whole word in the items shown. None when nothing is needed."""
    if not needed:
        return None
    text = " ".join(search_text(i) + " " + (i.sql or "") for i in items).lower()
    words = set(_IDENT.findall(text))
    return len(needed & words) / len(needed)
