"""Read-only SQL execution with a statement check, timeout and row cap, plus BIRD-style result comparison."""
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

# quoted text comes first so comment markers inside literals are left alone
_TOKEN = re.compile(r"""'(?:[^']|'')*'|"(?:[^"]|"")*"|`[^`]*`|\[[^\]]*\]|--[^\n]*|/\*.*?(?:\*/|$)""", re.S)
_FORBIDDEN = re.compile(r"\b(PRAGMA|ATTACH|DETACH|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE|VACUUM|REINDEX)\b", re.I)


@dataclass
class Result:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False


def _strip_comments(sql: str) -> str:
    def repl(m: re.Match) -> str:
        return " " if m.group(0).startswith(("--", "/*")) else m.group(0)

    return _TOKEN.sub(repl, sql)


def check_statement(sql: str) -> str | None:
    """Return an error message when sql is not a single SELECT/WITH statement, else None."""
    text = _strip_comments(sql).strip()
    while text.endswith(";"):
        text = text[:-1].rstrip()
    if not text:
        return "empty statement"
    bare = _TOKEN.sub(" ", text)  # literals and quoted names removed
    if ";" in bare:
        return "multiple statements are not allowed"
    if not re.match(r"(SELECT|WITH)\b", bare, re.I):
        return "only SELECT or WITH statements are allowed"
    # the replace() function is fine; REPLACE INTO is not
    m = _FORBIDDEN.search(re.sub(r"\bREPLACE\s*\(", "(", bare, flags=re.I))
    if m:
        return f"forbidden keyword: {m.group(1).upper()}"
    return None


def run(db_path: Path, sql: str, timeout_s: float = 10, max_rows: int = 10000) -> Result:
    err = check_statement(sql)
    if err:
        return Result(error=err)
    conn = None
    try:
        conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        return Result(rows=[tuple(r) for r in rows[:max_rows]], columns=columns, truncated=len(rows) > max_rows)
    except sqlite3.OperationalError as e:
        return Result(error="timeout" if "interrupted" in str(e) else str(e))
    except sqlite3.Error as e:
        return Result(error=str(e))
    finally:
        if conn is not None:
            conn.close()


def _norm(v):
    return round(v, 6) if isinstance(v, float) else v


def same_result(a: Result, b: Result) -> bool:
    """BIRD execution match: the sets of row tuples are equal. Order and duplicates are ignored."""
    if a.error or b.error or a.truncated or b.truncated:  # a cut-off result cannot be compared
        return False
    return {tuple(map(_norm, r)) for r in a.rows} == {tuple(map(_norm, r)) for r in b.rows}
