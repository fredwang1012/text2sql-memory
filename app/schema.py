"""Schema introspection for a SQLite file and a compact CREATE-TABLE-like rendering used in prompts."""
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

MAX_SAMPLES = 5
MAX_SAMPLE_LEN = 40


@dataclass
class Column:
    name: str
    type: str
    pk: bool
    samples: list[str] = field(default_factory=list)


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column]
    foreign_keys: list[ForeignKey]
    row_count: int


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_text(declared: str) -> bool:
    t = declared.upper()
    return any(k in t for k in ("CHAR", "TEXT", "CLOB"))


def introspect(db_path: Path) -> list[Table]:
    conn = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        tables = []
        for name in names:
            q = _quote(name)
            info = conn.execute(f"PRAGMA table_info({q})").fetchall()  # cid, name, type, notnull, dflt, pk
            columns = []
            for _, cname, ctype, _, _, pk in info:
                samples: list[str] = []
                if _is_text(ctype or ""):
                    qc = _quote(cname)
                    rows = conn.execute(
                        f"SELECT DISTINCT {qc} FROM {q} WHERE {qc} IS NOT NULL AND length({qc}) <= ? LIMIT ?",
                        (MAX_SAMPLE_LEN, MAX_SAMPLES),
                    ).fetchall()
                    samples = [str(r[0]) for r in rows]
                columns.append(Column(cname, ctype or "", bool(pk), samples))
            fks = [
                ForeignKey(r[3], r[2], r[4] or "")
                for r in conn.execute(f"PRAGMA foreign_key_list({q})").fetchall()  # id, seq, table, from, to
            ]
            count = conn.execute(f"SELECT COUNT(*) FROM {q}").fetchone()[0]
            tables.append(Table(name, columns, fks, count))
        return tables
    finally:
        conn.close()


def schema_text(db_path: Path) -> str:
    """Render every table as CREATE TABLE with PK markers, FK clauses, row counts and sample values as comments."""
    blocks = []
    for t in introspect(db_path):
        pk_cols = [c for c in t.columns if c.pk]
        # entries are (sql fragment, trailing comment); commas are added once the clause list is known
        entries: list[tuple[str, str]] = []
        for c in t.columns:
            frag = f"  {_quote(c.name)} {c.type}".rstrip()
            if c.pk and len(pk_cols) == 1:
                frag += " PRIMARY KEY"
            comment = "  -- e.g. " + ", ".join(repr(s) for s in c.samples) if c.samples else ""
            entries.append((frag, comment))
        if len(pk_cols) > 1:
            entries.append(("  PRIMARY KEY (" + ", ".join(_quote(c.name) for c in pk_cols) + ")", ""))
        for fk in t.foreign_keys:
            ref = f"{_quote(fk.ref_table)}({_quote(fk.ref_column)})" if fk.ref_column else _quote(fk.ref_table)
            entries.append((f"  FOREIGN KEY ({_quote(fk.column)}) REFERENCES {ref}", ""))
        body = "\n".join(
            frag + ("," if i < len(entries) - 1 else "") + comment for i, (frag, comment) in enumerate(entries)
        )
        blocks.append(f"CREATE TABLE {_quote(t.name)} (  -- {t.row_count} rows\n{body}\n);")
    return "\n\n".join(blocks)
