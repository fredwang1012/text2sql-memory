"""Typed, versioned knowledge store in SQLite: items, their previous versions, and optional embedding vectors."""
from __future__ import annotations  # the Store.list method shadows the builtin inside the class body

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KINDS = ("glossary", "metric", "join_path", "convention", "example", "value_hint")
STATUSES = ("draft", "reviewed", "certified")
SCOPES = ("global", "team", "user")

# provenance is one or more sources joined with "+": schema, docs (BIRD description files), data, or seed:<question_id>
_SOURCE = r"(?:schema|docs|data|seed:\d+)"
_PROVENANCE = re.compile(rf"{_SOURCE}(?:\+{_SOURCE})*")

# fields a user edits; a change to any of them is a new version
CONTENT_FIELDS = ("db_id", "kind", "title", "terms", "body", "sql", "provenance", "status", "scope")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, db_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, terms TEXT NOT NULL,
    body TEXT NOT NULL, sql TEXT, provenance TEXT NOT NULL, status TEXT NOT NULL, scope TEXT NOT NULL,
    version INTEGER NOT NULL, uses INTEGER NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS item_versions (
    id TEXT NOT NULL, version INTEGER NOT NULL, db_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
    terms TEXT NOT NULL, body TEXT NOT NULL, sql TEXT, provenance TEXT NOT NULL, status TEXT NOT NULL,
    scope TEXT NOT NULL, uses INTEGER NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL, archived TEXT NOT NULL,
    PRIMARY KEY (id, version)
);
CREATE TABLE IF NOT EXISTS item_vectors (id TEXT PRIMARY KEY, vector BLOB NOT NULL);
CREATE INDEX IF NOT EXISTS items_db_kind ON items (db_id, kind);
"""

_COLUMNS = ("id",) + CONTENT_FIELDS + ("version", "uses", "created", "updated")


@dataclass
class Item:
    id: str
    db_id: str
    kind: str
    title: str
    body: str
    provenance: str
    terms: list[str] = field(default_factory=list)
    sql: str | None = None
    status: str = "draft"
    scope: str = "global"
    version: int = 1
    uses: int = 0
    created: str = ""
    updated: str = ""

    def validate(self) -> None:
        if not self.id:
            raise ValueError("item id is required")
        if self.kind not in KINDS:
            raise ValueError(f"{self.id}: unknown kind {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"{self.id}: unknown status {self.status!r}")
        if self.scope not in SCOPES:
            raise ValueError(f"{self.id}: unknown scope {self.scope!r}")
        if not self.provenance or not _PROVENANCE.fullmatch(self.provenance):
            raise ValueError(f"{self.id}: invalid provenance {self.provenance!r}")

    def content(self) -> dict:
        return {f: getattr(self, f) for f in CONTENT_FIELDS}


def seed_ids(provenance: str) -> list[int]:
    """Seed question ids cited in a provenance string."""
    return [int(p[5:]) for p in provenance.split("+") if p.startswith("seed:")]


def search_text(item: Item) -> str:
    """The text an item is matched on: title, terms and body."""
    return " ".join([item.title, *item.terms, item.body])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_item(row: sqlite3.Row) -> Item:
    d = {c: row[c] for c in _COLUMNS}
    d["terms"] = json.loads(d["terms"])
    return Item(**d)


class Store:
    """SQLite-backed item store. Vectors are written only when an embedder (anything with
    embed_passages(list[str]) -> array of vectors) is given."""

    def __init__(self, path: Path, embedder=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _write(self, item: Item) -> None:
        values = asdict(item)
        values["terms"] = json.dumps(item.terms, ensure_ascii=False)
        self.conn.execute(
            f"INSERT OR REPLACE INTO items ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
            [values[c] for c in _COLUMNS],
        )
        if self.embedder is not None:
            vec = np.asarray(self.embedder.embed_passages([search_text(item)])[0], dtype=np.float32)
            self.conn.execute("INSERT OR REPLACE INTO item_vectors (id, vector) VALUES (?, ?)", (item.id, vec.tobytes()))

    def add(self, item: Item) -> Item:
        item.validate()
        if self.get(item.id) is not None:
            raise ValueError(f"item {item.id} already exists")
        now = _now()
        item = Item(**{**asdict(item), "version": 1, "uses": 0, "created": now, "updated": now})
        with self.conn:
            self._write(item)
        return item

    def update(self, item_id: str, **changes) -> Item:
        """Apply content changes, archive the current version and bump the version number.
        Returns the item unchanged (same version) when nothing differs."""
        unknown = set(changes) - set(CONTENT_FIELDS)
        if unknown:
            raise ValueError(f"cannot update fields: {sorted(unknown)}")
        old = self.get(item_id)
        if old is None:
            raise KeyError(item_id)
        if all(getattr(old, k) == v for k, v in changes.items()):
            return old
        new = Item(**{**asdict(old), **changes, "version": old.version + 1, "updated": _now()})
        new.validate()
        with self.conn:
            archived = asdict(old)
            archived["terms"] = json.dumps(old.terms, ensure_ascii=False)
            archived["archived"] = new.updated
            cols = [c for c in _COLUMNS if c != "version"] + ["version", "archived"]
            self.conn.execute(
                f"INSERT INTO item_versions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [archived[c] for c in cols],
            )
            self._write(new)
        return new

    def upsert(self, item: Item) -> tuple[Item, str]:
        """Add a new item or update an existing one from its content. Returns the item and one of
        "added", "updated" or "unchanged"."""
        old = self.get(item.id)
        if old is None:
            return self.add(item), "added"
        new = self.update(item.id, **item.content())
        if new.version != old.version:
            return new, "updated"
        if self.embedder is not None and self.vector(item.id) is None:
            # the item may have been stored before an embedder was available
            with self.conn:
                self._write(new)
        return new, "unchanged"

    def get(self, item_id: str) -> Item | None:
        row = self.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return _row_to_item(row) if row else None

    def history(self, item_id: str) -> list[Item]:
        """Previous versions, oldest first (the current version is not included)."""
        rows = self.conn.execute("SELECT * FROM item_versions WHERE id = ? ORDER BY version", (item_id,)).fetchall()
        return [_row_to_item(r) for r in rows]

    def list(self, db_id: str, kind: str | None = None, status: str | None = None) -> list[Item]:
        sql, args = "SELECT * FROM items WHERE db_id = ?", [db_id]
        if kind is not None:
            sql, args = sql + " AND kind = ?", args + [kind]
        if status is not None:
            sql, args = sql + " AND status = ?", args + [status]
        return [_row_to_item(r) for r in self.conn.execute(sql + " ORDER BY id", args).fetchall()]

    def record_use(self, ids: list[str]) -> None:
        with self.conn:
            self.conn.executemany("UPDATE items SET uses = uses + 1 WHERE id = ?", [(i,) for i in ids])

    def vector(self, item_id: str) -> np.ndarray | None:
        row = self.conn.execute("SELECT vector FROM item_vectors WHERE id = ?", (item_id,)).fetchone()
        return np.frombuffer(row["vector"], dtype=np.float32) if row else None
