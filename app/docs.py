"""Flat-RAG baseline: BIRD's database_description CSVs, one chunk per column, retrieved by the same hybrid search.

This is the "plain documentation" control for typed memory. It sees the data dictionary but has no item types, no
term matching, no knowledge from the seed split and no examples, so memory beating it means the typing and curation
help, not only the extra text.
"""
import csv
import io
from pathlib import Path

from app import bird
from app.config import settings
from app.memory import Item, search_text
from app.retrieve import DEFAULT_K
from app.textindex import Index


def description_dir(db_id: str, bird_dir: Path | None = None) -> Path:
    return bird.db_path(db_id, bird_dir).parent / "database_description"


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")  # several BIRD description files are not UTF-8


def _clean(value: str | None) -> str:
    return " ".join((value or "").split())


def load_chunks(db_id: str, bird_dir: Path | None = None) -> list[Item]:
    """One item per described column, ordered by table then file order. Empty when the folder is missing."""
    folder = description_dir(db_id, bird_dir)
    if not folder.is_dir():
        return []
    chunks = []
    for path in sorted(folder.glob("*.csv")):
        table = path.stem
        reader = csv.DictReader(io.StringIO(_read_text(path)))
        reader.fieldnames = [_clean(f).lower() for f in reader.fieldnames or []]
        for row in reader:
            column = _clean(row.get("original_column_name"))
            if not column:
                continue
            parts = [f"Column {column} of table {table}."]
            if (name := _clean(row.get("column_name"))) and name.lower() != column.lower():
                parts.append(f"Name: {name}.")
            for label, key in (("Description", "column_description"), ("Format", "data_format"),
                               ("Values", "value_description")):
                if text := _clean(row.get(key)):
                    parts.append(f"{label}: {text}")
            chunks.append(Item(id=f"doc:{table}.{column}", db_id=db_id, kind="glossary", title=f"{table}.{column}",
                               body=" ".join(parts), provenance="docs"))
    return chunks


class Docs:
    """A search index over the description chunks of one database, built in memory."""

    def __init__(self, chunks: list[Item], embedder=None):
        self.chunks = chunks
        vectors = None
        if embedder is not None and chunks:
            vectors = list(embedder.embed_passages([search_text(c) for c in chunks]))
        self.index = Index(chunks, vectors, embedder)

    @classmethod
    def load(cls, db_id: str, embedder=None, bird_dir: Path | None = None) -> "Docs":
        return cls(load_chunks(db_id, bird_dir or settings.bird_dir), embedder)

    def retrieve(self, question: str, k: int = DEFAULT_K) -> list[Item]:
        return [h.item for h in self.index.search(question, k)]
