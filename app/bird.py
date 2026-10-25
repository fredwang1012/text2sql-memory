"""BIRD dev loader: questions per database, SQLite paths, and the deterministic seed/eval split."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


@dataclass(frozen=True)
class Question:
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str


def load_questions(db_id: str, bird_dir: Path | None = None) -> list[Question]:
    path = (bird_dir or settings.bird_dir) / "dev.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            question_id=int(r["question_id"]),
            db_id=r["db_id"],
            question=r["question"],
            evidence=r.get("evidence") or "",
            sql=r.get("SQL") or r.get("sql") or "",
            difficulty=r.get("difficulty") or "",
        )
        for r in rows
        if r["db_id"] == db_id
    ]


def db_path(db_id: str, bird_dir: Path | None = None) -> Path:
    return (bird_dir or settings.bird_dir) / "dev_databases" / db_id / f"{db_id}.sqlite"


MISSING_MESSAGE = "BIRD data not found — run `py -3.12 -m uv run python scripts/get_bird.py`"


def data_missing(db_id: str) -> bool:
    """True when dev.json or the database file has not been downloaded yet."""
    return not (settings.bird_dir / "dev.json").is_file() or not db_path(db_id).is_file()


def safe_join(root: Path, rel: str) -> Path:
    """Join an archive member name under root, rejecting traversal, absolute paths and drive letters."""
    name = rel.replace("\\", "/")
    parts = [p for p in name.split("/") if p]
    if name.startswith("/") or any(p == ".." or ":" in p for p in parts):
        raise ValueError(f"unsafe archive member: {rel}")
    root = root.resolve()
    target = root.joinpath(*parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"unsafe archive member: {rel}")
    return target


def split_path(db_id: str) -> Path:
    return settings.data_dir / f"split-{db_id}.json"


def _bucket(question_id: int, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split(
    questions: list[Question], seed_fraction: float = 0.3, seed: int = 13, record_to: Path | None = None
) -> tuple[list[Question], list[Question]]:
    """Assign each question by a hash of its id, so the split does not depend on list order or on other questions.
    When record_to is given, the ids of both halves are written there."""
    seed_qs = [q for q in questions if _bucket(q.question_id, seed) < seed_fraction]
    eval_qs = [q for q in questions if _bucket(q.question_id, seed) >= seed_fraction]
    if record_to is not None:
        record_to.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "seed": seed,
            "seed_fraction": seed_fraction,
            "seed_ids": sorted(q.question_id for q in seed_qs),
            "eval_ids": sorted(q.question_id for q in eval_qs),
        }
        record_to.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return seed_qs, eval_qs
