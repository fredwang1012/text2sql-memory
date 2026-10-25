"""YAML import/export for the knowledge store. knowledge/<db_id>.yaml is the human-edited source of truth;
importing upserts items by id, so an edited item gets a new version and an unchanged one is left alone."""
import argparse
from pathlib import Path

import yaml

from app.config import ROOT, settings
from app.embedder import get_embedder
from app.memory import CONTENT_FIELDS, Item, Store

KNOWLEDGE_DIR = ROOT / "knowledge"


def knowledge_path(db_id: str) -> Path:
    return KNOWLEDGE_DIR / f"{db_id}.yaml"


def load_items(path: Path) -> list[Item]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    db_id = doc.get("db_id")
    items, seen = [], set()
    for raw in doc.get("items") or []:
        raw = dict(raw)
        raw.setdefault("db_id", db_id)
        unknown = set(raw) - set(CONTENT_FIELDS) - {"id"}
        if unknown:
            raise ValueError(f"{raw.get('id')}: unknown fields {sorted(unknown)}")
        if raw.get("id") in seen:
            raise ValueError(f"duplicate id {raw.get('id')}")
        seen.add(raw.get("id"))
        raw["terms"] = [str(t) for t in raw.get("terms") or []]
        for key in ("title", "body", "sql"):
            if isinstance(raw.get(key), str):
                raw[key] = raw[key].strip()
        item = Item(**raw)
        item.validate()
        items.append(item)
    return items


def import_yaml(store: Store, path: Path) -> dict[str, int]:
    """Upsert every item in the file. Returns counts of added, updated and unchanged items."""
    counts = {"added": 0, "updated": 0, "unchanged": 0}
    for item in load_items(path):
        _, outcome = store.upsert(item)
        counts[outcome] += 1
    return counts


class _Dumper(yaml.SafeDumper):
    pass


def _str(dumper: yaml.SafeDumper, s: str):
    return dumper.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None)


_Dumper.add_representer(str, _str)


def export_yaml(store: Store, db_id: str, path: Path) -> int:
    """Write the current version of every item for db_id. Returns the number of items written."""
    items = []
    for item in store.list(db_id):
        d = {"id": item.id}
        d.update({f: getattr(item, f) for f in CONTENT_FIELDS if f != "db_id"})
        if d["sql"] is None:
            del d["sql"]
        if d["scope"] == "global":
            del d["scope"]
        items.append(d)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump({"db_id": db_id, "items": items}, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(text, encoding="utf-8")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import or export knowledge YAML")
    parser.add_argument("action", choices=["import", "export"])
    parser.add_argument("--db-id", default=settings.db_id)
    parser.add_argument("--file", type=Path, default=None)
    args = parser.parse_args()
    path = args.file or knowledge_path(args.db_id)
    with Store(settings.memory_db, embedder=get_embedder()) as store:
        if args.action == "import":
            print(import_yaml(store, path))
        else:
            print(f"wrote {export_yaml(store, args.db_id, path)} items to {path}")


if __name__ == "__main__":
    main()
