"""Command line: ask a question, import the seed YAML, list the stored items."""
import argparse
import sys
from pathlib import Path

from app import bird
from app.config import settings
from app.embedder import get_embedder
from app.llm import LLMTransient, LLMUnavailable
from app.memory import Store
from app.pipeline import MODES, answer
from app.seed import import_yaml, knowledge_path

SHOW_ROWS = 10


def _ask(args: argparse.Namespace) -> int:
    if bird.data_missing(args.db_id):
        print(bird.MISSING_MESSAGE, file=sys.stderr)
        return 1
    if args.mode == "memory":
        with Store(settings.memory_db) as store:
            empty = not store.list(args.db_id)
        if empty:
            print(f"The memory store has no items for {args.db_id}: run `py -3.12 -m uv run python -m app.cli seed` first.",
                  file=sys.stderr)
    try:
        result = answer(args.question, args.db_id, args.mode)
    except LLMUnavailable:
        print("The Anthropic API is not available. Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)
        return 1
    except LLMTransient as e:
        print(f"{e}. Check the connection and try again.", file=sys.stderr)
        return 1
    print("SQL:")
    print(result.sql)
    if result.explanation:
        print("\nRules used:")
        for line in result.explanation:
            print(f"  - {line}")
    if result.error:
        print(f"\nError: {result.error}")
        return 1
    print(f"\nRows ({len(result.rows)}):")
    print(" | ".join(result.columns))
    for row in result.rows[:SHOW_ROWS]:
        print(" | ".join(str(v) for v in row))
    if len(result.rows) > SHOW_ROWS:
        print(f"... {len(result.rows) - SHOW_ROWS} more")
    return 0


def _seed(args: argparse.Namespace) -> int:
    path = args.file or knowledge_path(args.db_id)
    with Store(settings.memory_db, embedder=get_embedder()) as store:
        print(import_yaml(store, path))
    return 0


def _items(args: argparse.Namespace) -> int:
    with Store(settings.memory_db) as store:
        for it in store.list(args.db_id):
            print(f"{it.id}\t{it.kind}\t{it.status}\tuses={it.uses}\t{it.title}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    parser.add_argument("--db-id", default=settings.db_id)
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer a question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default="memory")
    ask.set_defaults(func=_ask)
    seed = sub.add_parser("seed", help="import the knowledge YAML")
    seed.add_argument("--file", type=Path, default=None)
    seed.set_defaults(func=_seed)
    sub.add_parser("items", help="list stored items").set_defaults(func=_items)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
