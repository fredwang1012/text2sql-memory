"""Command line: ask a question, give feedback on the answer, review the rules learned from it, import the seed YAML
and list the stored items."""
import argparse
import sys
from pathlib import Path

from app import bird
from app.config import settings
from app import feedback
from app.embedder import get_embedder
from app.llm import LLMRefusal, LLMTransient, LLMUnavailable, make_llm, unavailable_hint
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
        print(unavailable_hint(), file=sys.stderr)
        return 1
    except LLMTransient as e:
        print(f"{e}. Check the connection and try again.", file=sys.stderr)
        return 1
    with Store(settings.memory_db) as store:
        answer_id = feedback.record(store, args.db_id, args.question, args.mode, result).id
    print("SQL:")
    print(result.sql)
    if result.explanation:
        print("\nRules used:")
        for line in result.explanation:
            print(f"  - {line}")
    if result.error:
        print(f"\nError: {result.error}")
        return 1
    for w in result.warnings:
        print(f"\nWarning: {w}")
    print(f"\nRows ({len(result.rows)}):")
    print(" | ".join(result.columns))
    for row in result.rows[:SHOW_ROWS]:
        print(" | ".join(str(v) for v in row))
    if len(result.rows) > SHOW_ROWS:
        print(f"... {len(result.rows) - SHOW_ROWS} more")
    print(f"\nAnswer #{answer_id}. Was it right? `python -m app.cli good {answer_id}` or "
          f"`python -m app.cli wrong {answer_id} \"what is wrong\"`")
    return 0


def _good(args: argparse.Namespace) -> int:
    with Store(settings.memory_db) as store:
        try:
            promoted = feedback.mark_good(store, args.answer_id)
        except KeyError:
            print(f"No answer #{args.answer_id}.", file=sys.stderr)
            return 1
    print(f"Recorded answer #{args.answer_id} as good.")
    for item in promoted:
        print(f"  promoted to reviewed: {item.id} {item.title}")
    return 0


def _wrong(args: argparse.Namespace) -> int:
    with Store(settings.memory_db, embedder=get_embedder()) as store:
        answer = store.answer(args.answer_id)
        if answer is None:
            print(f"No answer #{args.answer_id}.", file=sys.stderr)
            return 1
        try:
            added = feedback.mark_wrong(store, args.answer_id, args.correction, make_llm(), bird.db_path(answer.db_id),
                                        fixed_sql=args.sql)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
        except LLMUnavailable:
            print(unavailable_hint(), file=sys.stderr)
            return 1
        except (LLMTransient, LLMRefusal) as e:
            print(f"Could not distill the correction: {e}. Nothing was recorded; try again.", file=sys.stderr)
            return 1
    print(f"Recorded answer #{args.answer_id} as wrong. New draft rules:" if added else
          f"Recorded answer #{args.answer_id} as wrong. No new rule came out of the correction.")
    for item in added:
        print(f"  {item.id}\t{item.kind}\t{item.title}: {item.body}")
    if added:
        print("Drafts are used right away, ranked below reviewed rules. `python -m app.cli review` lists them.")
    return 0


def _review(args: argparse.Namespace) -> int:
    with Store(settings.memory_db) as store:
        queue = feedback.review_queue(store, args.db_id)
    if not queue:
        print("No draft rules to review.")
        return 0
    for item, counts in queue:
        print(f"{item.id}\t{item.kind}\tgood={counts['good']} wrong={counts['wrong']}\t{item.title}")
        print(f"    {item.body}" + (f"\n    SQL: {item.sql}" if item.sql else ""))
    print("\n`approve ID` marks a rule reviewed, `certify ID` certifies it, `reject ID` removes it (history is kept).")
    return 0


def _set_status(status: str):
    def handler(args: argparse.Namespace) -> int:
        with Store(settings.memory_db) as store:
            try:
                if status == "rejected":
                    store.archive(args.item_id)
                    print(f"Removed {args.item_id}; its history is kept.")
                else:
                    item = feedback.set_status(store, args.item_id, status)
                    print(f"{item.id} is now {item.status} (version {item.version}).")
            except KeyError:
                print(f"No item {args.item_id}.", file=sys.stderr)
                return 1
        return 0
    return handler


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
    good = sub.add_parser("good", help="mark an answer as right")
    good.add_argument("answer_id", type=int)
    good.set_defaults(func=_good)
    wrong = sub.add_parser("wrong", help="mark an answer as wrong and say why; the correction becomes draft rules")
    wrong.add_argument("answer_id", type=int)
    wrong.add_argument("correction", nargs="?", default="", help="what is wrong, in business terms")
    wrong.add_argument("--sql", default=None, help="the right query, if you have it (kept as a draft example)")
    wrong.set_defaults(func=_wrong)
    sub.add_parser("review", help="list draft rules waiting for review").set_defaults(func=_review)
    for name, status, text in (("approve", "reviewed", "mark a rule reviewed"), ("certify", "certified", "certify a rule"),
                               ("reject", "rejected", "remove a rule, keeping its history")):
        cmd = sub.add_parser(name, help=text)
        cmd.add_argument("item_id")
        cmd.set_defaults(func=_set_status(status))
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
