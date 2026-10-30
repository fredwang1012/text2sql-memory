"""Execution-accuracy comparison on the EVAL split of a BIRD database, with the question's evidence hidden.

Modes: schema (schema only), docs (schema plus the closest data-dictionary chunks: the flat-RAG baseline), memory
(schema plus retrieved rules) and, with --oracle, the BIRD evidence handed to the model as a single rule. The oracle row
is an upper-bound reference, not a fair system: it reads eval evidence. Each pair of fair modes is compared with an
exact McNemar test and a bootstrap interval on the EX gap.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from app import bird
from app.config import settings
from app.docs import Docs, load_chunks
from app.embedder import get_embedder
from app.execute import Result as ExecResult
from app.execute import run, same_result
from app.llm import LLM, Draft, LLMUnavailable, make_llm, unavailable_hint
from app.memory import KINDS, Item, Store
from app.pipeline import Result, answer
from app.retrieve import DEFAULT_K
from app.sanity import coverage, referenced_columns
from app.schema import introspect
from app.seed import knowledge_path, load_items
from app.stats import compare
from app.textindex import Index

ORACLE = "oracle"
MODES = ("schema", "docs", "memory")
# (A, B) pairs reported when both ran: does A beat B on the same questions
COMPARISONS = (("memory", "schema"), ("docs", "schema"), ("memory", "docs"))
MAX_ROWS = 10**7  # effectively uncapped when scoring: a capped gold and prediction would compare different row subsets
MAX_FAILURES = 5  # consecutive per-question failures before the run is abandoned


class TooManyFailures(RuntimeError):
    """MAX_FAILURES questions in a row failed; partial holds the mode's results so far."""

    def __init__(self, partial: dict):
        super().__init__(f"{MAX_FAILURES} questions in a row failed")
        self.partial = partial


class _OracleLLM:
    """Wraps an LLM so every call sees the question's evidence as one convention item. No store is involved."""

    def __init__(self, llm: LLM, db_id: str):
        self.llm, self.db_id = llm, db_id
        self.evidence = ""

    def _items(self) -> list[Item]:
        item = Item(id=ORACLE, db_id=self.db_id, kind="convention", title="BIRD evidence", body=self.evidence,
                    provenance="data")
        return [item] if self.evidence else []

    def generate(self, question: str, schema_text: str, items: list[Item]) -> Draft:
        return self.llm.generate(question, schema_text, self._items())

    def repair(self, question: str, schema_text: str, items: list[Item], sql: str, error: str) -> Draft:
        return self.llm.repair(question, schema_text, self._items(), sql, error)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _summary(rows: list[dict]) -> dict:
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"] or "unknown"].append(r)
    return {
        "n": len(rows),
        "correct": sum(r["correct"] for r in rows),
        "n_gold_error": sum(r["gold_error"] is not None for r in rows),
        "n_truncated": sum(r["truncated"] for r in rows),
        "ex": _mean([r["correct"] for r in rows]),
        "error_rate": _mean([r["error"] is not None for r in rows]),
        "mean_rules_used": _mean([len(r["items_used"]) for r in rows]),
        "n_warned": sum(bool(r.get("warnings")) for r in rows),
        # share of the gold query's (non-key) columns named somewhere in the items shown; None for modes without items
        "mean_coverage": _mean(cov) if (cov := [r["coverage"] for r in rows if r.get("coverage") is not None]) else None,
        "by_difficulty": {d: {"n": len(v), "ex": _mean([r["correct"] for r in v])} for d, v in sorted(by_diff.items())},
    }


def eval_questions(db_id: str, limit: int | None = None) -> list[bird.Question]:
    _, eval_qs = bird.split(bird.load_questions(db_id))
    eval_qs = sorted(eval_qs, key=lambda q: q.question_id)
    return eval_qs[:limit] if limit else eval_qs


def open_eval_store(db_id: str, tmp_dir: Path, embedder=None, exclude_kinds: tuple[str, ...] = ()) -> tuple[Store, str]:
    """A throwaway store seeded from knowledge/<db_id>.yaml, so a run is reproducible from the committed file and
    never touches the real store or its use counts. Items of the excluded kinds are left out (the ablation)."""
    store = Store(tmp_dir / "memory-eval.db", embedder=embedder)
    items = [i for i in load_items(knowledge_path(db_id)) if i.kind not in exclude_kinds]
    for item in items:
        store.upsert(item)
    note = f", without {', '.join(exclude_kinds)}" if exclude_kinds else ""
    return store, f"seeded from {knowledge_path(db_id)} ({len(items)} items{note})"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                             cwd=Path(__file__).resolve().parents[1])
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def run_metadata(store: Store | None, db_id: str, docs: Docs | None = None, exclude_kinds: tuple[str, ...] = ()) -> dict:
    """What the results depend on: the retrieval mode actually in use, the knowledge file, any ablation and the code."""
    meta: dict = {"k": DEFAULT_K, "git_commit": _git_commit(), "excluded_kinds": list(exclude_kinds)}
    path = knowledge_path(db_id)
    meta["knowledge_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    meta["retrieval_mode"] = None
    if store is not None:
        items = store.list(db_id)
        meta["retrieval_mode"] = Index(items, [store.vector(i.id) for i in items], store.embedder).mode
    if docs is not None:
        meta["docs_chunks"] = len(docs.chunks)
        meta["docs_retrieval_mode"] = docs.index.mode
    return meta


def score(res: Result, q: bird.Question, db_file: Path) -> tuple[bool, ExecResult, ExecResult]:
    """Whether the answer matches the gold result, plus both results, each run without the row cap."""
    gold = run(db_file, q.sql, max_rows=MAX_ROWS)
    pred = ExecResult(rows=res.rows, columns=res.columns, error=res.error)
    if res.truncated and res.sql:  # the pipeline's own row cap is lower; score the full result
        pred = run(db_file, res.sql, max_rows=MAX_ROWS)
    return same_result(pred, gold), pred, gold


def run_mode(mode: str, questions: list[bird.Question], db_id: str, llm: LLM, store: Store | None = None,
             db_file: Path | None = None, progress: Callable[[dict], None] | None = None,
             docs: Docs | None = None) -> dict:
    """Run one mode over the questions. Only the question text reaches the pipeline, except in oracle mode.
    progress gets the partial results after every question; TooManyFailures is raised after MAX_FAILURES
    consecutive failures."""
    db_file = db_file or bird.db_path(db_id)
    oracle = _OracleLLM(llm, db_id) if mode == ORACLE else None
    tables = introspect(db_file) if mode in ("docs", "memory") else []
    rows = []
    streak = 0
    for q in questions:
        try:
            if oracle:
                oracle.evidence = q.evidence
                res = answer(q.question, db_id, "schema", llm=oracle, db_file=db_file)
            else:
                res = answer(q.question, db_id, mode, llm=llm, store=store if mode == "memory" else None,
                             db_file=db_file, docs=docs)
            streak = 0
        except LLMUnavailable:
            raise  # no credentials at all: nothing else will work either
        except Exception as e:  # rate limit, overload, timeout: lose one question, not the run
            res = Result(error=f"{type(e).__name__}: {e}")
            streak += 1
        correct, pred, gold = score(res, q, db_file)
        rows.append({
            "question_id": q.question_id,
            "difficulty": q.difficulty,
            "correct": correct,
            "error": res.error,
            "gold_error": gold.error,
            "truncated": pred.truncated or gold.truncated,
            "sql": res.sql,
            "items_used": res.items_used,
            "items_shown": [i.id for i in res.shown],
            "warnings": res.warnings,
            "coverage": coverage(referenced_columns(q.sql, tables), res.shown) if tables else None,
        })
        partial = {"summary": _summary(rows), "questions": rows}
        if progress:
            progress(partial)
        if streak >= MAX_FAILURES:
            raise TooManyFailures(partial)
    return {"summary": _summary(rows), "questions": rows}


def comparisons(results: dict[str, dict]) -> dict[str, dict]:
    """Paired tests for every pair in COMPARISONS that ran, keyed "A vs B"."""
    return {f"{a} vs {b}": compare(results[a]["questions"], results[b]["questions"])
            for a, b in COMPARISONS if a in results and b in results}


def format_report(results: dict[str, dict]) -> str:
    lines = [f"{'mode':<8} {'n':>4} {'EX':>7} {'errors':>7} {'rules/q':>8} {'coverage':>9} {'warned':>7}"]
    for mode, r in results.items():
        s = r["summary"]
        label = "oracle*" if mode == ORACLE else mode
        rules = "n/a" if mode == ORACLE else f"{s['mean_rules_used']:.2f}"  # its one rule is not a store item
        cov = "n/a" if s.get("mean_coverage") is None else f"{s['mean_coverage']:.1%}"
        lines.append(f"{label:<8} {s['n']:>4} {s['ex']:>7.1%} {s['error_rate']:>7.1%} {rules:>8} {cov:>9} "
                     f"{s.get('n_warned', 0):>7}")
    gold_bad = max((r["summary"]["n_gold_error"] for r in results.values()), default=0)
    footer = f"Gold queries that failed to run (counted as wrong): {gold_bad} per mode."
    cut = max((r["summary"]["n_truncated"] for r in results.values()), default=0)
    if cut:
        footer += f"\nResults cut off by the row cap (counted as wrong): {cut} in the worst mode."
    diffs = sorted({d for r in results.values() for d in r["summary"]["by_difficulty"]})
    if diffs:
        lines.append("")
        lines.append(f"{'EX by difficulty':<18}" + "".join(f"{'oracle*' if m == ORACLE else m:>9}" for m in results))
        for d in diffs:
            cells = []
            for r in results.values():
                b = r["summary"]["by_difficulty"].get(d)
                cells.append(f"{b['ex']:>9.1%}" if b else f"{'-':>9}")
            lines.append(f"{d:<18}" + "".join(cells))
    tests = comparisons(results)
    if tests:
        lines.append("")
        lines.append(f"{'paired test':<18} {'gap':>7} {'95% CI':>17} {'wins':>6} {'losses':>7} {'p':>7}")
        for name, t in tests.items():
            ci = f"[{t['ci95'][0]:+.1%}, {t['ci95'][1]:+.1%}]"
            lines.append(f"{name:<18} {t['ex_diff']:>+7.1%} {ci:>17} {t['only_a']:>6} {t['only_b']:>7} "
                         f"{t['p_mcnemar']:>7.3f}")
        lines.append("wins/losses: questions only the first/second mode got right. p: exact McNemar, two-sided.")
    footer += ("\ncoverage: share of the gold query's columns (keys left out) named in the items shown. "
               "warned: queries that ran but returned no rows, an all-NULL column or mostly duplicate rows.")
    lines.append("\n" + footer)
    if ORACLE in results:
        lines.append("\n* oracle reads the BIRD evidence of the eval questions: an upper-bound reference, not a fair system.")
    return "\n".join(lines)


def _positive(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return n


def _write_json(path: Path, args, questions: list[bird.Question], modes: list[str], results: dict,
                meta: dict | None = None, aborted: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "db_id": args.db,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "effort": settings.llm_effort if settings.llm_provider == "anthropic" else None,
        "base_url": settings.llm_base_url if settings.llm_provider == "openai" else None,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modes_requested": modes,
        "oracle": args.oracle,
        "limit": args.limit,
        "questions_per_mode": len(questions),
        "eval_question_ids": [q.question_id for q in questions],
        "split_file": str(bird.split_path(args.db)),
        "run": meta,
        "aborted": aborted,
        "results": results,
        "comparisons": comparisons(results),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None, llm: LLM | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.evaluate", description=__doc__)
    p.add_argument("--db", default=settings.db_id)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--oracle", action="store_true", help="also run the evidence upper-bound reference")
    p.add_argument("--exclude-kinds", nargs="+", choices=KINDS, default=[], metavar="KIND",
                   help=f"ablation: leave these item kinds out of the memory store ({', '.join(KINDS)})")
    p.add_argument("--limit", type=_positive, default=None, help="only the first N eval questions")
    p.add_argument("--json", type=Path, default=None, help="write the summary and per-question detail here")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = p.parse_args(argv)

    if bird.data_missing(args.db):
        print(bird.MISSING_MESSAGE, file=sys.stderr)
        return 1
    questions = eval_questions(args.db, args.limit)
    if not questions:
        print(f"No eval questions for {args.db}. Run scripts/get_bird.py first.", file=sys.stderr)
        return 1
    modes = list(dict.fromkeys(args.modes)) + ([ORACLE] if args.oracle else [])
    exclude = tuple(dict.fromkeys(args.exclude_kinds))
    chunks = load_chunks(args.db) if "docs" in modes else []
    if "docs" in modes and not chunks:
        print(f"No data-dictionary files for {args.db} (database_description/*.csv), so docs mode cannot run. "
              "Rerun scripts/get_bird.py or leave docs out of --modes.", file=sys.stderr)
        return 1
    calls = len(questions) * len(modes)
    print(f"{len(questions)} questions x {len(modes)} modes ({', '.join(modes)}) = {calls} to {2 * calls} LLM calls "
          f"(one call each, a second when the SQL fails to run).")
    where = f"effort: {settings.llm_effort}" if settings.llm_provider == "anthropic" else f"at {settings.llm_base_url}"
    print(f"Model: {settings.llm_model} via {settings.llm_provider}, {where}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 1

    llm = llm or make_llm()
    results: dict[str, dict] = {}
    store = None
    docs = None
    meta: dict = {}
    aborted = None
    status = 0
    mode = ""
    tmp = tempfile.TemporaryDirectory()

    def save() -> None:
        if args.json:
            _write_json(args.json, args, questions, modes, results, meta, aborted)

    def progress(partial: dict) -> None:
        results[mode] = partial
        save()  # a crash or Ctrl-C keeps everything finished so far

    try:
        if "memory" in modes:
            store, source = open_eval_store(args.db, Path(tmp.name), get_embedder(), exclude)
            print(f"Memory store for this run: {source}. The real store is not modified.")
        if "docs" in modes:
            docs = Docs(chunks, get_embedder())  # after the confirmation: embedding the chunks can take a while
            print(f"Docs baseline: {len(docs.chunks)} column descriptions ({docs.index.mode} retrieval).")
        meta = run_metadata(store, args.db, docs, exclude)
        for mode in modes:
            print(f"Running {mode}...", flush=True)
            results[mode] = run_mode(mode, questions, args.db, llm, store, progress=progress, docs=docs)
            save()
    except LLMUnavailable:
        print(unavailable_hint(), file=sys.stderr)
        return 1
    except TooManyFailures as e:
        results[mode] = e.partial
        aborted = f"{mode}: {e}"
        status = 1
        print(f"\nAborted: {aborted}. Partial results follow.", file=sys.stderr)
        save()
    finally:
        if store is not None:
            store.close()
        tmp.cleanup()

    print()
    print(format_report(results))
    if args.json:
        print(f"\nWrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
