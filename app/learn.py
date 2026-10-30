"""Simulated learning on BIRD: does the memory get better as users correct it?

The eval split is cut again, by a hash of the question id, into a stream and a held-out test half. The stream is
answered in memory mode, one question at a time. A wrong answer is corrected by a simulated user, a model that sees
the gold query but must explain the mistake in business terms without revealing SQL, and the correction is distilled
into draft rules exactly as a real user's would be (app.feedback). A right answer confirms the rules it cited.
Every --checkpoint-every stream questions, the test half is answered with the store as it stands. The test half never
feeds the store, so its accuracy over checkpoints is the learning curve. The evidence of every question stays hidden.
"""
import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app import bird, feedback
from app.config import settings
from app.embedder import get_embedder
from app.evaluate import (MAX_FAILURES, TooManyFailures, _git_commit, _positive, eval_questions, open_eval_store,
                          run_mode, score)
from app.execute import Result as ExecResult
from app.llm import LLMUnavailable, make_llm, unavailable_hint
from app.memory import Store
from app.pipeline import Result, answer
from app.schema import schema_text
from app.stats import compare

STREAM_SEED = 29  # differs from the seed/eval split's seed, so the two cuts are independent
PREVIEW_ROWS = 5

# SQL in a simulated correction would hand the answer to the store; such corrections are recorded but not learned from
_SQL = re.compile(r"\bselect\b[\s\S]+?\bfrom\b|\bjoin\s+\w+(\s+as\s+\w+)?\s+on\b|\bgroup\s+by\b|\border\s+by\b", re.I)


def split_stream(questions: list[bird.Question], fraction: float = 0.5,
                 seed: int = STREAM_SEED) -> tuple[list[bird.Question], list[bird.Question]]:
    stream = [q for q in questions if bird._bucket(q.question_id, seed) < fraction]
    test = [q for q in questions if bird._bucket(q.question_id, seed) >= fraction]
    return stream, test


def reveals_sql(text: str, gold_sql: str) -> bool:
    gold = " ".join(gold_sql.lower().split())
    return bool(_SQL.search(text)) or (len(gold) > 20 and gold in " ".join(text.lower().split()))


def preview(result: ExecResult) -> str:
    if result.error:
        return f"(error: {result.error})"
    lines = [" | ".join(result.columns)] + [" | ".join(map(str, r)) for r in result.rows[:PREVIEW_ROWS]]
    if len(result.rows) > PREVIEW_ROWS:
        lines.append(f"... {len(result.rows)} rows in all")
    return "\n".join(lines) if result.rows else "(no rows)"


def store_counts(store: Store, db_id: str) -> dict[str, int]:
    counts = Counter(i.status for i in store.list(db_id))
    return {s: counts.get(s, 0) for s in ("draft", "reviewed", "certified")}


def checkpoint(after: int, test: list[bird.Question], db_id: str, llm, store: Store, db_file: Path) -> dict:
    out = run_mode("memory", test, db_id, llm, store, db_file)
    return {"after": after, "ex": out["summary"]["ex"], "items": store_counts(store, db_id),
            "questions": [{"question_id": r["question_id"], "correct": r["correct"]} for r in out["questions"]]}


def step(q: bird.Question, db_id: str, llm, learner, store: Store, db_file: Path, schema: str) -> dict:
    """Answer one stream question, then learn from the outcome. Returns what happened."""
    failed = False
    try:
        res = answer(q.question, db_id, "memory", llm=llm, store=store, db_file=db_file)
    except LLMUnavailable:
        raise
    except Exception as e:  # rate limit, overload, timeout: lose one question, not the run
        res, failed = Result(error=f"{type(e).__name__}: {e}"), True
    correct, pred, gold = score(res, q, db_file)
    event = {"question_id": q.question_id, "correct": correct, "error": res.error, "items_used": res.items_used,
             "added": [], "promoted": [], "correction": None, "revealed_sql": False, "feedback_error": None,
             "call_failed": failed}
    record = feedback.record(store, db_id, q.question, "memory", res)
    try:
        if correct:
            event["promoted"] = [i.id for i in feedback.mark_good(store, record.id)]
        elif gold.error is None:  # a broken gold query cannot be explained
            text = learner.correct(q.question, schema, res.sql, preview(pred), q.sql, preview(gold))
            event["correction"] = text
            if reveals_sql(text, q.sql):
                event["revealed_sql"] = True
                store.judge(record.id, "wrong", text)
            else:
                event["added"] = [i.id for i in feedback.mark_wrong(store, record.id, text, learner, db_file)]
    except LLMUnavailable:
        raise
    except Exception as e:
        event["feedback_error"] = f"{type(e).__name__}: {e}"
    return event


def run_loop(stream: list[bird.Question], test: list[bird.Question], db_id: str, llm, store: Store,
             learner=None, db_file: Path | None = None, every: int = 10, progress=None) -> dict:
    """The whole simulation. progress gets the partial state after every step and checkpoint."""
    learner = learner or llm
    db_file = db_file or bird.db_path(db_id)
    schema = schema_text(db_file)
    state: dict = {"events": [], "checkpoints": [], "baseline_schema": None}

    def emit():
        if progress:
            progress(state)

    state["baseline_schema"] = run_mode("schema", test, db_id, llm, None, db_file)["summary"]["ex"]
    state["checkpoints"].append(checkpoint(0, test, db_id, llm, store, db_file))
    emit()
    streak = 0
    for n, q in enumerate(stream, 1):
        event = step(q, db_id, llm, learner, store, db_file, schema)
        state["events"].append(event)
        streak = streak + 1 if event["call_failed"] or event["feedback_error"] else 0
        emit()
        if streak >= MAX_FAILURES:
            raise TooManyFailures(state)
        if n % every == 0 or n == len(stream):
            state["checkpoints"].append(checkpoint(n, test, db_id, llm, store, db_file))
            emit()
    return state


def summarize(state: dict) -> dict:
    events, cps = state["events"], state["checkpoints"]
    out = {
        "stream_n": len(events),
        "stream_ex": sum(e["correct"] for e in events) / len(events) if events else 0.0,
        "corrections": sum(e["correction"] is not None for e in events),
        "revealed_sql": sum(e["revealed_sql"] for e in events),
        "rules_added": sum(len(e["added"]) for e in events),
        "promoted": sum(len(e["promoted"]) for e in events),
        "feedback_errors": sum(e["feedback_error"] is not None for e in events),
    }
    if len(cps) >= 2:
        out["held_out"] = compare(cps[-1]["questions"], cps[0]["questions"])
    return out


def format_report(state: dict) -> str:
    s = summarize(state)
    lines = [f"Stream: {s['stream_n']} questions, {s['stream_ex']:.1%} right as they came. "
             f"{s['corrections']} corrections ({s['revealed_sql']} not learned from: they revealed SQL), "
             f"{s['rules_added']} rules added, {s['promoted']} promoted to reviewed.", ""]
    lines.append(f"{'after':>5} {'test EX':>8} {'draft':>6} {'reviewed':>9} {'certified':>10} {'stream EX since last':>21}")
    prev = 0
    for cp in state["checkpoints"]:
        window = state["events"][prev:cp["after"]]
        win = f"{sum(e['correct'] for e in window) / len(window):.1%}" if window else "-"
        i = cp["items"]
        lines.append(f"{cp['after']:>5} {cp['ex']:>8.1%} {i['draft']:>6} {i['reviewed']:>9} {i['certified']:>10} {win:>21}")
        prev = cp["after"]
    if state.get("baseline_schema") is not None:
        lines.append(f"\nSchema only, on the same test questions: {state['baseline_schema']:.1%}")
    if "held_out" in s:
        t = s["held_out"]
        lines.append(f"Held-out change, last vs first checkpoint: {t['ex_diff']:+.1%} "
                     f"[{t['ci95'][0]:+.1%}, {t['ci95'][1]:+.1%}], {t['only_a']} gained, {t['only_b']} lost, "
                     f"McNemar p = {t['p_mcnemar']:.3f}")
    if s["feedback_errors"]:
        lines.append(f"Feedback steps that failed (the question still counts): {s['feedback_errors']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None, llm=None, learner=None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.learn", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=settings.db_id)
    p.add_argument("--start", choices=["seeded", "empty"], default="seeded",
                   help="seeded: begin from knowledge/<db>.yaml; empty: begin with no rules")
    p.add_argument("--stream-fraction", type=float, default=0.5, help="share of the eval split used as the stream")
    p.add_argument("--checkpoint-every", type=_positive, default=10, help="stream questions between test runs")
    p.add_argument("--limit", type=_positive, default=None, help="cap the stream and the test half at N each")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = p.parse_args(argv)
    if not 0 < args.stream_fraction < 1:
        p.error("--stream-fraction must be between 0 and 1")

    if bird.data_missing(args.db):
        print(bird.MISSING_MESSAGE, file=sys.stderr)
        return 1
    stream, test = split_stream(eval_questions(args.db), args.stream_fraction)
    if args.limit:
        stream, test = stream[:args.limit], test[:args.limit]
    if not stream or not test:
        print("The stream or the test half is empty; change --stream-fraction.", file=sys.stderr)
        return 1
    n_cp = (len(stream) + args.checkpoint_every - 1) // args.checkpoint_every + 1
    low = len(stream) + len(test) * (n_cp + 1)
    print(f"Stream {len(stream)} questions, test {len(test)}, {n_cp} checkpoints plus a schema-only baseline: "
          f"{low} to about {2 * low + 2 * len(stream)} model calls (wrong answers add a correction and a distillation).")
    print(f"Model: {settings.llm_model} via {settings.llm_provider}")
    if not args.yes and input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return 1

    llm = llm or make_llm()
    tmp = tempfile.TemporaryDirectory()
    meta = {"db_id": args.db, "start": args.start, "provider": settings.llm_provider, "model": settings.llm_model,
            "stream_fraction": args.stream_fraction, "stream_seed": STREAM_SEED, "checkpoint_every": args.checkpoint_every,
            "stream_ids": [q.question_id for q in stream], "test_ids": [q.question_id for q in test],
            "git_commit": _git_commit(), "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    state: dict = {}
    aborted = None

    def save(partial: dict) -> None:
        state.update(partial)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            payload = {"run": meta, "aborted": aborted, "summary": summarize(state) if "events" in state else None,
                       **state}
            args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    store = None
    status = 0
    try:
        if args.start == "seeded":
            store, source = open_eval_store(args.db, Path(tmp.name), get_embedder())
        else:
            store, source = Store(Path(tmp.name) / "memory-learn.db", embedder=get_embedder()), "empty"
        meta["store"] = source
        print(f"Store for this run: {source}. The real store is not modified.")
        run_loop(stream, test, args.db, llm, store, learner, every=args.checkpoint_every, progress=save)
    except LLMUnavailable:
        print(unavailable_hint(), file=sys.stderr)
        return 1
    except TooManyFailures as e:
        aborted = str(e)
        status = 1
        print(f"\nAborted: {aborted}. Partial results follow.", file=sys.stderr)
        save(e.partial if "events" in e.partial else {})
    finally:
        if store is not None:
            store.close()
        tmp.cleanup()
    if "events" in state:
        print()
        print(format_report(state))
    if args.json:
        print(f"\nWrote {args.json}")
    return status


if __name__ == "__main__":
    sys.exit(main())
