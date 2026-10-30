# text2sql-memory

A text-to-SQL layer that retrieves business logic, not just schema. The knowledge lives in a typed, versioned store of
glossary terms, metric definitions, join paths, conventions, verified examples and value hints. A question pulls the
few items that matter, and the model writes SQL with them in front of it. The goal is to beat two baselines on
execution accuracy on BIRD, with the BIRD evidence hidden: the schema alone, and flat RAG over the database's data
dictionary.

**Status:** Tier 1 (retrieval MVP) is built and tested. The comparison has not been run yet, so there are no numbers.

## Why

A schema says what a column is called, not what it means. In BIRD's `financial` database, `district.A11` is the
average salary, `A3` is the region and `A14` is the number of entrepreneurs per 1000 inhabitants. The schema does not
say so. The meanings come from the evidence hint attached to each question and from the data dictionary CSVs that ship
with the database (`database_description`). With both taken away, a model guesses from the syntax. This project keeps
the meaning in a store that can be reviewed and corrected, and retrieves it per question.

## Quick start

Prerequisites: Python 3.12 with [uv](https://docs.astral.sh/uv/) and a model to write the SQL: an Anthropic API key,
or a free option such as a local Ollama model (see [Choosing a model](#choosing-a-model)). An Anthropic key can go in
`.env` (`ANTHROPIC_API_KEY=...`) or in the environment. If neither is set, the SDK's own credential lookup is used,
such as an `ant auth login` profile.

```bash
py -3.12 -m uv sync                          # or plain: uv sync
# optional, for hybrid retrieval (a large PyTorch download): py -3.12 -m uv sync --extra embed
py -3.12 -m uv run python scripts/get_bird.py   # downloads BIRD dev into data/ (about 330 MB)
cp .env.example .env                         # then set ANTHROPIC_API_KEY (or export it)
py -3.12 -m uv run python -m app.cli seed    # import knowledge/financial.yaml into the store
py -3.12 -m uv run python -m app.cli ask "How many clients are in the Prague region?"
py -3.12 -m uv run python -m app.evaluate    # the comparison, see below
```

`ask` takes `--mode schema`, `--mode docs` or `--mode memory` (the default) and prints the SQL, the rules it used, any
warnings about the result and the first rows.
`python -m app.cli items` lists what is stored. Settings come from the environment or `.env`: `LLM_PROVIDER` (see
below), `LLM_MODEL` (default `claude-opus-5-5`), `LLM_EFFORT` (default `medium`, Anthropic only), `EMBED` (`auto` or
`off`), `DB_ID`, `DATA_DIR` and `MEMORY_DB`.

### Choosing a model

The comparison needs a model to write the SQL; it does not have to be Claude. `LLM_PROVIDER=anthropic` (the default)
uses the Anthropic API with structured outputs. `LLM_PROVIDER=openai` uses any OpenAI-compatible `/chat/completions`
endpoint in JSON mode, set by `LLM_BASE_URL`, `LLM_MODEL` and, when the endpoint needs one, `LLM_API_KEY`:

| Option | Cost | Notes |
| --- | --- | --- |
| Ollama, local | Free | No key and no rate limit. A coder model such as Qwen2.5-Coder; the `financial` schema prompt needs a context window of at least 8K tokens. |
| Gemini API | Free tier | OpenAI-compatible endpoint; limits depend on the model. |
| OpenRouter `:free` models | Free | Low daily request caps, so a full run can take more than a day. |
| Anthropic API | Paid | The default. |

Rate limits (HTTP 429) and server errors are retried up to 3 times with backoff, honouring `Retry-After`. Any model
works for the claim being tested, because every mode uses the same one; numbers from different models are not
comparable with each other. The model and provider are recorded in `results.json`.

## How it works

1. **Terms.** The question is first matched exactly against the `terms` of glossary, metric, value hint and convention
   items (lowercased, simple plurals folded, whole-word and whole-phrase). Exact matches come first, more certain and
   more specific ones before the rest.
2. **Retrieval.** The remaining slots (8 items in all) are filled by hybrid search: BM25Plus scaled by the highest
   score, plus cosine similarity from multilingual-e5 embeddings when the `embed` extra is installed (0.6 cosine, 0.4
   BM25). Without embeddings a hit must share a query term. The best verified example is included when one exists. The
   items are passed to the model along with the schema.
3. **Generation.** The model returns SQL and the ids of the items it relied on. Items it cites are counted as used.
4. **Execution.** The SQL runs read-only against the BIRD SQLite file. If it fails, the model gets the error and one
   chance to repair it.

Execution is read-only: a single `SELECT` or `WITH` statement, opened with `mode=ro`, a timeout and a row cap of
10,000 rows. The eval scores gold and predicted SQL without that cap, so a long result is compared in full.

## Knowledge items

| Type | What it holds |
| --- | --- |
| `glossary` | What a column, code or business term means ("`A11` is the average salary"). |
| `metric` | How a number is computed, including the exact SQL expression. |
| `join_path` | Which tables join and on which keys, including the non-obvious ones. |
| `convention` | Defaults and rules of thumb ("a question about the Prague branch filters the region column"). |
| `example` | A verified question and SQL pair. |
| `value_hint` | The exact spelling and meaning of stored values ("`'POPLATEK MESICNE'` means monthly issuance"). |

Every item has a status (`draft`, `reviewed`, `certified`), a version history and a provenance label. Items are
edited as YAML in `knowledge/<db>.yaml` and imported with `python -m app.cli seed`; an edited item becomes a new
version. `knowledge/financial.yaml` holds 40 hand-written items for the `financial` database.

Provenance names every source a claim rests on: `schema` (table and column definitions), `docs` (BIRD's
`database_description` files), `data` (checked by querying the database) and `seed:<id>` (the evidence of that
seed-split question).

## Honesty rule for the eval

The comparison is only meaningful if the knowledge does not contain the answers. So:

- Knowledge may be written from (a) the database schema and data, and (b) the evidence of the **seed** split of
  questions. Eval-split questions, gold SQL and evidence are never read while seeding.
- The split is deterministic: a question lands in seed or eval by a hash of its id (seed value 13, 30% to seed), so it
  does not depend on list order. For `financial` that is 30 seed and 76 eval questions, recorded in
  `data/split-financial.json`.
- The evidence of eval questions is hidden from every mode except `--oracle`, which exists to show the ceiling.
- BIRD's `database_description` CSVs are a data dictionary that ships with the database, so they may seed knowledge
  (label: `docs`). The per-question evidence of eval questions never may.

Provenance format is validated on import, and a test fails if any item cites an eval-split question.

## Running the comparison

```bash
py -3.12 -m uv run python -m app.evaluate --json results.json
```

This runs every eval-split question in three modes with the evidence hidden:

| Mode | What the model gets besides the schema |
| --- | --- |
| `schema` | Nothing. |
| `docs` | Flat RAG: the 8 closest chunks of BIRD's `database_description` CSVs, one chunk per column, found by the same hybrid search. No item types, no term matching, no seed-split knowledge, no examples. |
| `memory` | The 8 items retrieved from the typed store. |

`docs` is the control that matters most. `memory` beating `schema` could just mean "more text helps"; `memory` beating
`docs` means the typed, curated items help beyond the data dictionary they were partly written from.

It executes the gold and predicted SQL and reports execution accuracy overall and by BIRD difficulty, the error rate,
the mean number of rules cited per question, and two diagnostics:

- **coverage:** the share of the gold query's columns (join keys left out) that are named anywhere in the items the
  model was shown. It is a proxy for retrieval recall: low coverage on a wrong answer points at retrieval, high
  coverage at generation. It reads the gold SQL, so it is measured after the fact and never shown to the model.
- **warned:** queries that ran but returned no rows, an all-NULL column, or mostly duplicate rows (often a join that
  fans out). `ask` prints the same warnings. Each pair of modes (memory vs schema, docs vs schema, memory vs docs) gets
a paired test on the same questions: the EX gap, a 95% bootstrap interval (10,000 resamples of the questions, fixed
seed), the questions only one side got right, and an exact two-sided McNemar p-value. With 76 questions, one question
is 1.3 points, so a gap whose interval crosses zero is not a result. `results.json` holds the same plus the SQL, the
item ids and the outcome of every question.

Options: `--db`, `--modes schema docs memory`, `--limit N`, `--json FILE`, `--oracle`, `--exclude-kinds KIND ...` and
`--yes`.

**Ablation.** `--exclude-kinds` leaves item kinds out of the memory store for that run, for example
`--modes memory --exclude-kinds example value_hint`. Run it once per kind to see which kinds carry the gain. The
excluded kinds are recorded in `results.json`.

Per mode, every question costs one LLM call, or two if the first SQL fails to run, so the total is questions x modes
(times one to two): 228 to 456 calls for a full three-mode run on `financial`. The harness prints the number of calls, the model and the effort first and asks for confirmation;
`--yes` skips the prompt. Try `--limit 5` before a full run.

Memory mode always seeds a temporary store from `knowledge/<db>.yaml`, so a run is reproducible from the committed
file. Your real `memory.db` is never read or changed. A failed call on one question (a timeout, an overloaded API) is
recorded as that question's error and the run continues; after 5 failures in a row the run stops and prints the partial
report. Missing credentials stop it at once. With `--json`, results are saved after every question, together with the
retrieval mode actually used (`hybrid` or `bm25`), the SHA-256 of the knowledge file, the number of items retrieved and
the git commit.

`--oracle` adds a third row where the question's BIRD evidence is given to the model as a single rule. It is an
upper-bound reference. It reads eval evidence, so it is not a fair system and its number must not be compared with the
other two as a result.

Execution match is BIRD's: the result sets are compared as sets of rows. One deviation: floats are rounded to 6
decimal places before comparing, so tiny floating-point differences do not count as wrong answers. A result cut off by
the row limit counts as wrong and is reported as `n_truncated`.

### Results

Pending. The comparison needs an Anthropic API key and the BIRD download, and has not been run yet. The harness itself is tested end to end
with a fake model, which returns gold SQL only when the needed rule is present, so no numbers are claimed here.

## Tests

```bash
py -3.12 -m uv run pytest -q
```

No test uses the network or a paid API; the model is faked and the databases are tiny SQLite files built in a temp
folder.

## Limitations

- One database. The knowledge for `financial` is hand-written, so the result says how well good rules help, not how
  well they can be mined automatically.
- The seed split is only 30 questions, so the knowledge covers what those questions show plus what the schema, docs and
  data reveal. Eval questions touching other columns get less help.
- Retrieval is keyword-based unless the `embed` extra is installed.
- The result depends on the model and effort setting; numbers from one setting do not carry over to another.
- Gold SQL is trusted as is, including the few BIRD dev queries that are wrong or fail to run.
- Gold queries that fail to run count as wrong; the summary shows how many.

## Roadmap

1. **Retrieval MVP (built).** Typed store, hand-seeded for one database, term matching plus hybrid retrieval,
   generation and execution, the eval harness with a flat-RAG baseline, paired tests and per-kind ablation. Done when
   memory beats both baselines on `financial` with an interval that excludes zero.
2. **Learning from usage.** Capture corrections, edits and thumbs; distill a correction into a rule (not just a
   question and SQL pair) that lands as a draft in a review queue; run a simulated-user loop on BIRD with the evidence
   hidden, where an LLM user who can see the gold SQL gives corrections. Done when accuracy rises with the number of
   interactions and plateaus above the baseline. Closest prior work:
   [Cook et al. 2025](https://arxiv.org/abs/2511.10674).
3. **Bootstrapping and a graph.** Mine query logs and docs into draft rules; expand retrieval through
   term → metric → column → table → join path; ask a clarifying question when a term has several definitions. Also a
   schema editor: an LLM drafts table and column descriptions, editable in a UI, with a setup chatbot that asks about
   ambiguous columns.
4. **Training the retriever.** Log (question, rules used, accepted) triples and fine-tune the embedder or a reranker
   on them. The business logic stays in the editable store; only finding it gets learned.
5. **Serving and cost-aware execution.** An MCP server for the store and the query tool; EXPLAIN-based cost checks
   with a budget, rewrites, and efficient patterns learned into memory. Done when BIRD's VES and the estimated cost
   improve without losing execution accuracy.

Also planned for the eval: result sanity checks (empty results, row counts blown up by a bad join), retrieval recall
of the rules each question needs, and a writeup with the full ablation ladder (schema, flat RAG, typed memory, plus
graph, plus trained retriever).

## License

Copyright (c) 2026 Frederick Wang. All rights reserved. See [LICENSE](LICENSE).
