# Contributing

Thanks for hacking on gpuprof. This document is a runbook — the
minimum to get productive.

## Local dev

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[server,postgres,dev,host]"

# Optional — required only for the auto-instrumentation tests:
pip install torch pytorch-lightning transformers
```

Run the test suite:

```bash
pytest tests/                        # ~140 tests, ~30 seconds
pytest tests/ -x                     # stop on first failure
pytest tests/test_insights.py -v     # just one file
pytest -k regression                 # subset by name
```

CI runs the mock-GPU matrix (Linux + macOS × Python 3.11–3.13) plus
a separate torch-enabled job. All jobs must be green to merge.

## Adding an insight rule

Rules live in [`src/gpuprof/insights.py`](src/gpuprof/insights.py).
Each one is a small pure function that takes a `Ctx` and returns
`Optional[dict]`:

```python
def rule_my_thing(c: Ctx) -> Optional[dict]:
    """One-line docstring naming the phenomenon."""
    if not c.something_i_need:            # bail fast
        return None
    if c.metric_of_interest < threshold:  # not firing
        return None
    return {
        "severity": "medium",              # "low" | "medium" | "high"
        "title":    f"...concise, <80 chars...",
        "recommendation": (
            "One-paragraph actionable fix. Name the exact torch API "
            "or config knob. If the fix depends on which other rule "
            "also fired, cross-reference by name."
        ),
        "evidence": {"...": "raw numbers so a savvy user can verify"},
    }
```

Then:

1. Add it to the `RULES = [...]` list at the bottom of the module.
2. Add a test in `tests/test_insights.py` (or `test_host_diagnosis.py`
   / etc.) with a synthetic `Ctx` that both fires the rule and one
   that must not. Use the existing helpers as templates.
3. If your rule needs new data, add the field to `Ctx` and populate
   it in `_build_ctx` — one SQL aggregate query is preferred over
   per-row Python.

Style:
- **Titles are diagnoses**, not metrics — "34% of step time waiting
  on the loader" beats "dataloader_wait_s = 0.034".
- **Recommendations are commands**, not adjectives — "increase
  num_workers to 8" beats "consider tuning the DataLoader".
- **Severities are calibrated**: `high` = fix before the next
  training run; `medium` = fix this week; `low` = worth knowing.
- **False positives** are worse than false negatives — a rule that
  cries wolf gets ignored. Test with the noisy end of your data.

## Adding a framework adapter

Adapters live in `src/gpuprof/integrations/`. Pattern:

- Deferred import of the framework at class-definition time (so
  `import gpuprof` never pulls in Lightning / HF / DeepSpeed).
- Late-bind the class to the framework's base via `type()` at
  module load time so the framework's dispatcher recognizes it.
- Wire per-step callbacks to `prof.step()` / `s.phase()`; use
  `prof.on_end(...)` for end-of-run hooks.

Look at [`lightning.py`](src/gpuprof/integrations/lightning.py) as
the reference — it handles the trickiest case (MRO ordering with
`pl.Callback`).

## Adding a new storage kind

Both `src/gpuprof/store.py` (client) and `src/gpuprof/server/store.py`
(server) go through:

1. Add the kind to `BATCH_KINDS` in
   [`_batch.py`](src/gpuprof/_batch.py).
2. Add the SQL table to `SCHEMA` + a `_MIGRATIONS` entry that lets
   old DBs get the new column without a schema wipe.
3. Add an `_insert_KIND(conn, run_id, rows)` helper next to the
   existing ones.
4. Wire it into the client Store's `_flush` and the server's
   `_INSERT_FN` dispatch table + `PostgresServerStore`.
5. Add a `push_KIND(...)` method on both stores + the
   `Remote` pusher.
6. Update the ingest handler in
   [`server/app.py`](src/gpuprof/server/app.py) to accept the new
   field on the JSON body.
7. Add a test.

## Coding conventions

- **Type hints on public functions.** Optional on private helpers.
- **`from __future__ import annotations`** at the top of every
  module.
- **No exception-swallowing without a reason in a comment.** Bare
  `except Exception: pass` is only OK in "never crash the training
  loop" hooks; annotate why.
- **80-column soft limit.** Break long strings; wrap docstrings.
- **Docstrings for every module and every public class.** One
  paragraph is fine; the "why" matters more than the "what".

## Running the dashboard locally

```bash
python test_dashboard.py     # mock 4 GPUs, in-process server
```

Open the printed URL. Charts update live. See
[`test_dashboard.py`](test_dashboard.py) for the shortest-possible
integration example.

## What to open an issue about

- **Wrong insight**: paste the output + describe what you actually
  observed. Rule thresholds are the easiest thing to tune.
- **Overhead >3%**: with a reproducer. Auto-instrumentation
  overhead is the one hot path.
- **API footgun**: something in the `import gpuprof` surface that
  bit you. Ergonomics is the top priority for 1.0.
