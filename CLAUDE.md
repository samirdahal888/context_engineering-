# CLAUDE.md — Context Engineering Lab

Read this first, then start working. It captures how the owner of this repo likes
to work and the conventions every experiment follows.

---

## What this project is

An interactive **Context Engineering Lab**: a Streamlit platform where each
experiment shows a context-engineering *problem* (naive arm) and its *fix*
(engineered arm), measured with **real** AWS Bedrock (Claude) calls. There are
10 planned experiments; built so far:

- **Exp 1 — Progressive Disclosure** (load-everything vs index + `load_document`)
- **Exp 2 — Context Rot** (truncate + clear tool outputs across an agent loop)
- **Exp 3 — Compaction** (summarize history at a threshold → the sawtooth)
- **Exp 4 — External Memory** (note-taking to `data/memory/notes.json` that
  survives a full session reset; two sessions, naive ≈ 2 papers vs engineered 4)
- **Exp 5 — Multi-Agent** (3 isolated sub-agents + a composer; parent/composer
  context ~1k vs single-agent ~80k — compression by architecture, with an honest
  latency tradeoff: multi is ~2× slower)

---

## How the owner likes to work (IMPORTANT)

1. **Teach while building.** Explain in simple, beginner-friendly language —
   plain words, analogies, small tables, the occasional emoji. Assume curiosity,
   not prior jargon. Keep it warm and concrete.
2. **Notebook FIRST, always.** For every new experiment: build the Jupyter
   notebook, run/validate it, *then* extract the logic into clean `.py` files.
   Never write the module before the notebook is proven. This is non-negotiable.
3. **Small notebook cells.** One job per cell, short and sweet (≈≤30 lines).
   "Peek" cells that print intermediate values are encouraged.
4. **Verify before advancing.** Run the acceptance check for each step and show
   the real output before moving on. Don't claim something works without proof
   (a booted server ≠ a run script — execute the actual code path).
5. **Be honest, flag tradeoffs.** The owner values being told *why* and being
   shown design decisions (e.g., "the built-in can't express this precisely, so
   here's a custom approach"). Push back with reasons when warranted.
6. **Real numbers only.** Every token count / latency / cost must come from a
   real API response. Never hardcode or fake. When a value is an estimate
   (e.g., the local tokenizer fallback), label it clearly as approximate.
7. **Verify the official source first.** Strands (and other deps) change
   frequently — confirm the current API from the official docs before using it.
   (In this repo, the working `@tool`/`Agent` patterns already in the code count
   as proof; re-verify externally when adding something new.)
8. **Follow clean-code principles (Uncle Bob).** Small single-responsibility
   functions, intention-revealing names, no dead code.



---

## Architecture (three layers per experiment)

```
notebooks/expNN_*.ipynb   ← scratchpad: build + debug here. NEVER imported.
experiments/expNN.py      ← clean extracted logic. Pure Python, no UI, no prints.
pages/N_*.py              ← Streamlit page. Thin shell: collect input, call
                            experiments/, render. NO experiment logic here.
streamlit_app.py          ← landing page + multipage entry point.
core/                     ← framework-agnostic shared code (NO strands imports).
tools/                    ← Strands @tool functions (load_document, memory_tools).
memory/                   ← framework-agnostic file-backed note store (Exp 4).
                            NO strands imports; the @tools wrap it.
agents/                   ← Strands sub-agents exposed as tools (Exp 5,
                            Agents-as-Tools). Builds Agents, so it MAY import strands.
data/corpus/ + index.json ← the 7 arXiv papers + lightweight index (reused).
data/memory/notes.json    ← Exp 4 external memory (gitignored; regenerated at run).
```

**Reuse, don't rebuild.** New experiments reuse `core/tokenizer.py`,
`core/settings.py`, `core/context_models.py`, `core/compaction.py`, `data/`,
`tools/load_document.py`, `memory/store.py`, and the agent + agent-loop patterns
from earlier experiments.

---

## Coding conventions

- **Config lives in `core/settings.py`** — a Pydantic `Settings` object loaded
  from `.env`. Model id, region, token budgets, pricing, paths all come from
  `settings`. No magic numbers, no scattered `os.getenv`. Import `settings` and
  read from it. (`settings.py` also calls `load_dotenv()` so boto3 sees AWS creds.)
- **Clean code in `experiments/*.py`:** small single-responsibility functions,
  type hints, docstrings, no prints, no pandas/matplotlib. Return plain dicts.
- **One pipeline, policy-driven.** Naive vs engineered arms differ only by a
  `ContextPolicy` (in `core/context_models.py`) or by which tools are enabled —
  not by forking into separate code paths. Build a single `_run_loop(..., policy)`
  (or `_run_session(...)`) and two thin wrappers.
- **`core/` and `memory/` stay framework-agnostic** — no `strands` imports.
  Strands `@tool` wrappers live in `tools/` and call into those pure modules
  (e.g. `tools/memory_tools.py` → `memory/store.py`). Inject the model call as a
  callable when a `core/` service needs an LLM (see `core/compaction.py`).
- **Pages** import from `experiments/` and start with a 3-line shim so they also
  work when run directly:
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  ```

---

## Environment & running

- **Python env:** `uv` venv at `.venv/`. Run things with `.venv/bin/python`
  (or `source .venv/bin/activate`). Install deps:
  `VIRTUAL_ENV="$(pwd)/.venv" uv pip install -r requirements.txt`.
- **Run the app:** `.venv/bin/streamlit run streamlit_app.py` (sidebar lists all
  experiment pages).
- **Run a notebook:** `.venv/bin/jupyter notebook notebooks/<file>.ipynb`.
- **Notebooks are built via a generator script** (`/tmp/build_nbN.py` using
  `nbformat`) so the JSON is always valid — then validated with `nbformat.validate`.
- **Verify pages** with `streamlit.testing.v1.AppTest` (it executes the script
  body, catching import errors that a plain server boot misses).

---

## AWS Bedrock specifics (learned the hard way)

- Auth via **temporary** AWS creds in `.env` (`AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) — they **expire after a few hours**;
  if calls start failing with auth errors, the creds need refreshing.
- **Model:** `us.anthropic.claude-3-5-haiku-20241022-v1:0` (set in `.env` as
  `BEDROCK_MODEL_ID`). Newer Sonnet models aren't enabled on this account.
- **`CountTokens` is NOT supported by Claude 3 models** → `core/tokenizer.py`
  gracefully falls back to a `words × 1.3` estimate (with a one-time warning).
  Exact counts come from each call's `usage.inputTokens`. Headline numbers are
  real; the local tokenizer is only for per-segment X-ray breakdowns.
- **Claude context window = input ceiling; output (`max_tokens`) is a separate
  budget.** Academic papers run **~2 tokens/word** (not 1.3) — see
  `settings.tokens_per_word`.
- Use the **Converse API** (`bedrock-runtime.converse`); messages must
  **alternate user/assistant** and start with user. Put task instructions in the
  `system` param when you need consecutive same-role turns.
- **Haiku tool-use is reliable for *fetching*, flaky for *re-weaving*.** In Exp 4
  the agent always called `read_progress` and got the notes, but inconsistently
  folded that recovered tool-result back into a prose answer (coverage swung
  2–3/4, and a *stricter* prompt made it worse). Fix: ground the final synthesis
  in the structured artifact — write the brief *directly from* `notes.json`
  rather than hoping the model re-derives it. The durable note IS the source of
  truth. Prefer grounding over prompt-nagging when output must be reproducible.
- **Real per-call token usage lives in Strands' agent metrics**, not just the
  cumulative total. `result.metrics.accumulated_usage` is the SUM across cycles
  (overcounts a multi-turn agent). For a real "peak context" use the max single
  call: `max(c.usage["inputTokens"] for inv in result.metrics.agent_invocations
  for c in inv.cycles)` (helper: `agents.subagents.peak_input_tokens`).
- **Agents-as-Tools = an `@tool` that builds an inner `Agent` and returns
  `str(response)`** (verified at strandsagents.com). Only the returned string
  enters the caller's context, so a sub-agent's raw work is isolated by
  construction. Capture sub-agent metrics OUT-OF-BAND (a module list) — never
  route them through the parent, or you defeat the isolation you're
  demonstrating. Use `callback_handler=None` to mute sub-agent streaming.
- **Don't trust Haiku as an autonomous orchestrator.** Giving the parent the
  three sub-agent tools and telling it to call them, it sometimes answered the
  multi-part question from its own parametric knowledge and skipped the tools —
  a *silently invalid* run (empty sub-agent usage, a tiny "parent" context that
  never delegated). Exp 5 therefore **orchestrates the specialists explicitly**
  (`run_all_specialists`) and feeds their summaries to a tool-less **composer**.
  Each specialist still runs in its own window, so isolation holds and the run is
  reproducible. (And always guard UI code against an empty usage list.)

---

## Per-experiment build checklist

1. Extend `core/context_models.py` (`ContextPolicy`) / `core/settings.py` if the
   experiment needs new knobs. Add any new framework-agnostic service to `core/`
   or `memory/`, and its Strands `@tool` wrapper to `tools/`.
2. Build `notebooks/expNN_*.ipynb` (small cells) and **run it** to prove the
   effect (`jupyter nbconvert --to notebook --execute --inplace` is a clean way
   to run + capture real outputs). Gitignore any runtime-generated artifacts.
3. Extract pure functions to `experiments/expNN.py` (config from `settings`).
4. Add `pages/N_*.py` (thin shell) and verify with `AppTest`.
5. Show the acceptance check output at each step (run the real code path, not
   just imports — for agent experiments that means real Bedrock calls).
