"""
experiments/exp03.py

Clean, importable logic for Experiment 3 (Compaction), extracted from
notebooks/exp03_compaction.ipynb.

PURE PYTHON ONLY: no notebook code, no plotting, no prints.

ONE pipeline, policy-driven: both arms call _run_loop with a different
ContextPolicy -- only `compaction_enabled` differs.

    run_naive(n_papers)       -> dict   (history grows until the window overflows)
    run_engineered(n_papers)  -> dict   (compact at threshold -> the sawtooth)

Method (running-synthesis task):
  1. Read papers one at a time; after each, update a running theme list.
  2. Naive: keep appending; once real input tokens cross the window, the task
     dies (overflow).
  3. Engineered: when tokens cross the threshold, summarize the history
     (recall-first via core.compaction) and reset -- so the loop survives and
     completes.

Every value in `tokens_per_iter` is the REAL `usage.inputTokens` from the
Converse call -- that growing-then-dropping curve IS the experiment. The window
and threshold are small demo numbers (from settings) so overflow is fast and
visible.
"""

import time
from typing import Any

from core.compaction import CompactionService
from core.context_models import ContextPolicy
from core.settings import settings
from tools.load_document import load_document

# The running-synthesis task (kept in `system` so each turn alternates
# user(paper)/assistant(themes), as the Converse API requires).
TASK = (
    "Read each paper in turn. After each, update a running list of the key "
    "themes across all papers read so far. Output ONLY the theme list."
)

import json

_INDEX = json.loads(settings.index_path.read_text(encoding="utf-8"))
_COMPACTOR = CompactionService()


# --------------------------------------------------------------------------- #
# message + bedrock helpers
# --------------------------------------------------------------------------- #
def _user(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def _messages_text(messages: list[dict]) -> str:
    return " ".join(b["text"] for m in messages for b in m["content"])


def _bedrock_client():
    import boto3

    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


def _ask(system: str, messages: list[dict], max_out: int) -> tuple[str, int]:
    """One Converse call. Returns (answer, REAL input tokens)."""
    resp = _bedrock_client().converse(
        modelId=settings.bedrock_model_id,
        system=[{"text": system}],
        messages=messages,
        inferenceConfig={"maxTokens": max_out, "temperature": settings.temperature},
    )
    return resp["output"]["message"]["content"][0]["text"], resp["usage"]["inputTokens"]


def _summarize(instruction: str, user_text: str) -> str:
    """Summarizer injected into CompactionService (keeps core/ framework-agnostic)."""
    summary, _ = _ask(instruction, [_user(user_text)], settings.summary_output_tokens)
    return summary


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Keep roughly the first `max_tokens` worth of words (per-paper feed cap)."""
    max_words = max(1, int(max_tokens / settings.tokens_per_word))
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"


# --------------------------------------------------------------------------- #
# the single policy-driven loop
# --------------------------------------------------------------------------- #
def _run_loop(n_papers: int, policy: ContextPolicy) -> dict[str, Any]:
    """The one loop both arms share; behaviour comes entirely from `policy`."""
    papers = _INDEX[:n_papers]
    messages: list[dict] = []
    tokens_per_iter: list[int] = []
    compaction_events: list[int] = []
    overflowed = False
    last_answer = ""

    for i, paper in enumerate(papers, start=1):
        text = _truncate_to_tokens(load_document(paper["id"]), settings.per_paper_tokens)
        messages.append(_user(f"Paper {paper['id']}:\n{text}\n\nUpdate the running theme list."))
        answer, in_tokens = _ask(TASK, messages, settings.loop_output_tokens)
        messages.append(_assistant(answer))

        tokens_per_iter.append(in_tokens)
        last_answer = answer

        # Naive: no budget -- once we cross the window, the task dies.
        if not policy.compaction_enabled and in_tokens > policy.max_tokens:
            overflowed = True
            break

        # Engineered: compact + reset before the window fills.
        if _COMPACTOR.should_compact(in_tokens, policy):
            summary = _COMPACTOR.compact(TASK, _messages_text(messages), _summarize)
            messages = [
                _user(f"Summary so far [history compacted]:\n{summary}"),
                _assistant("Understood. Continuing the task."),
            ]
            compaction_events.append(i)

    return {
        "completed": not overflowed,
        "overflowed": overflowed,
        "final_answer": last_answer,
        "tokens_per_iter": tokens_per_iter,
        "compaction_events": compaction_events,
        "papers_processed": len(tokens_per_iter),
        "window_tokens": policy.max_tokens,
    }


# --------------------------------------------------------------------------- #
# public arms
# --------------------------------------------------------------------------- #
def _policy(compaction_enabled: bool) -> ContextPolicy:
    return ContextPolicy(
        mode="engineered",
        max_tokens=settings.compaction_window_tokens,
        compaction_enabled=compaction_enabled,
        compaction_threshold=settings.compaction_threshold,
    )


def run_naive(n_papers: int = len(_INDEX)) -> dict[str, Any]:
    """History grows unbounded -- crosses the window and the task dies."""
    return _run_loop(n_papers, _policy(compaction_enabled=False))


def run_engineered(n_papers: int = len(_INDEX)) -> dict[str, Any]:
    """Compact at the threshold -> the sawtooth -> the task completes."""
    return _run_loop(n_papers, _policy(compaction_enabled=True))


__all__ = ["run_naive", "run_engineered", "TASK"]
