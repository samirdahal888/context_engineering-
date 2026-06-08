"""
experiments/exp02.py

Clean, importable logic for Experiment 2 (Context Growth & Context Rot),
extracted from notebooks/exp02_context_rot.ipynb.

PURE PYTHON ONLY: no notebook code, no plotting.

ONE pipeline, policy-driven: both arms call _run_loop with a different
ContextPolicy -- nothing forks into unrelated code.

    run_naive_loop(n_papers)       -> dict   (raw history, no budget)
    run_engineered_loop(n_papers)  -> dict   (truncate + clear old + cap)

Method (needle-in-growing-context):
  1. Plant a synthetic needle fact as the first message.
  2. Load N papers in sequence (each a large tool output appended to history).
  3. Probe: ask the model to recall the needle.
  4. Measure recall, tokens, latency and cost.

`final_tokens`, `output_tokens`, `latency` are REAL (from the Bedrock Converse
call). `cost` uses published per-1M rates from settings. The per-iteration
`tokens_per_iter` curve uses core.tokenizer (its shape -- growing vs flat -- is
the point).
"""

import json
import time
from typing import Any

from core.context_models import ContextPolicy
from core.settings import settings
from core.tokenizer import count_tokens
from tools.load_document import load_document

# --- The needle (synthetic + fixed for reproducibility) ---
NEEDLE = (
    "IMPORTANT - remember this for later: the internal review code "
    "for this session is GLASSWING-2026."
)
NEEDLE_ANSWER = "GLASSWING-2026"
PROBE = "What is the internal review code for this session?"
CLEARED_STUB = "[document processed - content cleared to preserve budget]"

_INDEX = json.loads(settings.index_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# message + bedrock helpers
# --------------------------------------------------------------------------- #
def _user(text: str) -> dict:
    return {"role": "user", "content": [{"text": text}]}


def _assistant(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


def _bedrock_client():
    import boto3

    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


def _messages_text(messages: list[dict]) -> str:
    return " ".join(b["text"] for m in messages for b in m["content"])


def _messages_tokens(messages: list[dict]) -> int:
    return count_tokens(_messages_text(messages))


def _cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1e6 * settings.price_in_per_1m
        + output_tokens / 1e6 * settings.price_out_per_1m
    )


# --------------------------------------------------------------------------- #
# policy operations (truncate / clear / cap)
# --------------------------------------------------------------------------- #
def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Keep roughly the first `max_tokens` worth of words."""
    max_words = max(1, int(max_tokens / settings.tokens_per_word))
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "\n[...truncated...]"


def _new_tool_output(doc_id: str, policy: ContextPolicy) -> str:
    """Load a paper, optionally truncating it per the tool-output cap."""
    raw = load_document(doc_id)
    if policy.truncate_tool_output_to is not None:
        raw = _truncate_to_tokens(raw, policy.truncate_tool_output_to)
    return f"Loaded document {doc_id}:\n{raw}"


def _is_doc_message(message: dict) -> bool:
    text = message["content"][0]["text"]
    return message["role"] == "user" and text.startswith("Loaded document")


def _clear_old_docs(messages: list[dict]) -> None:
    """Replace every doc message except the most recent with a stub."""
    doc_idxs = [i for i, m in enumerate(messages) if _is_doc_message(m)]
    for idx in doc_idxs[:-1]:
        messages[idx] = _user(CLEARED_STUB)


def _enforce_budget(messages: list[dict], policy: ContextPolicy) -> None:
    """Last-resort cap: shrink the most recent doc until under max_tokens."""
    while _messages_tokens(messages) > policy.max_tokens:
        doc_idxs = [i for i, m in enumerate(messages) if _is_doc_message(m)]
        if not doc_idxs:
            break
        idx = doc_idxs[-1]
        words = messages[idx]["content"][0]["text"].split()
        if len(words) <= 50:
            break
        messages[idx] = _user(" ".join(words[: len(words) // 2]) + "\n[...truncated...]")


# --------------------------------------------------------------------------- #
# the single policy-driven loop
# --------------------------------------------------------------------------- #
def _probe(messages: list[dict]) -> tuple[str, dict, float]:
    """Append the probe, call the model, return (answer, usage, latency)."""
    messages.append(_user(PROBE))
    start = time.time()
    resp = _bedrock_client().converse(
        modelId=settings.bedrock_model_id,
        messages=messages,
        inferenceConfig={"maxTokens": 200, "temperature": settings.temperature},
    )
    latency = time.time() - start
    answer = resp["output"]["message"]["content"][0]["text"]
    return answer, resp["usage"], latency


def _run_loop(n_papers: int, policy: ContextPolicy) -> dict[str, Any]:
    """The one loop both arms share; behaviour comes entirely from `policy`."""
    messages: list[dict] = [_user(NEEDLE), _assistant("Acknowledged.")]
    doc_ids = [p["id"] for p in _INDEX[:n_papers]]

    tokens_per_iter: list[int] = []
    steps: list[dict] = []
    for i, doc_id in enumerate(doc_ids, start=1):
        messages.append(_user(_new_tool_output(doc_id, policy)))
        messages.append(_assistant(f"Loaded {doc_id}."))

        if policy.clear_old_tool_results:
            _clear_old_docs(messages)
        _enforce_budget(messages, policy)

        total = _messages_tokens(messages)
        tokens_per_iter.append(total)
        steps.append(
            {
                "iteration": i,
                "total_tokens": total,
                "needle_present": NEEDLE_ANSWER in _messages_text(messages),
            }
        )

    answer, usage, latency = _probe(messages)
    return {
        "answer": answer,
        "needle_recalled": NEEDLE_ANSWER.lower() in answer.lower(),
        "tokens_per_iter": tokens_per_iter,
        "final_tokens": usage["inputTokens"],
        "output_tokens": usage["outputTokens"],
        "latency": latency,
        "cost": _cost_usd(usage["inputTokens"], usage["outputTokens"]),
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# public arms
# --------------------------------------------------------------------------- #
def run_naive_loop(n_papers: int) -> dict[str, Any]:
    """Unbounded history: raw tool outputs, no truncation, no clearing."""
    policy = ContextPolicy(
        mode="naive",
        truncate_tool_output_to=None,
        clear_old_tool_results=False,
        max_tokens=10**9,  # effectively no cap
    )
    return _run_loop(n_papers, policy)


def run_engineered_loop(n_papers: int) -> dict[str, Any]:
    """Budgeted history: truncate each tool output, clear old ones, cap total."""
    policy = ContextPolicy(mode="engineered")  # truncate=1500, clear=True, cap=8000
    return _run_loop(n_papers, policy)


__all__ = ["run_naive_loop", "run_engineered_loop", "NEEDLE", "NEEDLE_ANSWER", "PROBE"]
