"""
experiments/exp05.py

Clean, importable logic for Experiment 5 (Multi-Agent vs Single-Agent),
extracted from notebooks/exp05_multi_agent.ipynb.

PURE PYTHON ONLY: no notebook code, no plotting, no prints.

Both arms run the SAME 3-part comparison task with the SAME load_document tool;
the only difference is the architecture:

    run_single_agent()  -> dict   one agent loads every paper into one window
    run_multi_agent()   -> dict   a parent calls 3 isolated sub-agents (tools)

The headline is the real peak context size (max single-call inputTokens, from
the model's usage). For the multi-agent arm the parent's peak excludes the
sub-agents' raw work entirely -- that is "context compression by architecture"
(research doc Section 8.3). Sub-agent token usage is reported separately
(captured out-of-band in agents.subagents; the parent never sees it).

Latency is real and reported honestly: multi-agent is usually SLOWER because of
coordination overhead -- a genuine tradeoff, not a flaw to hide.
"""

import time
from typing import Any

from agents.subagents import (
    get_usage,
    peak_input_tokens,
    reset_usage,
    run_all_specialists,
)
from core.settings import settings

TASK = (
    "Compare how three areas handle long-context problems: "
    "(1) attention/architecture, (2) position effects, (3) retrieval. "
    "Cover the relevant paper(s) for each area."
)
# Every paper the single agent must pull into its one window.
SINGLE_PAPERS = ["1706.03762", "2307.03172", "2404.06654", "2005.11401"]


def _run_agent(system_prompt: str, user_prompt: str, tools: list) -> tuple[str, int, float]:
    """Run one agent; return (answer_text, real_peak_tokens, latency_seconds)."""
    from strands import Agent
    from strands.models import BedrockModel

    model = BedrockModel(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=settings.temperature,
        max_tokens=settings.output_max_tokens,
    )
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        callback_handler=None,
    )
    start = time.time()
    result = agent(user_prompt)
    latency = time.time() - start
    text = "".join(
        b.get("text", "") for b in result.message.get("content", []) if "text" in b
    )
    return text, peak_input_tokens(result), latency


def run_single_agent() -> dict[str, Any]:
    """Naive: one agent loads every paper itself and reasons in one window."""
    from tools.load_document import load_document

    system_prompt = (
        "You are a research analyst. Use load_document to read ALL of these papers: "
        f"{SINGLE_PAPERS}. Then write a structured comparison of how three areas "
        "handle long-context problems: (1) attention/architecture, (2) position "
        "effects, (3) retrieval. Cite paper ids."
    )
    text, peak, latency = _run_agent(system_prompt, TASK, [load_document])
    return {"answer": text, "peak_tokens": peak, "latency": latency}


def _compose_prompt(specialists: list[dict]) -> str:
    """Build the composer prompt from ONLY the specialists' summaries."""
    blocks = "\n\n".join(
        f"[{s['name']}] {s['area']}:\n{s['summary']}" for s in specialists
    )
    return (
        f"{TASK}\n\nThree specialists each studied one area and returned a summary. "
        "Write a single structured comparison of the three areas using ONLY these "
        f"summaries (do not add outside facts):\n\n{blocks}"
    )


def run_multi_agent() -> dict[str, Any]:
    """Engineered: 3 isolated specialist sub-agents + a composer that sees only summaries.

    We orchestrate the specialists explicitly (deterministic delegation) rather
    than hoping the coordinator LLM decides to call each one -- Haiku sometimes
    answers from its own knowledge instead, which silently skips the sub-agents.
    Each specialist still works in its own window; the composer's context holds
    only the three short summaries, so the isolation lesson is intact and the run
    is reproducible.
    """
    reset_usage()
    start = time.time()
    specialists = run_all_specialists(TASK)  # 3 isolated windows, guaranteed
    composer_system = (
        "You are a research coordinator. Compose a final structured comparison "
        "from the specialist summaries you are given. Use only what they provide."
    )
    text, peak, _ = _run_agent(composer_system, _compose_prompt(specialists), tools=[])
    latency = time.time() - start
    usage = get_usage()
    return {
        "answer": text,
        "parent_peak_tokens": peak,
        "subagent_usage": usage,
        "subagent_tokens": sum(u["peak_tokens"] for u in usage),
        "latency": latency,
    }


__all__ = ["run_single_agent", "run_multi_agent", "TASK", "SINGLE_PAPERS"]
