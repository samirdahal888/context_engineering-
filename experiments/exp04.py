"""
experiments/exp04.py

Clean, importable logic for Experiment 4 (External Memory / Structured
Note-Taking), extracted from notebooks/exp04_external_memory.ipynb.

PURE PYTHON ONLY: no notebook code, no plotting, no prints.

The idea: run a research task as TWO separate agent sessions with a HARD RESET
between them (a brand-new agent = a real session boundary, since LLMs are
stateless). Both arms differ only by whether the memory tools are enabled:

    run_naive()       -> dict   session 2 has no access to session 1's work
    run_engineered()  -> dict   session 1 writes notes; session 2 reads them

For the engineered arm, the final brief is synthesized DIRECTLY from the
structured notes (data/memory/notes.json) -- that durable, human-inspectable
artifact is the payoff of note-taking (research doc Section 8.2), and it is what
survives the reset. The naive arm has no notes, so it cannot recover session 1.

`papers_covered` is a real check on the real brief text (regex on each paper's
signature theme). Nothing is hardcoded.
"""

import re
from typing import Any

from core.settings import settings
from memory.store import read_notes, reset_memory

# Two papers per session; the reset happens between them.
SESSION_1_PAPERS = ["2307.03172", "2404.06654"]   # Lost in the Middle, RULER
SESSION_2_PAPERS = ["2005.11401", "2311.05232"]   # RAG, Hallucination
ALL_PAPERS = SESSION_1_PAPERS + SESSION_2_PAPERS
TASK = "Build a research brief on long-context and retrieval techniques."

# A paper is "covered" if the brief mentions its signature theme (real check).
_COVERAGE_KW = {
    "2307.03172": r"lost in the middle|u-shaped|positional",
    "2404.06654": r"\bruler\b|effective context",
    "2005.11401": r"retrieval-augmented|retrieval augmented|\brag\b",
    "2311.05232": r"hallucinat",
}


def papers_covered(brief: str) -> list[str]:
    """Return the ids of papers whose signature theme appears in the brief."""
    low = brief.lower()
    return [pid for pid, pat in _COVERAGE_KW.items() if re.search(pat, low)]


# --------------------------------------------------------------------------- #
# session + synthesis helpers (Strands imports kept local to call time)
# --------------------------------------------------------------------------- #
def _run_session(system_prompt: str, user_prompt: str, tools: list) -> str:
    """One isolated session: a brand-new agent with no prior context."""
    from strands import Agent
    from strands.models import BedrockModel

    model = BedrockModel(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=settings.temperature,
        max_tokens=settings.output_max_tokens,
    )
    agent = Agent(model=model, tools=tools, system_prompt=system_prompt)
    result = agent(user_prompt)
    return "".join(
        b.get("text", "") for b in result.message.get("content", []) if "text" in b
    )


def _synthesize_brief(notes: dict) -> str:
    """Write the final brief straight from the structured notes (no tools).

    Grounding the synthesis in the saved findings makes coverage reliable and is
    the whole point of structured note-taking: the durable artifact drives the
    output instead of the model re-deriving it from a tool result.
    """
    import boto3

    bullets = "\n".join(
        f"- {p['title']} ({p['id']}): {p['finding']}" for p in notes["papers_processed"]
    )
    prompt = (
        "Write a final research brief from these saved findings. Include one short "
        f"titled section per finding, naming each paper.\n\n{bullets}"
    )
    client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    resp = client.converse(
        modelId=settings.bedrock_model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={
            "maxTokens": settings.output_max_tokens,
            "temperature": settings.temperature,
        },
    )
    return resp["output"]["message"]["content"][0]["text"]


# --------------------------------------------------------------------------- #
# public arms
# --------------------------------------------------------------------------- #
def run_naive() -> dict[str, Any]:
    """Two isolated sessions, no shared memory: session 2 loses session 1's work."""
    from tools.load_document import load_document

    # Session 1: draft a partial brief (discarded -- nothing persists).
    _run_session(
        "You are a research assistant. Read the papers and draft a brief.",
        f"{TASK} Read these papers with load_document: {SESSION_1_PAPERS}. "
        "Draft a partial brief.",
        tools=[load_document],
    )
    # --- HARD RESET: a brand-new agent runs below, with no carried-over context.
    brief = _run_session(
        "You are a research assistant. Produce the FINAL combined research brief.",
        f"{TASK} Read these papers with load_document: {SESSION_2_PAPERS}. "
        "Produce the FINAL brief covering all papers you have read.",
        tools=[load_document],
    )
    return {"final_brief": brief, "papers_covered": papers_covered(brief)}


def run_engineered() -> dict[str, Any]:
    """Two sessions sharing external notes: session 2 reads them and resumes."""
    from tools.load_document import load_document
    from tools.memory_tools import read_progress, save_finding

    reset_memory()
    # Session 1: read each paper and SAVE each finding to external memory.
    _run_session(
        "You are a research assistant. After reading EACH paper, call save_finding "
        "to record its key contribution to external memory.",
        f"{TASK} Read these papers with load_document: {SESSION_1_PAPERS}. "
        "Call save_finding once per paper.",
        tools=[load_document, save_finding],
    )
    # --- HARD RESET: new agent, no in-context memory.
    # Session 2: recover prior notes, read new papers, SAVE their findings too.
    _run_session(
        "You are resuming earlier work. FIRST call read_progress to recover prior "
        "findings, THEN read each new paper and call save_finding for it.",
        f"{TASK} First call read_progress. Then read these with load_document: "
        f"{SESSION_2_PAPERS}. Call save_finding once per new paper.",
        tools=[load_document, read_progress, save_finding],
    )
    notes = read_notes()              # external memory now holds ALL findings
    brief = _synthesize_brief(notes)  # final brief grounded in that memory
    return {
        "final_brief": brief,
        "papers_covered": papers_covered(brief),
        "notes_snapshot": notes,
    }


__all__ = ["run_naive", "run_engineered", "papers_covered", "TASK"]
