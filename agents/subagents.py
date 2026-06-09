"""
agents/subagents.py

Specialist sub-agents for Experiment 5 (Multi-Agent vs Single-Agent), exposed
via Strands' Agents-as-Tools pattern: each is an @tool-decorated function that
builds its OWN Agent (own context window), does its verbose work in isolation,
and returns only a short summary string.

The key property: when a parent agent calls one of these tools, ONLY the
returned summary enters the parent's context. The sub-agent's raw paper loads
and reasoning never reach the parent -- that is "context compression by
architecture" (research doc Section 8.3).

We capture each sub-agent's real peak token usage OUT-OF-BAND (a module-level
list) purely so the UI can show it. This side-channel is for measurement only;
the parent agent never sees these numbers (the tool returns just the summary).
"""

from strands import Agent, tool
from strands.models import BedrockModel

from core.settings import settings
from tools.load_document import load_document

# Out-of-band capture for the UI. NOT visible to the parent agent.
_USAGE: list[dict] = []


def reset_usage() -> None:
    """Clear captured sub-agent usage (call before a multi-agent run)."""
    _USAGE.clear()


def get_usage() -> list[dict]:
    """Return a copy of the per-sub-agent usage captured in the last run."""
    return list(_USAGE)


def peak_input_tokens(result) -> int:
    """Real peak context = max inputTokens across the agent's event-loop cycles.

    Context grows monotonically as an agent loads more, so the largest single
    call's inputTokens is the agent's peak window occupancy -- a real number
    straight from the model's usage, not an estimate.
    """
    cycles = [c for inv in result.metrics.agent_invocations for c in inv.cycles]
    return max((c.usage.get("inputTokens", 0) for c in cycles), default=0)


def _result_text(result) -> str:
    """Extract the final assistant text from an AgentResult."""
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        return "".join(b.get("text", "") for b in message.get("content", []) if "text" in b)
    return str(result)


def _model() -> BedrockModel:
    """A Bedrock model capped short so sub-agents stay concise (~300 words)."""
    return BedrockModel(
        model_id=settings.bedrock_model_id,
        region_name=settings.aws_region,
        temperature=settings.temperature,
        max_tokens=settings.subagent_summary_tokens,
    )


def _run_subagent(name: str, area: str, papers: list[str], task: str) -> str:
    """Run one isolated specialist sub-agent and return ONLY its summary."""
    system_prompt = (
        f"You are a research specialist on {area}. Use load_document to read ONLY "
        f"these papers: {papers}. Then write a focused summary of how this area "
        "handles long-context problems, in 300 words or fewer, citing the paper "
        "id(s). Output only the summary."
    )
    agent = Agent(
        model=_model(),
        tools=[load_document],
        system_prompt=system_prompt,
        callback_handler=None,  # no streaming noise; this work is isolated
    )
    result = agent(f"{task}\n\nYour area: {area}. Papers to load: {papers}.")
    _USAGE.append({"name": name, "area": area, "papers": papers,
                   "peak_tokens": peak_input_tokens(result)})
    return _result_text(result)


# The three specialists as data, so a coordinator can run them in a fixed order.
SUBAGENT_SPECS = [
    ("attention", "attention / transformer architecture", ["1706.03762"]),
    ("position", "position effects in long context", ["2307.03172", "2404.06654"]),
    ("retrieval", "retrieval-augmented approaches", ["2005.11401"]),
]


def run_all_specialists(task: str) -> list[dict]:
    """Run every specialist sub-agent in isolation; return summaries + real usage.

    Deterministic delegation: unlike letting the parent LLM *decide* whether to
    call each tool (Haiku sometimes answers from its own knowledge instead), this
    guarantees all three sub-agents actually run. Each still works in its own
    window; only its summary leaves it. Populates the out-of-band usage list.
    """
    results: list[dict] = []
    for name, area, papers in SUBAGENT_SPECS:
        summary = _run_subagent(name, area, papers, task)
        results.append({**_USAGE[-1], "summary": summary})
    return results


# --------------------------------------------------------------------------- #
# The three specialists, each exposed as a tool for the parent agent.
# --------------------------------------------------------------------------- #
@tool
def research_attention(task: str) -> str:
    """Summarize how attention / transformer architecture handles long context.

    Delegates to a specialist sub-agent that reads the attention paper and
    returns a concise (~300 word) summary.

    Args:
        task: The overall comparison task, for context.
    """
    return _run_subagent("attention", "attention / transformer architecture",
                         ["1706.03762"], task)


@tool
def research_position(task: str) -> str:
    """Summarize how input POSITION affects long-context performance.

    Delegates to a specialist sub-agent that reads the position-effect papers and
    returns a concise (~300 word) summary.

    Args:
        task: The overall comparison task, for context.
    """
    return _run_subagent("position", "position effects in long context",
                         ["2307.03172", "2404.06654"], task)


@tool
def research_retrieval(task: str) -> str:
    """Summarize how RETRIEVAL augmentation addresses long-context limits.

    Delegates to a specialist sub-agent that reads the retrieval paper and
    returns a concise (~300 word) summary.

    Args:
        task: The overall comparison task, for context.
    """
    return _run_subagent("retrieval", "retrieval-augmented approaches",
                         ["2005.11401"], task)


SUBAGENT_TOOLS = [research_attention, research_position, research_retrieval]

__all__ = [
    "research_attention", "research_position", "research_retrieval",
    "SUBAGENT_TOOLS", "SUBAGENT_SPECS", "run_all_specialists",
    "reset_usage", "get_usage", "peak_input_tokens", "_run_subagent",
]
