"""
core/compaction.py

Framework-agnostic compaction logic for Experiment 3.

"Compaction" = when the running history nears the context window, summarize it
(MAXIMIZE RECALL FIRST, then precision), then REPLACE the raw history with that
summary and keep going. That is what turns an unbounded, doomed loop into the
"sawtooth" that survives past the window limit.

This module owns the *policy and the prompt* of compaction only. It deliberately
does NOT know about Bedrock, boto3, Strands, or Streamlit -- the caller injects a
`summarize` function. That keeps `core/` reusable and trivially testable:

    service = CompactionService()
    if service.should_compact(input_tokens, policy):
        summary = service.compact(task, history_text, summarize=my_llm_call)

`summarize` has the shape: (system_instruction, user_text) -> summary_text.
"""

from typing import Callable

from core.context_models import ContextPolicy

# A summarizer the caller provides: (system_instruction, user_text) -> summary.
SummarizeFn = Callable[[str, str], str]

# Recall-first instruction (research doc Section 8.1): when in doubt, KEEP it.
# Losing a decision or the running result is far worse than a longer summary.
DEFAULT_INSTRUCTION = (
    "You compress conversation history. MAXIMIZE RECALL FIRST, then precision. "
    "Preserve: decisions made, the running result/theme list, open steps, and "
    "key facts (names, numbers). Drop redundant raw source text whose "
    "conclusions are already captured. Output a compact summary."
)


class CompactionService:
    """Decide WHEN to compact and BUILD the recall-first summary."""

    def __init__(self, instruction: str = DEFAULT_INSTRUCTION) -> None:
        self.instruction = instruction

    def should_compact(self, input_tokens: int, policy: ContextPolicy) -> bool:
        """True once the (real) input tokens cross the policy's fill threshold."""
        if not policy.compaction_enabled:
            return False
        return input_tokens > policy.compaction_threshold * policy.max_tokens

    def compact(self, task: str, conversation_text: str, summarize: SummarizeFn) -> str:
        """Summarize the history so far; the result REPLACES that history.

        The model call itself is injected via `summarize`, so this stays
        framework-agnostic (no boto3/Strands here).
        """
        user_text = f"Original task: {task}\n\nConversation so far:\n{conversation_text}"
        return summarize(self.instruction, user_text)


__all__ = ["CompactionService", "SummarizeFn", "DEFAULT_INSTRUCTION"]
