"""
core/context_models.py

The three "labeled lunchboxes" (Pydantic models) that the whole lab speaks in.
These are framework-agnostic: NO Strands, NO Streamlit imports here on purpose,
so any layer (notebook, experiment module, UI) can use them.
"""

from pydantic import BaseModel
from typing import Literal


class ContextItem(BaseModel):
    """One piece of stuff we put in front of the model, plus its token size."""
    content: str
    source_type: Literal["system", "index", "document", "question"]
    tokens: int


class ContextPolicy(BaseModel):
    """The switch that decides HOW we build context.

    Experiment 1 used `mode` to pick load_everything vs progressive.
    Experiment 2 adds budgeting knobs used by the engineered agent loop:
    how big any single tool output may be, and whether old tool outputs get
    wiped so the context stays lean across loop iterations.

      naive       = keep everything raw, no budget (context grows forever)
      engineered  = truncate each tool output + clear old ones + cap total

    Experiment 3 (Compaction) reuses `max_tokens` as the context-window ceiling
    and adds two knobs: whether to compact at all, and the fill fraction at
    which compaction fires.
    """
    # Exp 1 modes still valid; Exp 2 adds naive/engineered. Kept permissive so
    # all experiments share one policy type.
    mode: Literal["load_everything", "progressive", "naive", "engineered"] = "engineered"

    # --- Exp 2 budgeting knobs (engineered arm only) ---
    max_tokens: int = 8000                       # budget / window ceiling (engineered only)
    truncate_tool_output_to: int | None = 1500   # cap per tool output, in tokens
    clear_old_tool_results: bool = True          # replace old raw tool outputs with a stub

    # --- Exp 3 compaction knobs (engineered arm only) ---
    compaction_enabled: bool = True              # summarize + reset history near the limit
    compaction_threshold: float = 0.8            # compact when tokens > threshold * max_tokens


class ContextTrace(BaseModel):
    """The X-ray report: where every token went."""
    segment_breakdown: dict[str, int]    # source_type -> token count
    total_tokens: int
    steps: list[dict] = []               # per-iteration: {iteration, total_tokens, needle_present}
