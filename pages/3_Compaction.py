"""
pages/3_Compaction.py

Streamlit UI for Experiment 3 (Compaction). Thin shell: collect N, call the
pure functions in experiments/exp03.py, render the token sawtooth + outcome.
"""

import sys
from pathlib import Path

# Ensure the project root is importable, whether launched via streamlit_app.py
# or directly via `streamlit run pages/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from core.settings import settings
from experiments.exp03 import run_engineered, run_naive

st.set_page_config(page_title="Exp 3: Compaction", layout="wide")

st.title("Experiment 3 — Compaction (the sawtooth)")
st.markdown(
    """
**The problem:** some tasks *need* their history -- you can't just truncate it
(that was Exp 2). But history that grows unbounded eventually **overflows the
context window and the task dies**.
**The fix:** when context nears the limit, **summarize** the history
(recall-first), replace the raw history with that summary, and keep going.
**Watch:** the naive line climbs into the ceiling and stops; the engineered line
climbs, then **drops** at each compaction event -- and finishes.
"""
)

window = settings.compaction_window_tokens
st.caption(
    f"Demo window: **{window:,} tokens** · compact at "
    f"**{int(settings.compaction_threshold * window):,}** "
    f"(small numbers so overflow is fast + visible; every token count is real)."
)

n = st.slider("Papers to read (running-synthesis task)", 2, 7, 7)


def outcome_badge(completed: bool) -> str:
    return "✅ completed" if completed else "❌ DIED (overflow)"


if st.button("▶ Run both loops", type="primary"):
    with st.spinner("Running naive loop (history grows unbounded)..."):
        naive = run_naive(n)
    with st.spinner("Running engineered loop (compact at threshold)..."):
        eng = run_engineered(n)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐘 Naive loop")
        st.metric("Outcome", outcome_badge(naive["completed"]))
        st.metric("Papers processed", f"{naive['papers_processed']} / {n}")
        st.metric("Peak tokens", f"{max(naive['tokens_per_iter']):,}")
        st.caption("Real input tokens after each iteration")
        st.line_chart(naive["tokens_per_iter"])
        st.markdown("**Final theme list:**")
        st.write(naive["final_answer"])

    with col2:
        st.subheader("🦊 Engineered loop")
        st.metric("Outcome", outcome_badge(eng["completed"]))
        st.metric("Papers processed", f"{eng['papers_processed']} / {n}")
        st.metric(
            "Compaction events",
            f"{len(eng['compaction_events'])}",
            help=f"Fired at iteration(s): {eng['compaction_events']}",
        )
        st.caption("Real input tokens after each iteration")
        st.line_chart(eng["tokens_per_iter"])
        st.markdown("**Final theme list:**")
        st.write(eng["final_answer"])

    # The sawtooth: naive vs engineered on one axis, with the window ceiling.
    st.divider()
    st.subheader("The sawtooth — context drops at each compaction event")
    naive_s = pd.Series(
        naive["tokens_per_iter"],
        index=range(1, len(naive["tokens_per_iter"]) + 1),
        name="naive",
    )
    eng_s = pd.Series(
        eng["tokens_per_iter"],
        index=range(1, len(eng["tokens_per_iter"]) + 1),
        name="engineered",
    )
    chart = pd.concat([naive_s, eng_s], axis=1)
    chart["context limit"] = window
    chart.index.name = "iteration"
    st.line_chart(chart)

    st.markdown(
        "**Same task, same papers — only `compaction_enabled` changed.** "
        "Naive hits the ceiling and dies; engineered compacts, drops, and completes."
    )
else:
    st.info("Pick N and press **Run both loops** to see the sawtooth.")
