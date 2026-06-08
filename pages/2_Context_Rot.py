"""
pages/2_Context_Rot.py

Streamlit UI for Experiment 2. Thin shell: collect N, call the pure functions
in experiments/exp02.py, render token growth + latency + needle recall.
"""

import sys
from pathlib import Path

# Ensure the project root is importable, whether launched via streamlit_app.py
# or directly via `streamlit run pages/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from experiments.exp02 import run_engineered_loop, run_naive_loop

st.set_page_config(page_title="Exp 2: Context Rot", layout="wide")

st.title("Experiment 2 — Context Growth & Context Rot")
st.markdown(
    """
**The problem:** every agent-loop iteration appends a tool output. Context grows,
attention thins, and the model starts missing facts that are still present.
**Watch:** tokens, latency and cost climb while needle recall is at risk — then
the fix holds them all steady.
"""
)

n = st.slider("Papers to load (context-growth driver)", 1, 7, 5)


def recall_badge(recalled: bool) -> str:
    return "✅ yes" if recalled else "❌ NO"


if st.button("▶ Run both loops", type="primary"):
    with st.spinner("Running naive loop (context grows unbounded)..."):
        naive = run_naive_loop(n)
    with st.spinner("Running engineered loop (truncate + clear)..."):
        eng = run_engineered_loop(n)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐘 Naive loop")
        st.metric("Final tokens", f"{naive['final_tokens']:,}")
        st.metric("Latency", f"{naive['latency']:.1f}s")
        st.metric("Cost", f"${naive['cost']:.4f}")
        st.metric("Needle recalled", recall_badge(naive["needle_recalled"]))
        st.caption("Tokens after each iteration")
        st.line_chart(naive["tokens_per_iter"])
        st.markdown("**Answer:**")
        st.write(naive["answer"])

    with col2:
        st.subheader("🦊 Engineered loop")
        st.metric(
            "Final tokens",
            f"{eng['final_tokens']:,}",
            delta=f"{eng['final_tokens'] - naive['final_tokens']:,} vs naive",
            delta_color="inverse",
        )
        st.metric(
            "Latency",
            f"{eng['latency']:.1f}s",
            delta=f"{eng['latency'] - naive['latency']:.1f}s vs naive",
            delta_color="inverse",
        )
        st.metric("Cost", f"${eng['cost']:.4f}")
        st.metric("Needle recalled", recall_badge(eng["needle_recalled"]))
        st.caption("Tokens after each iteration")
        st.line_chart(eng["tokens_per_iter"])
        st.markdown("**Answer:**")
        st.write(eng["answer"])

    # Combined growth curve
    st.divider()
    st.subheader("Context growth — naive vs engineered")
    growth = pd.DataFrame(
        {"naive": naive["tokens_per_iter"], "engineered": eng["tokens_per_iter"]},
        index=range(1, n + 1),
    )
    growth.index.name = "iteration"
    st.line_chart(growth)

    st.markdown("**Same task, same tools — only the context policy changed.**")
else:
    st.info("Pick N and press **Run both loops** to compare.")
