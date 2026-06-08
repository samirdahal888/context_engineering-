"""
pages/1_Progressive_Disclosure.py

Streamlit UI for Experiment 1. This is a thin shell: it only collects a query,
calls the pure functions in experiments/exp01.py, and renders the results
(especially the Token X-ray). No experiment logic lives here.
"""

import sys
from pathlib import Path

# Ensure the project root is importable, so this page works whether launched via
# `streamlit run streamlit_app.py` or directly via `streamlit run pages/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from experiments.exp01 import run_naive, run_progressive

st.set_page_config(page_title="Exp 1: Progressive Disclosure", layout="wide")

st.title("Experiment 1 — Progressive Disclosure vs Load-Everything")
st.markdown(
    """
**The question:** Why not just put all documents into the context?
**Watch:** the Token X-ray shows exactly where every token goes.
"""
)


def token_xray(breakdown: dict[str, int]) -> None:
    """Draw a horizontal bar of where tokens go (largest at top)."""
    df = (
        pd.DataFrame({"tokens": breakdown})
        .sort_values("tokens", ascending=True)
    )
    st.bar_chart(df, horizontal=True)


query = st.text_input("Query", value="What do attention scores sum to, and why?")

if st.button("▶ Run both arms", type="primary"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐘 Naive — load everything")
        with st.spinner("Sending all papers to the model..."):
            naive = run_naive(query)
        st.metric("Input tokens", f"{naive['input_tokens']:,}")
        st.metric("Latency", f"{naive['latency']:.1f}s")
        if naive["truncated"]:
            st.warning("Too big to fit — some papers were dropped to fit the window.")
        st.caption("Token X-ray — tokens per paper")
        token_xray(naive["breakdown"])
        st.markdown("**Answer:**")
        st.write(naive["answer"])

    with col2:
        st.subheader("🦊 Progressive disclosure")
        with st.spinner("Agent picks and loads one paper..."):
            prog = run_progressive(query)
        st.metric(
            "Total tokens",
            f"{prog['total_tokens']:,}",
            delta=f"{prog['total_tokens'] - naive['input_tokens']:,} vs naive",
            delta_color="inverse",
        )
        st.metric("Latency", f"{prog['latency']:.1f}s")
        st.success(f"Loaded only: {prog['doc_loaded']}")
        st.markdown("**Agent steps:**")
        for step in prog["steps"]:
            st.info(step)
        st.caption("Token X-ray — tokens per segment")
        token_xray(prog["breakdown"])
        st.markdown("**Answer:**")
        st.write(prog["answer"])

    # Metrics strip
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Token reduction", f"{naive['input_tokens'] / prog['total_tokens']:.1f}×")
    m2.metric("Naive tokens", f"{naive['input_tokens']:,}")
    m3.metric("Progressive tokens", f"{prog['total_tokens']:,}")
else:
    st.info("Enter a query and press **Run both arms** to compare.")
