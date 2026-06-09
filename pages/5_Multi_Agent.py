"""
pages/5_Multi_Agent.py

Streamlit UI for Experiment 5 (Multi-Agent vs Single-Agent). Thin shell: call
the pure functions in experiments/exp05.py and render the headline -- the parent
context staying tiny while the single agent's window bloats -- plus the honest
latency tradeoff and the isolated per-sub-agent token usage.
"""

import sys
from pathlib import Path

# Ensure the project root is importable, whether launched via streamlit_app.py
# or directly via `streamlit run pages/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from experiments.exp05 import run_multi_agent, run_single_agent

st.set_page_config(page_title="Exp 5: Multi-Agent", layout="wide")

st.title("Experiment 5 — Multi-Agent vs Single-Agent")
st.markdown(
    """
**The problem:** one agent doing a multi-part task bloats its single context with
all the exploration. **The fix:** sub-agents with isolated windows that return
only summaries -- context compression by **architecture** (Section 8.3). From the
parent's view, a sub-agent is just a tool call that returns a summary.
"""
)

if st.button("▶ Run both", type="primary"):
    with st.spinner("Single agent (loads every paper into one window)..."):
        single = run_single_agent()
    with st.spinner("Multi-agent (3 isolated sub-agents + a composer)..."):
        multi = run_multi_agent()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐘 Single agent")
        st.metric("Peak context tokens", f"{single['peak_tokens']:,}")
        st.metric("Latency", f"{single['latency']:.1f}s")
        st.markdown("**Comparison:**")
        st.write(single["answer"])

    with col2:
        st.subheader("🦊 Multi-agent")
        st.metric(
            "PARENT context tokens",
            f"{multi['parent_peak_tokens']:,}",
            delta=f"{multi['parent_peak_tokens'] - single['peak_tokens']:,} vs single",
            delta_color="inverse",
        )
        st.metric(
            "Latency",
            f"{multi['latency']:.1f}s",
            delta="coordination overhead",
            delta_color="inverse",
        )
        st.caption(
            f"Sub-agent work stayed isolated: {multi['subagent_tokens']:,} tokens "
            "never entered the parent context."
        )
        st.markdown("**Comparison:**")
        st.write(multi["answer"])

    # The headline: parent window vs single window, plus the hidden sub-agent work.
    st.divider()
    st.subheader("Context peak — single window vs delegated parent")
    chart = pd.DataFrame(
        {"peak context tokens": [single["peak_tokens"], multi["parent_peak_tokens"]]},
        index=["single agent", "multi-agent parent"],
    )
    st.bar_chart(chart)

    st.markdown("**Each sub-agent's isolated window (never seen by the parent):**")
    usage = multi["subagent_usage"]
    if usage:
        sub = pd.DataFrame(
            [{"sub-agent": u["name"], "peak tokens": u["peak_tokens"]} for u in usage]
        ).set_index("sub-agent")
        st.bar_chart(sub)
    else:
        st.warning("No sub-agents ran this time.")

    st.divider()
    st.markdown(
        "**Honest tradeoff:** multi-agent keeps the parent context lean but usually "
        "costs latency. Use it when isolation and quality matter more than speed."
    )
else:
    st.info("Press **Run both** to compare a single window against delegated sub-agents.")
