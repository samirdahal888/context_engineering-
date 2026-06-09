"""
pages/4_External_Memory.py

Streamlit UI for Experiment 4 (External Memory / Note-Taking). Thin shell: call
the pure functions in experiments/exp04.py and render the coverage difference
plus the visible external memory (notes.json) that survived a session reset.
"""

import sys
from pathlib import Path

# Ensure the project root is importable, whether launched via streamlit_app.py
# or directly via `streamlit run pages/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from experiments.exp04 import ALL_PAPERS, run_engineered, run_naive

st.set_page_config(page_title="Exp 4: External Memory", layout="wide")

st.title("Experiment 4 — External Memory / Note-Taking")
st.markdown(
    """
**The problem:** LLMs are stateless. When a session ends, its memory is gone --
a new session starts blank, repeating work or losing earlier decisions.
**The fix:** write notes to disk; read them next session. State lives **outside**
the context window, so it survives even a full reset.
**The demo runs the task as TWO sessions with a hard reset between them.**
"""
)

n_total = len(ALL_PAPERS)

if st.button("▶ Run both (two sessions each, with reset)", type="primary"):
    with st.spinner("Naive: two isolated sessions (no shared memory)..."):
        naive = run_naive()
    with st.spinner("Engineered: two sessions sharing notes.json..."):
        eng = run_engineered()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🐘 Naive — no external memory")
        st.metric("Papers covered in final brief", f"{len(naive['papers_covered'])} / {n_total}")
        st.caption("Session 2 never saw session 1's work.")
        st.markdown("**Final brief:**")
        st.write(naive["final_brief"])

    with col2:
        st.subheader("🦊 Engineered — external notes")
        st.metric(
            "Papers covered in final brief",
            f"{len(eng['papers_covered'])} / {n_total}",
            delta=f"+{len(eng['papers_covered']) - len(naive['papers_covered'])} vs naive",
        )
        st.markdown("**The external memory (notes.json) session 2 read:**")
        st.json(eng["notes_snapshot"])
        st.markdown("**Final brief (synthesized from the notes):**")
        st.write(eng["final_brief"])

    st.divider()
    st.markdown(
        "**The engineered agent survived a full session reset** because its state "
        "lived outside the context window, on disk -- a structured, human-inspectable "
        "`notes.json` it could read back and resume from."
    )
else:
    st.info("Press **Run both** to watch external memory survive a session reset.")
