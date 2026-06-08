"""
streamlit_app.py

The lab's landing page and multipage entry point. Run the whole lab with:

    streamlit run streamlit_app.py

Because this script lives at the project root, Streamlit puts the root on the
import path, so every page under pages/ can import experiments/, core/, tools/.
"""

import streamlit as st

st.set_page_config(page_title="Context Engineering Lab", layout="wide")

st.title("Context Engineering Lab")
st.markdown(
    """
A hands-on platform for observing context engineering problems and solutions.

> "Context engineering is curating what the model sees so that you get a better result."
> — Birgitta Böckeler, Thoughtworks, 2026

---
### What is context engineering?

LLMs are stateless. Every call starts from zero. The context window is the model's entire world
for one inference — and the **attention budget** is finite. Context engineering is the discipline
of deciding what goes in that window, in what order, and when.

**This lab has 10 experiments.** Each one shows a specific failure, then the fix.
Use the sidebar to navigate.

---
### Experiments
"""
)

experiments = [
    ("1 — Progressive Disclosure", "Why loading everything hurts accuracy"),
    ("2 — Context Rot", "How context degrades as the agent loop runs"),
    ("3 — Compaction", "Surviving context overflow on long tasks"),
    ("4 — External Memory", "Remembering across session resets"),
    ("5 — Multi-Agent", "Context isolation through sub-agents"),
    ("6 — Tool Explosion", "When tool outputs flood the window"),
    ("7 — Long-Running Harness", "Agents that survive session boundaries"),
    ("8 — Lost in the Middle", "Why position in context matters"),
    ("9 — Prompt Injection", "Security: attacking and defending context"),
    ("10 — System Prompt Design", "The Goldilocks zone for instructions"),
]

for name, desc in experiments:
    st.markdown(f"**{name}** — {desc}")
