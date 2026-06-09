"""
tools/memory_tools.py

Strands @tool wrappers that give an agent an EXTERNAL memory for Experiment 4.
These are thin: all structure/persistence lives in the framework-agnostic
memory/store.py. The docstrings are the "labels" the model reads to decide when
to call each tool.

  save_finding(paper_id, title, finding) -> persist one paper's finding to disk
  read_progress()                        -> read back everything saved so far
"""

from strands import tool

from memory import store


@tool
def save_finding(paper_id: str, title: str, finding: str) -> str:
    """Save a finding about one paper to external memory (persists to disk).

    Call this after reading each paper so your work survives even if this
    session ends. One call per paper.

    Args:
        paper_id: The arXiv ID of the paper, e.g. "2307.03172".
        title: The paper's title.
        finding: A 1-3 sentence summary of what this paper contributes.
    """
    store.add_finding(paper_id, title, finding)
    return f"Saved finding for {paper_id} ({title}) to external memory."


@tool
def read_progress() -> str:
    """Read everything saved in external memory so far (from a prior session).

    Call this FIRST at the start of a session to recover earlier work before
    continuing the task.
    """
    notes = store.read_notes()
    papers = notes.get("papers_processed", [])
    if not papers:
        return "External memory is empty. No prior findings have been saved."

    lines = [f"Recovered {len(papers)} prior finding(s) from external memory:"]
    for p in papers:
        lines.append(f"- [{p['id']}] {p['title']}: {p['finding']}")
    open_steps = notes.get("open_steps") or []
    if open_steps:
        lines.append(f"Open steps: {', '.join(open_steps)}")
    return "\n".join(lines)


__all__ = ["save_finding", "read_progress"]
