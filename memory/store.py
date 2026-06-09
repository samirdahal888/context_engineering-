"""
memory/store.py

A tiny, file-based, human-inspectable note store -- the "external memory" for
Experiment 4. State lives OUTSIDE the context window (in data/memory/notes.json)
so it survives a complete session reset.

FRAMEWORK-AGNOSTIC: no Strands, no Streamlit imports here. The Strands @tool
wrappers live in tools/memory_tools.py and call into this module.

Structured (per research doc Section 8.2 -- organized notes, not a free-form
dump):

    {
      "task": "research brief on long-context techniques",
      "papers_processed": [
        {"id": "2307.03172", "title": "Lost in the Middle", "finding": "..."}
      ],
      "running_state": "2 of 4 papers done",
      "open_steps": ["read RULER", "write final combined brief"]
    }
"""

import json

from core.settings import settings


def _empty_notes() -> dict:
    """The blank structure used when no notes exist yet."""
    return {
        "task": "",
        "papers_processed": [],
        "running_state": "",
        "open_steps": [],
    }


def read_notes() -> dict:
    """Return the current notes, or an empty structure if none exist."""
    path = settings.notes_path
    if not path.exists():
        return _empty_notes()
    return json.loads(path.read_text(encoding="utf-8"))


def _persist(notes: dict) -> None:
    """Write notes to disk immediately, as indented (human-readable) JSON."""
    path = settings.notes_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notes, indent=2), encoding="utf-8")


def write_note(key: str, value) -> None:
    """Add/update one top-level note (e.g. 'task', 'running_state'). Persists now."""
    notes = read_notes()
    notes[key] = value
    _persist(notes)


def add_finding(paper_id: str, title: str, finding: str) -> None:
    """Append a per-paper finding to `papers_processed` (dedup by id). Persists now.

    Kept here (not in the tool) so the structured shape stays framework-agnostic.
    """
    notes = read_notes()
    papers = [p for p in notes.get("papers_processed", []) if p["id"] != paper_id]
    papers.append({"id": paper_id, "title": title, "finding": finding})
    notes["papers_processed"] = papers
    notes["running_state"] = f"{len(papers)} paper(s) recorded"
    _persist(notes)


def reset_memory() -> None:
    """Clear the store (used to reset between demo runs)."""
    path = settings.notes_path
    if path.exists():
        path.unlink()


__all__ = ["read_notes", "write_note", "add_finding", "reset_memory"]
