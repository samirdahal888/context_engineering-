"""
tools/load_document.py

A Strands tool the progressive agent can call to fetch ONE paper's full text
by its arXiv id. The docstring below is read by the model -- it's the "label"
on the hand, telling the agent when and how to use it.
"""

from strands import tool

from core.settings import settings


@tool
def load_document(doc_id: str) -> str:
    """Load the full text of a research paper by its arXiv ID.

    Use this when you need to read a specific paper to answer a question.
    Returns the paper text, truncated to 12000 words if necessary.

    Args:
        doc_id: The arXiv ID of the paper, e.g. "1706.03762".
    """
    corpus_dir = settings.corpus_dir
    path = corpus_dir / f"{doc_id}.txt"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in corpus_dir.glob("*.txt")))
        return (
            f"ERROR: No document found with id '{doc_id}'. "
            f"Available document ids are: {available}"
        )

    max_words = settings.max_document_words
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
        text += f"\n\n[...TRUNCATED: showing first {max_words} of {len(words)} words...]"
    return text
