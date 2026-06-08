"""
ingest/download_corpus.py

The "book-fetching robot." Run once:

    python ingest/download_corpus.py

It downloads 7 arXiv papers, extracts their text to data/corpus/<id>.txt,
and writes data/index.json (the lightweight map the progressive agent reads).

If full PDF text extraction fails for a paper, it falls back to saving the
paper's abstract so the corpus is never empty.
"""

import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import arxiv
import fitz  # PyMuPDF

# Where this file lives -> project root is its parent's parent.
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_PATH = ROOT / "data" / "index.json"

# The 7 papers. (id, title, desc) -> desc is what the agent sees in the index.
PAPERS = [
    ("1706.03762", "Attention Is All You Need",
     "transformer architecture; self-attention; attention scores sum to 1.0"),
    ("2307.03172", "Lost in the Middle",
     "how LLMs use long contexts; U-shaped position performance curve"),
    ("2404.06654", "RULER",
     "measuring the real effective context length of long-context models"),
    ("2005.11401", "Retrieval-Augmented Generation",
     "RAG for knowledge-intensive NLP tasks"),
    ("2311.05232", "A Survey on Hallucination in LLMs",
     "taxonomy of hallucination including factuality vs faithfulness"),
    ("2302.12173", "Indirect Prompt Injection",
     "compromising LLM apps via malicious instructions in retrieved data"),
    ("2310.06452", "RLHF effects on diversity",
     "how RLHF reduces output diversity in language models"),
]


def extract_pdf_text(pdf_path: Path) -> str:
    """Open a PDF and scoop out all page text. Returns '' on failure."""
    try:
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"   ! PDF text extraction failed: {e}")
        return ""


def download_pdf(pdf_url: str, dest: Path) -> bool:
    """Fetch a PDF from arXiv to `dest`. Returns True on success."""
    # arXiv blocks requests without a real User-Agent header.
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "ce-lab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as e:
        print(f"   ! PDF download failed: {e}")
        return False


def fetch_one(result: arxiv.Result, arxiv_id: str) -> str:
    """Download the PDF for one paper and return its text (or abstract)."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / f"{arxiv_id}.pdf"
        if download_pdf(result.pdf_url, pdf_path):
            text = extract_pdf_text(pdf_path)
            if len(text) > 500:  # got real content
                return text
            print("   ! Extracted text too short, falling back to abstract.")
    # Fallback: the abstract is always available from the metadata.
    return f"{result.title}\n\n{result.summary}"


def main() -> int:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    ids = [p[0] for p in PAPERS]
    client = arxiv.Client()
    search = arxiv.Search(id_list=ids)

    # Map base-id -> arxiv result (arxiv returns ids like '1706.03762v7').
    results_by_id = {}
    for r in client.results(search):
        base = r.get_short_id().split("v")[0]
        results_by_id[base] = r

    index = []
    for arxiv_id, title, desc in PAPERS:
        print(f"-> {arxiv_id}  {title}")
        result = results_by_id.get(arxiv_id)
        out_path = CORPUS_DIR / f"{arxiv_id}.txt"

        if result is None:
            print("   ! NOT FOUND on arXiv; writing placeholder.")
            text = f"{title}\n\n(Paper {arxiv_id} could not be retrieved.)"
        else:
            text = fetch_one(result, arxiv_id)

        out_path.write_text(text, encoding="utf-8")
        words = len(text.split())
        print(f"   saved {out_path.name}  ({words:,} words)")

        index.append({
            "id": arxiv_id,
            "title": title,
            "desc": desc,
            "path": f"data/corpus/{arxiv_id}.txt",
        })

    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nWrote {INDEX_PATH.relative_to(ROOT)} with {len(index)} papers.")
    print("Done. Corpus ready in data/corpus/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
