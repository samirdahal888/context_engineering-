"""
core/settings.py

Single source of truth for configuration. Everything tunable lives here as a
typed Pydantic Settings object, populated from environment variables / .env.

Usage:
    from core.settings import settings
    settings.bedrock_model_id
    settings.naive_max_words

Note: importing this module also loads .env into the process environment via
python-dotenv, so AWS credentials (AWS_ACCESS_KEY_ID, ...) become visible to
boto3, which reads them from os.environ rather than from this object.
"""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = this file's parent's parent (core/ -> repo root).
ROOT = Path(__file__).resolve().parent.parent

# Make .env values (incl. AWS creds) available to boto3 via os.environ.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    """Typed application configuration, loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Bedrock model ---
    bedrock_model_id: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    aws_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("AWS_REGION", "BEDROCK_REGION"),
    )

    # --- Generation ---
    output_max_tokens: int = 1000
    temperature: float = 0.0

    # --- Pricing (USD per 1M tokens) for cost estimates ---
    price_in_per_1m: float = 0.80
    price_out_per_1m: float = 4.00

    # --- Naive arm: how much corpus to send before the context window fills ---
    # The window is measured in tokens; dense papers run ~2 tokens/word.
    naive_max_tokens: int = 190000
    tokens_per_word: float = 2.05

    # --- load_document tool: per-document truncation ---
    max_document_words: int = 12_000

    # --- Exp 5 multi-agent: hard cap so each sub-agent returns a concise summary
    # (~300 words) instead of dumping its raw exploration. ---
    subagent_summary_tokens: int = 450

    # --- Exp 3 compaction (a small DEMO window so overflow is fast + visible,
    # NOT a real 200k window). All real token counts still come from the API. ---
    compaction_window_tokens: int = 6000     # the demo context ceiling
    compaction_threshold: float = 0.8        # compact at 80% full
    per_paper_tokens: int = 1200             # how much of each paper we feed per step
    loop_output_tokens: int = 300            # max output per running-synthesis step
    summary_output_tokens: int = 500         # max output for the compaction summary

    # --- Derived paths (not configurable) ---
    @property
    def root(self) -> Path:
        return ROOT

    @property
    def corpus_dir(self) -> Path:
        return ROOT / "data" / "corpus"

    @property
    def index_path(self) -> Path:
        return ROOT / "data" / "index.json"

    @property
    def memory_dir(self) -> Path:
        """Exp 4: external memory lives here, OUTSIDE the context window."""
        return ROOT / "data" / "memory"

    @property
    def notes_path(self) -> Path:
        return self.memory_dir / "notes.json"

    # --- Derived values ---
    @property
    def naive_max_words(self) -> int:
        """Word budget for the naive blob, derived from the token budget."""
        return int(self.naive_max_tokens / self.tokens_per_word)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()
