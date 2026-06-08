"""
core/tokenizer.py

count_tokens(text) -> int

HOW WE COUNT (the real way)
---------------------------
AWS Bedrock exposes a `CountTokens` API that runs the *actual* Anthropic Claude
tokenizer on the server. We call it directly, so the number we get back is the
real, exact token count for the Claude model you have enabled -- not a guess.

    client.count_tokens(
        modelId=<your Claude model id>,
        input={"converse": {"messages": [{"role": "user",
                                          "content": [{"text": text}]}]}},
    ) -> {"inputTokens": <exact int>}

THE SPARE TIRE (fallback)
-------------------------
If no AWS credentials are configured yet (e.g. while you're still building the
lab offline), calling AWS would crash. Rather than crash, we fall back to a
documented approximation -- words * 1.3 -- and print a clear warning so you
ALWAYS know when a number is estimated vs. real. This fallback is never silent.

Model id and region come from core.settings (which loads .env). Note older
Claude 3 models do NOT support CountTokens, so this falls back to the estimate
on those; newer Sonnet/Claude-4 models support it (exact counts).
"""

import sys
import functools

from core.settings import settings

# Approximate tokens-per-word, used ONLY as an offline fallback.
_TOKENS_PER_WORD = 1.3

_warned_once = False


@functools.lru_cache(maxsize=1)
def _client():
    """Lazily build one cached bedrock-runtime client."""
    import boto3
    return boto3.client("bedrock-runtime", region_name=settings.aws_region)


def _approximate(text: str) -> int:
    """Offline spare tire: words * 1.3. Clearly an estimate."""
    global _warned_once
    if not _warned_once:
        print(
            "[tokenizer] WARNING: AWS CountTokens unavailable -- using the "
            "words*1.3 ESTIMATE. Numbers are approximate until AWS creds work.",
            file=sys.stderr,
        )
        _warned_once = True
    return int(round(len(text.split()) * _TOKENS_PER_WORD))


def count_tokens(text: str) -> int:
    """Return the token count of `text`.

    Tries the real Bedrock CountTokens API first (exact). Falls back to a
    words*1.3 estimate (with a one-time warning) only if AWS is unreachable.
    """
    if not text:
        return 0
    try:
        resp = _client().count_tokens(
            modelId=settings.bedrock_model_id,
            input={
                "converse": {
                    "messages": [{"role": "user", "content": [{"text": text}]}]
                }
            },
        )
        return int(resp["inputTokens"])
    except Exception:
        # No creds / offline / model not enabled -> graceful, loud fallback.
        return _approximate(text)
