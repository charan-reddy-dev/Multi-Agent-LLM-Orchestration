"""
Compression Agent.
Triggered when an agent's context approaches its budget limit.
Lossless for structured data (tool outputs, scores, citations).
Lossy only for conversational filler.
"""
from __future__ import annotations
import time
import logging
import json
from anthropic import Anthropic
from agents.context_schema import AgentID, AgentMessage, SharedContext
from context_manager.budget_manager import ContextBudgetManager, count_tokens

logger = logging.getLogger(__name__)
client = Anthropic()

COMPRESSION_SYSTEM = """You are a Compression Agent. Compress conversational context while preserving critical information.

Rules:
- LOSSLESS: Keep all structured data (JSON, scores, citations, tool outputs, claim IDs, chunk IDs)
- LOSSLESS: Keep all numbered facts, metrics, and specific values
- LOSSY: Compress conversational filler, repetitive explanations, and verbose prose
- Output a compressed summary that retains all factual content

Output format:
{
  "compressed_text": "The compressed version",
  "tokens_saved": 150,
  "preserved_structured": ["item1", "item2"],
  "removed_filler": ["removed phrase 1"]
}
"""


def run(ctx: SharedContext, budget_mgr: ContextBudgetManager,
        target_agent: AgentID = None, text_to_compress: str = "") -> AgentMessage:

    budget_mgr.declare_budget(AgentID.COMPRESSION, 2000)

    if not text_to_compress:
        # Auto-gather most verbose agent outputs
        all_content = []
        for msg in ctx.messages[-5:]:  # last 5 messages
            if msg.sender != AgentID.COMPRESSION:
                all_content.append(f"[{msg.sender.value}]: {msg.content}")
        text_to_compress = "\n\n".join(all_content)

    original_tokens = count_tokens(text_to_compress)
    prompt = f"Compress this context (currently {original_tokens} tokens):\n\n{text_to_compress[:3000]}"

    t0 = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        system=COMPRESSION_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - t0) * 1000
    raw = response.content[0].text.strip()

    try:
        parsed = json.loads(raw)
        compressed = parsed.get("compressed_text", raw)
        tokens_saved = parsed.get("tokens_saved", 0)
    except json.JSONDecodeError:
        compressed = raw
        tokens_saved = max(0, original_tokens - count_tokens(raw))

    # Reset token usage for target agent after compression
    if target_agent:
        new_usage = count_tokens(compressed)
        ctx.token_used[target_agent] = new_usage
        logger.info(f"[Compression] Reset {target_agent.value} token usage: "
                    f"{original_tokens} → {new_usage} (saved {tokens_saved})")

    msg = AgentMessage(
        sender=AgentID.COMPRESSION,
        content=compressed,
        tokens_used=count_tokens(prompt + raw),
        metadata={
            "original_tokens": original_tokens,
            "compressed_tokens": count_tokens(compressed),
            "tokens_saved": tokens_saved,
            "target_agent": target_agent.value if target_agent else "general",
            "latency_ms": latency,
        }
    )
    ctx.add_message(msg)
    return msg