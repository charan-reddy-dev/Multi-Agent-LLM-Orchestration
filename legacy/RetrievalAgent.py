"""
Retrieval-Augmented Agent.
Performs multi-hop reasoning across at least two retrieved chunks before forming an answer.
Single-hop retrieval is NOT sufficient — the agent must chain reasoning across chunks.
Each chunk contribution is cited explicitly.
"""
from __future__ import annotations
import time
import uuid
import logging
import json
from typing import List, Dict, Any, Optional
from anthropic import Anthropic

from agents.context_schema import (
    AgentID, AgentMessage, Claim, SharedContext, SubTask
)
from context_manager.budget_manager import ContextBudgetManager, count_tokens
from tools.tools import dispatch_tool

logger = logging.getLogger(__name__)
client = Anthropic()

# Simulated vector store — in production replace with real embeddings + /Pinecone
KNOWLEDGE_CHUNKS = [
    {"chunk_id": "c001", "text": "Multi-agent systems allow parallelization of complex tasks by distributing work across specialized agents.", "source": "AutoGen Paper 2023"},
    {"chunk_id": "c002", "text": "Orchestrator patterns in LLM systems require careful context passing to avoid information loss between agent calls.", "source": "LangGraph Docs"},
    {"chunk_id": "c003", "text": "RAG improves factual accuracy by grounding LLM responses in retrieved document chunks rather than relying solely on parametric memory.", "source": "RAG Paper 2020"},
    {"chunk_id": "c004", "text": "Token budget management is critical in multi-agent pipelines; agents must track their own consumption to avoid context overflow.", "source": "Production LLM Systems Guide"},
    {"chunk_id": "c005", "text": "Critique agents improve output quality by identifying specific flawed claims rather than rejecting outputs wholesale.", "source": "Constitutional AI Paper"},
    {"chunk_id": "c006", "text": "Streaming SSE (Server-Sent Events) enables real-time token delivery from LLM APIs to browser clients.", "source": "FastAPI SSE Docs"},
    {"chunk_id": "c007", "text": "Evaluation harnesses for LLMs must include adversarial cases to test robustness against prompt injection and false premises.", "source": "HELM Evaluation Framework"},
    {"chunk_id": "c008", "text": "Self-improving prompt loops allow systems to identify underperforming prompts and propose rewrites, closing the feedback loop.", "source": "DSPy Paper 2023"},
    {"chunk_id": "c009", "text": "Docker Compose enables reproducible multi-service deployments with a single command, critical for production ML systems.", "source": "Docker Docs"},
    {"chunk_id": "c010", "text": "Dependency graphs in task decomposition ensure that downstream tasks only execute after their upstream dependencies are resolved.", "source": "DAG Systems Survey"},
]

RETRIEVAL_SYSTEM = """You are a Retrieval-Augmented Agent. You receive a query and multiple retrieved text chunks.
You MUST perform multi-hop reasoning — use at least 2 chunks to build your answer.
Cite each chunk explicitly in your answer using [chunk_id].

Output ONLY valid JSON:
{
  "answer": "Your full answer with inline citations like [c001]",
  "hop_reasoning": [
    {"hop": 1, "chunk_id": "c001", "contribution": "what this chunk contributed"},
    {"hop": 2, "chunk_id": "c003", "contribution": "what this chunk contributed"}
  ],
  "confidence": 0.85
}
"""


def _retrieve_chunks(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Simple keyword-based retrieval. Replace with vector search in production."""
    query_words = set(query.lower().split())
    scored = []
    for chunk in KNOWLEDGE_CHUNKS:
        chunk_words = set(chunk["text"].lower().split())
        overlap = len(query_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


def run(ctx: SharedContext, budget_mgr: ContextBudgetManager,
        subtask: Optional[SubTask] = None) -> AgentMessage:

    budget_mgr.declare_budget(AgentID.RETRIEVAL, 6000)

    query = subtask.query if subtask else ctx.original_query

    # Step 1: Retrieve chunks
    chunks = _retrieve_chunks(query, top_k=4)
    if len(chunks) < 2:
        chunks = KNOWLEDGE_CHUNKS[:4]  # ensure at least 2 chunks

    chunks_text = "\n\n".join(
        f"[{c['chunk_id']}] (Source: {c['source']})\n{c['text']}"
        for c in chunks
    )

    # Step 2: Also call web_search tool for additional context
    search_result = dispatch_tool("web_search", query=query)
    web_context = ""
    if search_result.success and search_result.data:
        web_context = "\n\nWeb Search Results:\n" + "\n".join(
            f"- {r['title']}: {r['snippet']} ({r['url']})"
            for r in search_result.data[:3]
        )

    prompt = f"""Query: {query}

Retrieved Chunks:
{chunks_text}
{web_context}

Perform multi-hop reasoning using at least 2 chunks. Cite contributions explicitly."""

    if not budget_mgr.check_budget(AgentID.RETRIEVAL, prompt):
        logger.warning("[Retrieval] Budget exceeded")
        return AgentMessage(sender=AgentID.RETRIEVAL, content="BUDGET_EXCEEDED",
                            metadata={"error": "budget exceeded"})

    t0 = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=RETRIEVAL_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    latency = (time.time() - t0) * 1000

    raw = response.content[0].text.strip()
    tokens_used = count_tokens(prompt + raw)
    budget_mgr.consume(AgentID.RETRIEVAL, prompt + raw)

    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer", raw)
        hop_reasoning = parsed.get("hop_reasoning", [])
        confidence = parsed.get("confidence", 0.8)
    except json.JSONDecodeError:
        answer = raw
        hop_reasoning = []
        confidence = 0.7

    # Build claims with chunk citations
    claims = []
    for hop in hop_reasoning:
        claim = Claim(
            claim_id=str(uuid.uuid4()),
            text=hop.get("contribution", ""),
            source_agent=AgentID.RETRIEVAL,
            source_chunk=hop.get("chunk_id"),
            confidence=confidence,
        )
        claims.append(claim)
        ctx.claims.append(claim)

    # Mark subtask done
    if subtask:
        subtask.status_done(answer)

    msg = AgentMessage(
        sender=AgentID.RETRIEVAL,
        content=answer,
        claims=claims,
        tokens_used=tokens_used,
        metadata={
            "chunks_retrieved": [c["chunk_id"] for c in chunks],
            "hop_count": len(hop_reasoning),
            "confidence": confidence,
            "latency_ms": latency,
            "subtask_id": subtask.task_id if subtask else None,
        }
    )
    ctx.add_message(msg)
    logger.info(f"[Retrieval] {len(hop_reasoning)}-hop answer, confidence={confidence}")
    return msg