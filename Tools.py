"""
Tool implementations with explicit failure contracts.
Each tool defines what it returns on: timeout, empty results, malformed input.
Tool calls are logged with input, output, latency, and agent acceptance status.
"""
from __future__ import annotations
import time
import uuid
import sqlite3
import subprocess
import tempfile
import os
import json
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/app/data/agent_store.db")


# ──────────────────────────────────────────────
# Failure contract types
# ──────────────────────────────────────────────
@dataclass
class ToolResult:
    success: bool
    data: Any
    error_type: Optional[str]   # "timeout" | "empty" | "malformed" | None
    error_msg: Optional[str]
    latency_ms: float
    tool_name: str
    call_id: str = ""

    def __post_init__(self):
        if not self.call_id:
            self.call_id = str(uuid.uuid4())


# ──────────────────────────────────────────────
# Tool 1: Web Search Stub
# Returns structured results with source URLs and relevance scores.
# Failure contract:
#   timeout     → success=False, error_type="timeout", data=[]
#   empty       → success=True,  error_type="empty",   data=[]
#   malformed   → success=False, error_type="malformed", data=[]
# ──────────────────────────────────────────────
MOCK_SEARCH_DB = {
    "multi-agent systems": [
        {"url": "https://arxiv.org/abs/2309.07864", "title": "AutoGen: Multi-Agent Framework", "snippet": "AutoGen enables multi-agent conversations for complex tasks.", "relevance": 0.95},
        {"url": "https://langchain.com/docs/agents", "title": "LangChain Agents", "snippet": "LangChain provides tools for building LLM-powered agents.", "relevance": 0.88},
    ],
    "rag retrieval augmented generation": [
        {"url": "https://arxiv.org/abs/2005.11401", "title": "RAG: Retrieval-Augmented Generation", "snippet": "RAG combines parametric and non-parametric memory for NLP.", "relevance": 0.97},
        {"url": "https://www.pinecone.io/learn/rag", "title": "RAG with Vector DBs", "snippet": "Pinecone explains RAG with vector database integration.", "relevance": 0.82},
    ],
    "context window management llm": [
        {"url": "https://arxiv.org/abs/2307.03172", "title": "Lost in the Middle", "snippet": "LLMs struggle with information in the middle of long contexts.", "relevance": 0.91},
    ],
    "default": [
        {"url": "https://example.com/result1", "title": "General Result 1", "snippet": "A relevant result for the query.", "relevance": 0.70},
        {"url": "https://example.com/result2", "title": "General Result 2", "snippet": "Another relevant result.", "relevance": 0.65},
    ]
}

def web_search(query: str, timeout: float = 5.0) -> ToolResult:
    t0 = time.time()
    try:
        if not isinstance(query, str):
            return ToolResult(False, [], "malformed", "Query must be a string", (time.time()-t0)*1000, "web_search")
        if not query.strip():
            return ToolResult(True, [], "empty", "Empty query string", (time.time()-t0)*1000, "web_search")

        # Stub: match against mock DB
        results = MOCK_SEARCH_DB.get("default", [])
        for key in MOCK_SEARCH_DB:
            if any(word in query.lower() for word in key.split()):
                results = MOCK_SEARCH_DB[key]
                break

        # Simulate occasional timeout
        elapsed = (time.time() - t0) * 1000
        if elapsed > timeout * 1000:
            return ToolResult(False, [], "timeout", f"Search timed out after {timeout}s", elapsed, "web_search")

        if not results:
            return ToolResult(True, [], "empty", "No results found", elapsed, "web_search")

        return ToolResult(True, results, None, None, elapsed, "web_search")
    except Exception as e:
        return ToolResult(False, [], "malformed", str(e), (time.time()-t0)*1000, "web_search")


# ──────────────────────────────────────────────
# Tool 2: Code Execution Sandbox
# Runs Python snippets, returns stdout, stderr, exit_code.
# Failure contract:
#   timeout   → exit_code=-1, stderr="Execution timed out"
#   empty     → stdout="", stderr="", exit_code=0
#   malformed → exit_code=-2, stderr="Invalid code input"
# ──────────────────────────────────────────────
BANNED_IMPORTS = ["os.system", "subprocess", "shutil.rmtree", "__import__('os').system"]

def code_executor(code: str, timeout: int = 10) -> ToolResult:
    t0 = time.time()
    try:
        if not isinstance(code, str):
            return ToolResult(False, {"stdout": "", "stderr": "Invalid code input", "exit_code": -2},
                              "malformed", "Code must be a string", (time.time()-t0)*1000, "code_executor")
        if not code.strip():
            return ToolResult(True, {"stdout": "", "stderr": "", "exit_code": 0},
                              "empty", "Empty code", (time.time()-t0)*1000, "code_executor")

        # Basic safety filter
        for banned in BANNED_IMPORTS:
            if banned in code:
                return ToolResult(False,
                    {"stdout": "", "stderr": f"Banned operation: {banned}", "exit_code": -3},
                    "malformed", "Unsafe code detected", (time.time()-t0)*1000, "code_executor")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            fname = f.name

        try:
            proc = subprocess.run(
                ["python3", fname],
                capture_output=True, text=True, timeout=timeout
            )
            elapsed = (time.time() - t0) * 1000
            return ToolResult(
                proc.returncode == 0,
                {"stdout": proc.stdout, "stderr": proc.stderr, "exit_code": proc.returncode},
                None, None, elapsed, "code_executor"
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.time() - t0) * 1000
            return ToolResult(False,
                {"stdout": "", "stderr": f"Execution timed out after {timeout}s", "exit_code": -1},
                "timeout", "Timeout", elapsed, "code_executor")
        finally:
            os.unlink(fname)

    except Exception as e:
        return ToolResult(False,
            {"stdout": "", "stderr": str(e), "exit_code": -99},
            "malformed", str(e), (time.time()-t0)*1000, "code_executor")


# ──────────────────────────────────────────────
# Tool 3: Structured Data Lookup (NL → SQL)
# Queries a local SQLite DB. The agent converts natural language to SQL.
# Failure contract:
#   timeout   → success=False, error_type="timeout"
#   empty     → success=True,  data=[], error_type="empty"
#   malformed → success=False, error_type="malformed"
# ──────────────────────────────────────────────
NL_TO_SQL_MAP = {
    "eval runs": "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT 10",
    "failed test": "SELECT * FROM eval_results WHERE score < 0.5 ORDER BY created_at DESC LIMIT 10",
    "prompt rewrites": "SELECT * FROM prompt_rewrites ORDER BY created_at DESC LIMIT 10",
    "policy violations": "SELECT * FROM execution_log WHERE event_type='policy_violation' LIMIT 20",
    "tool calls": "SELECT * FROM execution_log WHERE event_type='tool_call' LIMIT 20",
    "job": "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 5",
}

def _nl_to_sql(natural_query: str) -> Optional[str]:
    """Simple NL→SQL mapping. In production, this would call an LLM."""
    nl_lower = natural_query.lower()
    for key, sql in NL_TO_SQL_MAP.items():
        if key in nl_lower:
            return sql
    # Fallback: try to extract table name
    for table in ["jobs", "eval_runs", "eval_results", "prompt_rewrites", "execution_log"]:
        if table.replace("_", " ") in nl_lower or table in nl_lower:
            return f"SELECT * FROM {table} LIMIT 10"
    return None

def data_lookup(natural_query: str, timeout: float = 5.0) -> ToolResult:
    t0 = time.time()
    try:
        if not isinstance(natural_query, str) or not natural_query.strip():
            return ToolResult(False, [], "malformed", "Query must be a non-empty string",
                              (time.time()-t0)*1000, "data_lookup")

        sql = _nl_to_sql(natural_query)
        if not sql:
            return ToolResult(True, [], "empty",
                              f"Could not map '{natural_query}' to SQL",
                              (time.time()-t0)*1000, "data_lookup")

        db_path = DB_PATH
        if not os.path.exists(db_path):
            # Return empty gracefully if DB not yet initialized
            return ToolResult(True, [], "empty", "Database not yet initialized",
                              (time.time()-t0)*1000, "data_lookup")

        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
            elapsed = (time.time() - t0) * 1000
            if not rows:
                return ToolResult(True, [], "empty", "Query returned no rows", elapsed, "data_lookup")
            return ToolResult(True, rows, None, None, elapsed, "data_lookup")
        finally:
            conn.close()

    except sqlite3.OperationalError as e:
        elapsed = (time.time() - t0) * 1000
        if "timeout" in str(e).lower():
            return ToolResult(False, [], "timeout", str(e), elapsed, "data_lookup")
        return ToolResult(False, [], "malformed", str(e), elapsed, "data_lookup")
    except Exception as e:
        return ToolResult(False, [], "malformed", str(e), (time.time()-t0)*1000, "data_lookup")


# ──────────────────────────────────────────────
# Tool 4: Self-Reflection Tool
# Agent re-reads its own previous outputs, identifies contradictions.
# Failure contract:
#   empty     → no prior outputs found
#   malformed → invalid agent_id
# ──────────────────────────────────────────────
def self_reflection(agent_id: str, ctx_messages: list) -> ToolResult:
    """
    Scans the agent's previous outputs in the session for contradictions.
    Returns a list of detected contradiction pairs.
    """
    t0 = time.time()
    try:
        if not isinstance(agent_id, str) or not agent_id.strip():
            return ToolResult(False, {}, "malformed", "agent_id must be a non-empty string",
                              (time.time()-t0)*1000, "self_reflection")

        agent_outputs = [m for m in ctx_messages if m.get("sender") == agent_id]
        if not agent_outputs:
            return ToolResult(True, {"contradictions": [], "outputs_reviewed": 0},
                              "empty", f"No prior outputs for agent {agent_id}",
                              (time.time()-t0)*1000, "self_reflection")

        # Simple contradiction detection: look for opposing keywords
        CONTRADICTION_PAIRS = [
            ("always", "never"), ("true", "false"), ("yes", "no"),
            ("increases", "decreases"), ("positive", "negative"),
            ("correct", "incorrect"), ("valid", "invalid"),
        ]
        contradictions = []
        contents = [m.get("content", "") for m in agent_outputs]
        for i, c1 in enumerate(contents):
            for j, c2 in enumerate(contents):
                if i >= j:
                    continue
                for w1, w2 in CONTRADICTION_PAIRS:
                    if w1 in c1.lower() and w2 in c2.lower():
                        contradictions.append({
                            "output_1_index": i,
                            "output_2_index": j,
                            "word_1": w1,
                            "word_2": w2,
                            "snippet_1": c1[:200],
                            "snippet_2": c2[:200],
                        })

        elapsed = (time.time() - t0) * 1000
        return ToolResult(True, {
            "contradictions": contradictions,
            "outputs_reviewed": len(agent_outputs),
            "contradiction_count": len(contradictions),
        }, None, None, elapsed, "self_reflection")

    except Exception as e:
        return ToolResult(False, {}, "malformed", str(e), (time.time()-t0)*1000, "self_reflection")


# ──────────────────────────────────────────────
# Tool dispatcher (used by orchestrator)
# ──────────────────────────────────────────────
TOOL_REGISTRY = {
    "web_search": web_search,
    "code_executor": code_executor,
    "data_lookup": data_lookup,
    "self_reflection": self_reflection,
}

def dispatch_tool(tool_name: str, **kwargs) -> ToolResult:
    if tool_name not in TOOL_REGISTRY:
        return ToolResult(False, {}, "malformed",
                          f"Unknown tool: {tool_name}", 0.0, tool_name)
    return TOOL_REGISTRY[tool_name](**kwargs)