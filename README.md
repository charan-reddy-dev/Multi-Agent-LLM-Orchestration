# Real-Time Multi-Agent LLM Orchestration System

A containerized, production-grade multi-agent system with a self-improving evaluation loop, dynamic tool orchestration, and adversarial robustness testing.

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/multi-agent-llm-orchestration
cd multi-agent-llm-orchestration

# 2. Set up environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Start everything
docker compose up

# API available at http://localhost:8000
# Log interface at http://localhost:8001
# API docs at http://localhost:8000/docs
```

That's it. Zero manual steps beyond adding your API key.

---

## Architecture

See `architecture.txt` for the full ASCII diagram.

The system is organized around a **Master Orchestrator** that dynamically routes tasks to specialized sub-agents via a **SharedContext** object. Agents never communicate directly — all handoffs go through the orchestrator.

```
Client → FastAPI → Orchestrator → [Decomp | Retrieval | Critique | Synthesis | Compression]
                                          ↕
                                   SharedContext (schema-defined)
                                          ↕
                                       Tools ← [web_search | code_executor | data_lookup | self_reflection]
                                          ↕
                                       SQLite DB
```

---

## Agent Descriptions & Decision Boundaries

### Master Orchestrator (`agents/orchestrator.py`)
The orchestrator uses the LLM itself to decide routing at every step — not a hardcoded chain. It receives the current pipeline state (which agents have run, budget status, flagged claims) and asks the LLM which agent should run next.

**Decision boundaries:**
- Will trigger COMPRESSION before any agent that is near its budget limit
- Will stop routing to RETRIEVAL once all subtasks are resolved
- Will not skip CRITIQUE before SYNTHESIS
- Hard ceiling of 8 routing steps to prevent infinite loops
- Logs every routing decision with structured justification

### Decomposition Agent (`agents/decomposition_agent.py`)
Breaks ambiguous queries into typed sub-tasks with explicit dependency graphs.

**Decision boundaries:**
- Produces tasks of types: `factual`, `analytical`, `comparative`, `procedural`, `evaluative`
- Enforces that dependent tasks have `dependencies` populated
- Does NOT execute tasks — only defines them
- Outputs structured JSON only (no prose)

### Retrieval Agent (`agents/retrieval_agent.py`)
Performs multi-hop RAG across at least 2 knowledge chunks before answering. Single-hop is explicitly insufficient.

**Decision boundaries:**
- Requires at minimum 2 chunk citations per answer
- Calls `web_search` tool in addition to local chunk retrieval
- Cites which chunk contributed to which reasoning hop
- Runs once per pending subtask that has resolved dependencies

### Critique Agent (`agents/critique_agent.py`)
Reviews output of every other agent. Scores PER CLAIM, not per output.

**Decision boundaries:**
- Never flags an entire output as wrong — only specific spans
- Assigns confidence 0.0–1.0 per claim
- Updates existing claims in SharedContext in-place
- Reviews all prior agents by default, or a targeted agent if specified

### Synthesis Agent (`agents/synthesis_agent.py`)
Merges all outputs, resolves contradictions flagged by critique, produces final answer with provenance map.

**Decision boundaries:**
- MUST resolve all flagged contradictions — does not surface them to the user
- Every sentence in the final answer is linked to its source agent and chunk
- Uses `synthesis_confidence` to communicate output reliability

### Compression Agent (`agents/compression_agent.py`)
Triggered when any agent reaches 85% of its token budget.

**Decision boundaries:**
- LOSSLESS for structured data: JSON, scores, claim IDs, chunk IDs, tool outputs
- LOSSY only for conversational filler and verbose prose
- Resets the target agent's token_used counter after compression

### Meta Agent (`agents/meta_agent.py`)
After each eval run, reads failure cases, identifies the worst-performing prompt, proposes a rewrite.

**Decision boundaries:**
- Proposes ONE rewrite at a time (the worst-performing dimension)
- Proposed rewrite is stored with status=`pending` — never auto-applied
- Human must approve via `/approve-rewrite`
- After approval, re-eval runs ONLY on previously failed cases
- Every approval, rejection, and delta is stored and queryable

---

## Tools (`tools/tools.py`)

All 4 tools have explicit failure contracts. The orchestrator handles each failure mode differently — logic is in code, not prompts.

| Tool | Timeout | Empty | Malformed |
|---|---|---|---|
| `web_search` | `success=False, error_type="timeout"` | `success=True, data=[]` | `success=False, error_type="malformed"` |
| `code_executor` | `exit_code=-1, stderr="timed out"` | `stdout="", exit_code=0` | `exit_code=-2, stderr="Invalid"` |
| `data_lookup` | `success=False, error_type="timeout"` | `success=True, data=[]` | `success=False` |
| `self_reflection` | N/A | `contradictions=[], error_type="empty"` | `success=False` |

Tool calls log: input, output, latency_ms, accepted, retry_of, timestamp, error.
Each tool allows up to 2 retries with modified input on failure.

---

## Evaluation Harness (`eval/eval_harness.py`)

15 test cases across 3 categories:

**Straightforward (5):** Known correct answers for baseline scoring. Tests basic RAG, context window, Docker, SQL/NoSQL, SSE.

**Ambiguous (5):** Underspecified inputs like "Make it better" and "Compare the two approaches." Scored on whether decomposition correctly identifies the ambiguity and produces typed sub-tasks.

**Adversarial (5):** Prompt injection attempts, factually wrong premises, forced contradictions. System must resist injections, correct false premises, and resolve critique-synthesis disagreements without surfacing them.

### Scoring Dimensions (all built from scratch)

| Dimension | What it measures |
|---|---|
| `answer_correctness` | Keyword coverage vs. expected/forbidden words |
| `citation_accuracy` | Claims with chunk citations + multi-hop bonus |
| `contradiction_resolution` | Flagged claims resolved in synthesis |
| `tool_efficiency` | Penalizes excess tool calls and failures |
| `budget_compliance` | Policy violations tracked and scored |
| `critique_agreement` | Synthesis confidence × unresolved flag ratio |

Every dimension returns a `float` score + `justification` string. No black-box frameworks used.

---

## API Endpoints

All documented at `http://localhost:8000/docs`

| Method | Path | Description |
|---|---|---|
| POST | `/query` | Submit query → SSE stream with real-time agent activity |
| GET | `/trace/{job_id}` | Full execution trace for a completed job |
| GET | `/eval/latest` | Latest eval summary by category and dimension |
| POST | `/approve-rewrite` | Human approval/rejection for pending prompt rewrite |
| POST | `/eval/rerun-failed` | Re-eval on previously failed cases with latest prompts |
| POST | `/eval/trigger` | Trigger a full eval run (background) |
| POST | `/meta/propose-rewrite` | Trigger meta-agent to propose a rewrite |
| GET | `/health` | Health check |

Error responses always include `error_code` (machine-readable), `message` (human-readable), and `job_id` if applicable.

---

## Context Window Management

Each agent declares a max token budget before execution. The `ContextBudgetManager` tracks consumption via tiktoken. If an agent tries to exceed its budget:

1. `check_budget()` returns `False` and logs a policy violation
2. The violation is stored in `SharedContext.policy_violations` (not silently dropped)
3. If budget reaches 85%, the orchestrator triggers the Compression Agent before proceeding

---

## Known Limitations

**Retrieval is keyword-based, not semantic.** The knowledge chunk retrieval uses word overlap scoring instead of vector embeddings. In production, replace `_retrieve_chunks()` with a FAISS/Pinecone lookup. Multi-hop reasoning quality depends entirely on the LLM's ability to connect retrieved chunks.

**Web search is a stub.** The `web_search` tool returns from a hardcoded mock database. It demonstrates the interface contract and failure modes but does not perform real web searches. Integration with SerpAPI or Brave Search would be straightforward.

**NL→SQL is a lookup table.** The `data_lookup` tool uses keyword matching to map natural language to SQL. For production, this should call the LLM with the database schema in context.

**Orchestrator routing adds latency.** Every routing step requires an LLM call to decide the next agent. For low-latency use cases, a hybrid approach (heuristic routing with LLM override) would be faster.

**Meta-agent prompt rewrites are not validated before storage.** Proposed rewrites are stored as-is from the LLM. A human reviewer should inspect the diff before approving.

**Eval scoring is heuristic.** Scoring dimensions like `answer_correctness` rely on keyword matching rather than semantic similarity. This produces false negatives for paraphrased correct answers.

**No authentication on endpoints.** All API endpoints are open. Add an API key middleware before exposing to the internet.

---

## Self-Improving Loop: What It Does and Doesn't Do

**What it does:**
- After every eval run, reads the bottom-scoring cases
- Identifies which scoring dimension had the worst average score
- Maps that dimension to the responsible agent
- Asks the LLM to propose a rewrite of that agent's system prompt
- Stores the proposal with full diff and justification
- Allows a human to approve/reject via API
- On approval: activates the new prompt and re-runs only the failed cases
- Stores every delta in the database with timestamps

**What it doesn't do:**
- It does not automatically apply rewrites (human-in-the-loop is enforced)
- It does not validate that a proposed rewrite is actually better before presenting it
- It does not perform multiple iterations per eval run (one proposal per run)
- It does not handle regressions gracefully — if a rewrite improves one dimension but hurts another, that requires a new approval cycle
- It does not version-control the full prompt history (only the most recent approved prompt per agent is active)

---

## What I Would Build Next

1. **Real vector search** — Replace keyword chunk retrieval with FAISS + sentence-transformers for semantic multi-hop RAG
2. **Real web search** — Integrate SerpAPI or Brave Search for live retrieval
3. **Streaming token output** — Currently SSE streams agent events; extend to token-level streaming within each agent response
4. **Prompt versioning** — Full Git-style version history for all agent prompts, not just current active
5. **LLM-as-judge scoring** — Replace keyword-based eval scoring with an LLM evaluator for semantic correctness
6. **Parallelism** — Run independent subtasks in parallel (currently sequential)
7. **Auth + rate limiting** — API key middleware and per-key rate limits
8. **Observability dashboard** — Grafana dashboard over the execution_log table with latency percentiles per agent

---

## AI Collaboration Attestation

This project was built with assistance from Claude (Anthropic). Claude was used to:
- Scaffold initial code structure and module templates
- Write docstrings and inline documentation
- Debug async generator patterns in the orchestrator
- Draft the README structure

All architectural decisions, agent design, failure contract definitions, and evaluation logic were designed by the author. Claude-generated code was reviewed, modified, and integrated by hand.