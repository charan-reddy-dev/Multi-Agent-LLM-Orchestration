"""
FastAPI Application — 5 required endpoints:
1. POST /query              — Submit query, receive streaming SSE response
2. GET  /trace/{job_id}     — Full execution trace for a completed job
3. GET  /eval/latest        — Latest eval run summary by category and dimension
4. POST /approve-rewrite    — Human approval/rejection for pending prompt rewrites
5. POST /eval/rerun-failed  — Trigger re-eval on previously failed cases

All responses include machine-readable error codes.
Structured logging throughout with consistent schema.
"""
from __future__ import annotations
import time
import uuid
import json
import logging
import asyncio
import hashlib
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.context_schema import SharedContext
from agents.orchestrator import run_streaming
from agents import meta_agent
from db.database import (
    init_db, get_conn, save_job, get_job, get_execution_trace, log_event
)

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent LLM Orchestration System",
    description="Production-grade multi-agent pipeline with self-improving eval loop.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("Database initialized")


# ── Request/Response Models ────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {"example": {"query": "What is retrieval-augmented generation?"}}


class ApproveRewriteRequest(BaseModel):
    rewrite_id: str
    approved: bool
    approved_by: Optional[str] = "human"


class RerunFailedRequest(BaseModel):
    eval_run_id: str
    threshold: Optional[float] = 0.6


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    job_id: Optional[str] = None


# ── Helper: SSE formatter ──────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Endpoint 1: Submit query → streaming SSE ─────────────────────────────────
@app.post(
    "/query",
    summary="Submit a query and receive streaming agent activity via SSE",
    responses={200: {"description": "SSE stream"}, 400: {"model": ErrorResponse}},
)
async def submit_query(req: QueryRequest):
    """
    Submits a query to the multi-agent pipeline.
    Returns a Server-Sent Events stream showing:
    - Which agent is currently writing
    - Tool calls in flight
    - Current context budget remaining
    - Final answer with provenance map
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail={
            "error_code": "EMPTY_QUERY",
            "message": "Query must not be empty.",
            "job_id": None,
        })

    job_id = str(uuid.uuid4())
    ctx = SharedContext(job_id=job_id, original_query=req.query)
    save_job(job_id, req.query, status="running")

    log_event(job_id, "orchestrator", "job_start",
              payload={"query": req.query, "input_hash": hashlib.md5(req.query.encode()).hexdigest()})

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for event in run_streaming(ctx):
                event["job_id"] = job_id

                # Log every event
                log_event(
                    job_id,
                    event.get("agent", "orchestrator"),
                    event.get("event", "unknown"),
                    payload=event,
                    latency_ms=event.get("latency_ms"),
                    token_count=event.get("tokens_used"),
                    policy_violation=(
                        json.dumps(event.get("policy_violations"))
                        if event.get("policy_violations") else None
                    ),
                )
                yield _sse(event)

            # Save completed job
            save_job(
                job_id, req.query, status="done",
                final_answer=ctx.final_answer,
                context_json=json.dumps(ctx.to_dict()),
                completed_at=time.time(),
            )
            log_event(job_id, "orchestrator", "job_complete",
                      payload={"final_answer_length": len(ctx.final_answer or "")})

        except Exception as e:
            logger.exception(f"[API] Pipeline error for job {job_id}: {e}")
            save_job(job_id, req.query, status="failed", error=str(e))
            yield _sse({
                "event": "pipeline_error",
                "job_id": job_id,
                "error_code": "PIPELINE_ERROR",
                "message": str(e),
            })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "X-Job-ID": job_id,
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ── Endpoint 2: Get full execution trace ──────────────────────────────────────
@app.get(
    "/trace/{job_id}",
    summary="Retrieve full execution trace for a completed job",
    responses={
        200: {"description": "Ordered sequence of all agent decisions, tool calls, and handoffs"},
        404: {"model": ErrorResponse},
    },
)
async def get_trace(job_id: str):
    """
    Returns the complete execution trace for a job, in chronological order.
    Includes: all agent decisions, tool calls, handoffs, routing log,
    policy violations, and token usage per agent.
    """
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail={
            "error_code": "JOB_NOT_FOUND",
            "message": f"No job found with id={job_id}",
            "job_id": job_id,
        })

    trace = get_execution_trace(job_id)

    # Also reconstruct context if available
    context_data = None
    if job.get("context_json"):
        try:
            context_data = json.loads(job["context_json"])
        except json.JSONDecodeError:
            pass

    return {
        "job_id": job_id,
        "query": job["query"],
        "status": job["status"],
        "created_at": job["created_at"],
        "completed_at": job.get("completed_at"),
        "final_answer": job.get("final_answer"),
        "execution_trace": trace,
        "routing_log": context_data.get("routing_log", []) if context_data else [],
        "policy_violations": context_data.get("policy_violations", []) if context_data else [],
        "token_usage": context_data.get("token_used", {}) if context_data else {},
        "provenance_map": context_data.get("provenance_map", []) if context_data else [],
        "trace_length": len(trace),
    }


# ── Endpoint 3: Latest eval run summary ───────────────────────────────────────
@app.get(
    "/eval/latest",
    summary="Retrieve latest eval run summary broken down by test category and scoring dimension",
)
async def get_latest_eval():
    """
    Returns the most recent eval run with:
    - Scores by category (straightforward / ambiguous / adversarial)
    - Scores by dimension (answer_correctness, citation_accuracy, etc.)
    - Individual case results with per-dimension justification strings
    """
    conn = get_conn()
    try:
        run_row = conn.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        if not run_row:
            raise HTTPException(status_code=404, detail={
                "error_code": "NO_EVAL_RUNS",
                "message": "No eval runs found. Run POST /eval/trigger first.",
                "job_id": None,
            })

        run = dict(run_row)
        results = conn.execute(
            "SELECT * FROM eval_results WHERE run_id = ? ORDER BY total_score ASC",
            (run["run_id"],)
        ).fetchall()

        case_details = []
        for r in results:
            row = dict(r)
            try:
                row["justifications"] = json.loads(row.get("justifications") or "{}")
            except Exception:
                pass
            case_details.append(row)

        summary = json.loads(run.get("summary") or "{}")
        return {
            "run_id": run["run_id"],
            "created_at": run["created_at"],
            "total_cases": run["total_cases"],
            "overall_avg_score": run["avg_score"],
            "by_category": summary.get("by_category", {}),
            "by_dimension": summary.get("by_dimension", {}),
            "case_results": case_details,
        }
    finally:
        conn.close()


# ── Endpoint 4: Approve or reject prompt rewrite ──────────────────────────────
@app.post(
    "/approve-rewrite",
    summary="Submit human approval or rejection for a pending prompt rewrite",
    responses={
        200: {"description": "Approval recorded"},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def approve_rewrite(req: ApproveRewriteRequest):
    """
    Approves or rejects a proposed prompt rewrite from the meta-agent.
    If approved:
    - The new prompt is activated in the pipeline
    - A targeted re-eval can then be triggered via /eval/rerun-failed

    All approvals and rejections are stored with timestamps and are queryable.
    """
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM prompt_rewrites WHERE rewrite_id = ?",
            (req.rewrite_id,)
        ).fetchone()
    finally:
        conn.close()

    if not row:
        raise HTTPException(status_code=404, detail={
            "error_code": "REWRITE_NOT_FOUND",
            "message": f"No rewrite found with id={req.rewrite_id}",
            "job_id": None,
        })

    rewrite = dict(row)
    if rewrite["status"] != "pending":
        raise HTTPException(status_code=409, detail={
            "error_code": "REWRITE_ALREADY_PROCESSED",
            "message": f"Rewrite {req.rewrite_id} already has status={rewrite['status']}",
            "job_id": None,
        })

    result = meta_agent.approve_rewrite(req.rewrite_id, req.approved, req.approved_by)

    return {
        "rewrite_id": req.rewrite_id,
        "status": result["status"],
        "agent_id": result["agent_id"],
        "approved_by": req.approved_by,
        "message": (
            f"Prompt for {result['agent_id']} updated and active."
            if req.approved else
            f"Rewrite rejected. Existing prompt unchanged."
        ),
    }


# ── Endpoint 5: Trigger re-eval on failed cases ───────────────────────────────
@app.post(
    "/eval/rerun-failed",
    summary="Trigger targeted re-eval on previously failed cases using latest approved prompts",
)
async def rerun_failed(req: RerunFailedRequest, bg: BackgroundTasks):
    """
    Re-runs only the test cases that failed (below threshold) in the specified eval run.
    Uses the latest approved prompts.
    Returns delta scores showing improvement or regression.

    Every delta is stored with timestamps for auditability.
    """
    conn = get_conn()
    try:
        run = conn.execute(
            "SELECT * FROM eval_runs WHERE run_id = ?", (req.eval_run_id,)
        ).fetchone()
    finally:
        conn.close()

    if not run:
        raise HTTPException(status_code=404, detail={
            "error_code": "EVAL_RUN_NOT_FOUND",
            "message": f"No eval run found with id={req.eval_run_id}",
            "job_id": None,
        })

    result = meta_agent.rerun_failed_cases(req.eval_run_id, threshold=req.threshold)

    return {
        "original_run_id": req.eval_run_id,
        "new_run_id": result.get("new_run_id"),
        "cases_retested": result.get("cases_retested", 0),
        "avg_delta": result.get("avg_delta", 0.0),
        "per_case_delta": result.get("per_case_delta", {}),
        "message": (
            f"Re-eval complete. {result.get('cases_retested', 0)} cases retested. "
            f"Avg delta: {result.get('avg_delta', 0):+.3f}"
        ),
    }


# ── Extra: Trigger eval run ────────────────────────────────────────────────────
@app.post("/eval/trigger", summary="Trigger a full eval run (all 15 test cases)")
async def trigger_eval(bg: BackgroundTasks):
    """Kicks off a full evaluation run in the background."""
    run_id = str(uuid.uuid4())

    def _run():
        from eval.eval_harness import run_eval
        run_eval(run_id=run_id)

    bg.add_task(_run)
    return {"run_id": run_id, "message": "Eval run started in background. Check /eval/latest when complete."}


# ── Extra: Propose prompt rewrite ─────────────────────────────────────────────
@app.post("/meta/propose-rewrite", summary="Trigger meta-agent to propose a prompt rewrite")
async def propose_rewrite(body: dict):
    eval_run_id = body.get("eval_run_id")
    if not eval_run_id:
        raise HTTPException(status_code=400, detail={
            "error_code": "MISSING_EVAL_RUN_ID",
            "message": "eval_run_id is required",
        })
    result = meta_agent.propose_rewrite(eval_run_id)
    if not result:
        return {"message": "No failed cases or no rewrite needed."}
    return result


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}