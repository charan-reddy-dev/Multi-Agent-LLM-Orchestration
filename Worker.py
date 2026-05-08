"""
Background Worker.
Processes agent jobs asynchronously from a queue.
Reads pending jobs from the database and processes them.
"""
import time
import json
import logging
import asyncio
import os

from agents.context_schema import SharedContext
from agents.orchestrator import run_streaming
from db.database import init_db, get_conn, save_job, log_event

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("worker")

POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))


async def process_job(job_id: str, query: str):
    logger.info(f"[Worker] Processing job {job_id}")
    ctx = SharedContext(job_id=job_id, original_query=query)

    try:
        async for event in run_streaming(ctx):
            log_event(
                job_id,
                event.get("agent", "orchestrator"),
                event.get("event", "unknown"),
                payload=event,
                latency_ms=event.get("latency_ms"),
                token_count=event.get("tokens_used"),
            )

        save_job(job_id, query, status="done",
                 final_answer=ctx.final_answer,
                 context_json=json.dumps(ctx.to_dict()),
                 completed_at=time.time())
        logger.info(f"[Worker] Job {job_id} completed successfully")

    except Exception as e:
        logger.exception(f"[Worker] Job {job_id} failed: {e}")
        save_job(job_id, query, status="failed", error=str(e))


async def worker_loop():
    init_db()
    logger.info("[Worker] Started. Polling for pending jobs...")

    while True:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT job_id, query FROM jobs WHERE status = 'pending' ORDER BY created_at ASC LIMIT 5"
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            job_id, query = row["job_id"], row["query"]
            # Mark as running
            save_job(job_id, query, status="running")
            await process_job(job_id, query)

        await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(worker_loop())