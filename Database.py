"""
Database schema and initialization.
All eval runs, execution traces, and prompt rewrites are stored here.
Re-running the eval on the same inputs produces a diff-able output.
"""
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "/app/data/agent_store.db")


def get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create all tables. Idempotent."""
    conn = get_conn()
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id      TEXT PRIMARY KEY,
            query       TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            created_at  REAL NOT NULL,
            completed_at REAL,
            final_answer TEXT,
            context_json TEXT,
            error       TEXT
        );

        CREATE TABLE IF NOT EXISTS execution_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      TEXT NOT NULL,
            timestamp   REAL NOT NULL,
            agent_id    TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            input_hash  TEXT,
            output_hash TEXT,
            latency_ms  REAL,
            token_count INTEGER,
            payload     TEXT,
            policy_violation TEXT,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            call_id     TEXT PRIMARY KEY,
            job_id      TEXT NOT NULL,
            tool_name   TEXT NOT NULL,
            input_json  TEXT,
            output_json TEXT,
            latency_ms  REAL,
            accepted    INTEGER,
            retry_of    TEXT,
            error       TEXT,
            timestamp   REAL NOT NULL,
            FOREIGN KEY (job_id) REFERENCES jobs(job_id)
        );

        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id      TEXT PRIMARY KEY,
            created_at  REAL NOT NULL,
            total_cases INTEGER,
            avg_score   REAL,
            scores_json TEXT,
            summary     TEXT
        );

        CREATE TABLE IF NOT EXISTS eval_results (
            result_id       TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            case_id         TEXT NOT NULL,
            case_category   TEXT NOT NULL,
            query           TEXT NOT NULL,
            answer_correctness  REAL,
            citation_accuracy   REAL,
            contradiction_resolution REAL,
            tool_efficiency     REAL,
            budget_compliance   REAL,
            critique_agreement  REAL,
            total_score         REAL,
            justifications  TEXT,
            job_id          TEXT,
            created_at      REAL NOT NULL,
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS prompt_rewrites (
            rewrite_id      TEXT PRIMARY KEY,
            eval_run_id     TEXT NOT NULL,
            agent_id        TEXT NOT NULL,
            dimension       TEXT NOT NULL,
            original_prompt TEXT NOT NULL,
            proposed_prompt TEXT NOT NULL,
            diff_summary    TEXT NOT NULL,
            justification   TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            approved_by     TEXT,
            approved_at     REAL,
            delta_score     REAL,
            created_at      REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS prompt_rewrite_evals (
            eval_id         TEXT PRIMARY KEY,
            rewrite_id      TEXT NOT NULL,
            run_id          TEXT NOT NULL,
            delta_scores    TEXT,
            cases_retested  TEXT,
            created_at      REAL NOT NULL,
            FOREIGN KEY (rewrite_id) REFERENCES prompt_rewrites(rewrite_id)
        );

        CREATE INDEX IF NOT EXISTS idx_exec_log_job ON execution_log(job_id);
        CREATE INDEX IF NOT EXISTS idx_exec_log_event ON execution_log(event_type);
        CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id);
        CREATE INDEX IF NOT EXISTS idx_prompt_rewrites_status ON prompt_rewrites(status);
        """)
        conn.commit()
        logger.info(f"[DB] Initialized at {DB_PATH}")
    finally:
        conn.close()


def log_event(job_id: str, agent_id: str, event_type: str,
              payload: dict = None, latency_ms: float = None,
              token_count: int = None, policy_violation: str = None):
    """Write a structured log entry."""
    import hashlib, json, time
    conn = get_conn()
    try:
        payload_str = json.dumps(payload) if payload else None
        input_hash = hashlib.md5((payload_str or "").encode()).hexdigest() if payload_str else None
        conn.execute("""
            INSERT INTO execution_log
            (job_id, timestamp, agent_id, event_type, input_hash, latency_ms, token_count, payload, policy_violation)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, time.time(), agent_id, event_type, input_hash,
              latency_ms, token_count, payload_str, policy_violation))
        conn.commit()
    finally:
        conn.close()


def save_job(job_id: str, query: str, status: str = "pending",
             final_answer: str = None, context_json: str = None,
             error: str = None, completed_at: float = None):
    import time, json
    conn = get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO jobs
            (job_id, query, status, created_at, completed_at, final_answer, context_json, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (job_id, query, status, time.time(), completed_at, final_answer, context_json, error))
        conn.commit()
    finally:
        conn.close()


def get_job(job_id: str) -> dict:
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_execution_trace(job_id: str) -> list:
    """Returns full execution trace for a job in chronological order."""
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT * FROM execution_log
            WHERE job_id = ?
            ORDER BY timestamp ASC
        """, (job_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()