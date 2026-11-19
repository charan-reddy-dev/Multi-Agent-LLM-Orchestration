"""
Log Query Interface.
A lightweight FastAPI app for querying execution logs.
Accessible at port 8001.
"""
import json
import sqlite3
import os
import time
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

DB_PATH = os.getenv("DB_PATH", "/app/data/agent_store.db")
app = FastAPI(title="Log Query Interface", version="1.0.0")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><head><title>Agent Log Explorer</title>
    <style>
      body{font-family:monospace;background:#111;color:#0f0;padding:20px}
      input,select{background:#222;color:#0f0;border:1px solid #0f0;padding:4px;margin:4px}
      button{background:#0f0;color:#111;border:none;padding:6px 12px;cursor:pointer}
      table{border-collapse:collapse;width:100%}
      th,td{border:1px solid #333;padding:6px;font-size:12px;text-align:left}
      th{background:#222}
      pre{background:#222;padding:10px;overflow:auto;max-height:300px}
    </style></head>
    <body>
      <h2>🔍 Agent Execution Log Explorer</h2>
      <div>
        <label>Job ID: <input id="jobId" placeholder="optional job_id"></label>
        <label>Event Type: 
          <select id="eventType">
            <option value="">All</option>
            <option>agent_start</option><option>agent_done</option>
            <option>routing_decision</option><option>tool_call</option>
            <option>policy_violation</option><option>job_start</option>
            <option>job_complete</option>
          </select>
        </label>
        <label>Limit: <input id="limit" value="50" style="width:60px"></label>
        <button onclick="fetchLogs()">Query Logs</button>
      </div>
      <div id="results" style="margin-top:20px"></div>
      <script>
        async function fetchLogs(){
          const jobId=document.getElementById('jobId').value;
          const event=document.getElementById('eventType').value;
          const limit=document.getElementById('limit').value;
          let url=`/logs?limit=${limit}`;
          if(jobId) url+=`&job_id=${jobId}`;
          if(event) url+=`&event_type=${event}`;
          const res=await fetch(url);
          const data=await res.json();
          let html=`<p>Found ${data.total} logs</p><table><tr>
            <th>Time</th><th>Job</th><th>Agent</th><th>Event</th>
            <th>Latency</th><th>Tokens</th><th>Violation</th></tr>`;
          for(const r of data.logs){
            const t=new Date(r.timestamp*1000).toISOString().slice(11,23);
            html+=`<tr>
              <td>${t}</td>
              <td><small>${(r.job_id||'').slice(0,8)}</small></td>
              <td>${r.agent_id}</td>
              <td>${r.event_type}</td>
              <td>${r.latency_ms?r.latency_ms.toFixed(0)+'ms':'-'}</td>
              <td>${r.token_count||'-'}</td>
              <td style="color:${r.policy_violation?'#f00':'#0f0'}">${r.policy_violation||'✓'}</td>
            </tr>`;
          }
          html+='</table>';
          document.getElementById('results').innerHTML=html;
        }
        fetchLogs();
      </script>
    </body></html>
    """


@app.get("/logs")
def query_logs(
    job_id: str = Query(None),
    event_type: str = Query(None),
    agent_id: str = Query(None),
    limit: int = Query(100, le=1000),
    offset: int = Query(0),
):
    conn = get_conn()
    try:
        clauses, params = [], []
        if job_id:
            clauses.append("job_id = ?")
            params.append(job_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        total = conn.execute(f"SELECT COUNT(*) FROM execution_log {where}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM execution_log {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()
        return {"total": total, "logs": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.get("/jobs")
def list_jobs(status: str = Query(None), limit: int = Query(20)):
    conn = get_conn()
    try:
        where = "WHERE status = ?" if status else ""
        params = [status] if status else []
        rows = conn.execute(
            f"SELECT job_id, query, status, created_at, completed_at FROM jobs {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit]
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.get("/violations")
def list_violations(limit: int = Query(50)):
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT * FROM execution_log
            WHERE policy_violation IS NOT NULL
            ORDER BY timestamp DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()