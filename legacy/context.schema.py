"""
Shared context object schema for inter-agent communication.
Agents must NOT call each other directly — all handoffs go through this object.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time
import uuid


class AgentID(str, Enum):
    ORCHESTRATOR = "orchestrator"
    DECOMPOSITION = "decomposition"
    RETRIEVAL = "retrieval"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    COMPRESSION = "compression"
    META = "meta"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"  # waiting on dependency


@dataclass
class SubTask:
    task_id: str
    task_type: str          # e.g. "factual", "analytical", "comparative"
    query: str
    dependencies: List[str] = field(default_factory=list)   # list of task_ids
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[AgentID] = None
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None


@dataclass
class Claim:
    claim_id: str
    text: str
    source_agent: AgentID
    source_chunk: Optional[str] = None
    confidence: float = 1.0
    flagged: bool = False
    flag_reason: Optional[str] = None


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    input: Dict[str, Any]
    output: Any
    latency_ms: float
    accepted: bool
    retry_of: Optional[str] = None     # call_id of parent if this is a retry
    timestamp: float = field(default_factory=time.time)
    error: Optional[str] = None


@dataclass
class AgentMessage:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: AgentID = AgentID.ORCHESTRATOR
    content: str = ""
    claims: List[Claim] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tokens_used: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceEntry:
    sentence: str
    source_agent: AgentID
    source_chunk: Optional[str] = None
    claim_ids: List[str] = field(default_factory=list)


@dataclass
class SharedContext:
    """
    The single shared context object passed between all agents via the orchestrator.
    Agents read from and write to this object only through orchestrator-mediated handoffs.
    """
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    subtasks: List[SubTask] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    provenance_map: List[ProvenanceEntry] = field(default_factory=list)
    final_answer: Optional[str] = None
    routing_log: List[Dict[str, Any]] = field(default_factory=list)
    policy_violations: List[Dict[str, Any]] = field(default_factory=list)
    token_budget: Dict[AgentID, int] = field(default_factory=dict)
    token_used: Dict[AgentID, int] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def log_routing(self, from_agent: AgentID, to_agent: AgentID, reason: str, context_budget: int):
        self.routing_log.append({
            "timestamp": time.time(),
            "from": from_agent.value,
            "to": to_agent.value,
            "reason": reason,
            "context_budget_remaining": context_budget,
        })

    def log_policy_violation(self, agent: AgentID, detail: str):
        self.policy_violations.append({
            "timestamp": time.time(),
            "agent": agent.value,
            "detail": detail,
        })

    def add_message(self, msg: AgentMessage):
        self.messages.append(msg)
        self.token_used[msg.sender] = self.token_used.get(msg.sender, 0) + msg.tokens_used

    def get_agent_outputs(self, agent: AgentID) -> List[AgentMessage]:
        return [m for m in self.messages if m.sender == agent]

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        def _convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj
        return _convert(self)