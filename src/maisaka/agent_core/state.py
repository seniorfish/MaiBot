"""Mutable state of one agent loop, designed to be checkpointed later."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional, Set

EVENT_RECEIVED = "received"
EVENT_OFFERED = "offered"
EVENT_HANDLED = "handled"
EVENT_IGNORED = "ignored"


def _parse_injected_profiles(raw: Any) -> Dict[str, str]:
    """Person id -> profile fingerprint; older checkpoints stored a bare list."""

    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {str(item): "" for item in raw or []}


@dataclass(slots=True)
class WaitState:
    """A sleep/wait period that can be ended early by a new event."""

    session_id: str
    started_at: float
    seconds: float
    tool_call_id: str = ""
    logical_turn_id: str = ""
    reason: str = "sleep"

    @property
    def deadline(self) -> float:
        return self.started_at + self.seconds

    def remaining(self, now: Optional[float] = None) -> float:
        return max(0.0, self.deadline - (now if now is not None else time()))


@dataclass(slots=True)
class AgentState:
    """Per-scope state owned by ``AgentLoop``.

    ``event_seq`` is the tick used by context lifecycle policies (stage 3).
    ``event_states`` only keeps the recent tail to bound memory.
    """

    event_seq: int = 0
    pending_cursor: int = 0
    wait: Optional[WaitState] = None
    event_states: Dict[str, str] = field(default_factory=dict)
    discovered_tools: Set[str] = field(default_factory=set)
    # person_id -> fingerprint of the profile text the scope already saw.
    injected_profiles: Dict[str, str] = field(default_factory=dict)
    context_sidecar: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_event_states: int = 512

    def next_seq(self) -> int:
        self.event_seq += 1
        return self.event_seq

    def mark(self, event_id: str, state: str) -> None:
        self.event_states[event_id] = state
        if len(self.event_states) > self.max_event_states:
            for key in list(self.event_states)[: len(self.event_states) - self.max_event_states]:
                self.event_states.pop(key, None)

    def mark_offered(self, event_id: str) -> None:
        self.mark(event_id, EVENT_OFFERED)

    def mark_handled(self, event_id: str) -> None:
        self.mark(event_id, EVENT_HANDLED)

    def mark_ignored(self, event_id: str) -> None:
        self.mark(event_id, EVENT_IGNORED)

    def start_wait(
        self,
        *,
        session_id: str,
        seconds: float,
        tool_call_id: str = "",
        logical_turn_id: str = "",
    ) -> WaitState:
        self.wait = WaitState(
            session_id=session_id,
            started_at=time(),
            seconds=max(0.0, seconds),
            tool_call_id=tool_call_id,
            logical_turn_id=logical_turn_id,
        )
        return self.wait

    def clear_wait(self) -> Optional[WaitState]:
        previous = self.wait
        self.wait = None
        return previous

    def is_waiting(self, session_id: str = "") -> bool:
        if self.wait is None:
            return False
        return not session_id or self.wait.session_id == session_id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for checkpoints; keep the schema versioned."""

        return {
            "version": 1,
            "event_seq": self.event_seq,
            "pending_cursor": self.pending_cursor,
            "wait": None
            if self.wait is None
            else {
                "session_id": self.wait.session_id,
                "started_at": self.wait.started_at,
                "seconds": self.wait.seconds,
                "tool_call_id": self.wait.tool_call_id,
                "logical_turn_id": self.wait.logical_turn_id,
                "reason": self.wait.reason,
            },
            "event_states": dict(self.event_states),
            "discovered_tools": sorted(self.discovered_tools),
            "injected_profiles": dict(self.injected_profiles),
            "context_sidecar": {key: dict(value) for key, value in self.context_sidecar.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        wait_data = data.get("wait")
        wait = (
            WaitState(
                session_id=str(wait_data["session_id"]),
                started_at=float(wait_data["started_at"]),
                seconds=float(wait_data["seconds"]),
                tool_call_id=str(wait_data.get("tool_call_id") or ""),
                logical_turn_id=str(wait_data.get("logical_turn_id") or ""),
                reason=str(wait_data.get("reason") or "sleep"),
            )
            if isinstance(wait_data, dict) and wait_data.get("session_id")
            else None
        )
        sidecar = data.get("context_sidecar")
        return cls(
            event_seq=int(data.get("event_seq") or 0),
            pending_cursor=int(data.get("pending_cursor") or 0),
            wait=wait,
            event_states={str(k): str(v) for k, v in (data.get("event_states") or {}).items()},
            discovered_tools={str(item) for item in (data.get("discovered_tools") or [])},
            injected_profiles=_parse_injected_profiles(data.get("injected_profiles")),
            context_sidecar={
                str(key): dict(value)
                for key, value in (sidecar or {}).items()
                if isinstance(value, dict)
            },
        )
