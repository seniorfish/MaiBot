"""Process-wide event queues/states shared by every runtime in one scope.

Focus groups have one scope with several member runtimes, so the queue cannot
live on a single runtime. The registry is intentionally a plain dict: all
access happens on the asyncio loop.
"""

from __future__ import annotations

from typing import Dict

from .events import EventQueue
from .loop import AgentLoop
from .state import AgentState

_queues: Dict[str, EventQueue] = {}
_states: Dict[str, AgentState] = {}
_loops: Dict[str, AgentLoop] = {}


def get_event_queue(scope_id: str) -> EventQueue:
    queue = _queues.get(scope_id)
    if queue is None:
        queue = EventQueue()
        _queues[scope_id] = queue
    return queue


def get_agent_state(scope_id: str) -> AgentState:
    state = _states.get(scope_id)
    if state is None:
        state = AgentState()
        _states[scope_id] = state
    return state


def get_agent_loop(scope_id: str) -> AgentLoop:
    """One consumer per scope: every runtime in a Focus group shares it."""

    loop = _loops.get(scope_id)
    if loop is None:
        loop = AgentLoop(scope_id, queue=get_event_queue(scope_id), state=get_agent_state(scope_id))
        _loops[scope_id] = loop
    return loop


def reset_scope(scope_id: str) -> None:
    """Drop queues/state; used by /clear and tests."""

    _queues.pop(scope_id, None)
    _states.pop(scope_id, None)
    loop = _loops.pop(scope_id, None)
    if loop is not None:
        loop.cancel_scheduled()


def scope_ids() -> list[str]:
    return sorted(set(_queues) | set(_states) | set(_loops))
