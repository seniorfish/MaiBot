"""Runtime persists and restores the scope checkpoint."""

import pytest

from src.config.config import global_config
from src.maisaka import runtime as runtime_module
from src.maisaka.agent_core import Event, EventKind
from src.maisaka.agent_core.registry import get_agent_state, get_event_queue, reset_scope
from src.maisaka.agent_core.store import AgentStateStore
from src.maisaka.runtime import MaisakaHeartFlowChatting


class _ScopedStore:
    def __init__(self, scope_id, root=None):
        self._store = AgentStateStore(scope_id, root=root)

    def load(self):
        return self._store.load()

    def save(self, state, events):
        return self._store.save(state, events)


def _runtime(scope_id="persist-scope"):
    runtime = object.__new__(MaisakaHeartFlowChatting)
    runtime.session_id = scope_id
    runtime.log_prefix = "[persist-test]"
    runtime._agent_scope_id = lambda: scope_id
    return runtime


@pytest.mark.asyncio
async def test_persist_and_restore_scope(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(global_config.agent_core, "enabled", True)
    monkeypatch.setattr(runtime_module, "AgentStateStore", lambda scope_id: _ScopedStore(scope_id, tmp_path))
    scope_id = "persist-scope"
    reset_scope(scope_id)
    runtime = _runtime(scope_id)

    state = get_agent_state(scope_id)
    state.next_seq()
    state.start_wait(session_id=scope_id, seconds=30)
    queue = get_event_queue(scope_id)
    queue.put(
        Event(
            kind=EventKind.MESSAGE,
            scope_id=scope_id,
            session_id=scope_id,
            payload={"text": "hello"},
            dedupe_key="persist-1",
        )
    )

    await runtime._persist_agent_core_state()
    reset_scope(scope_id)
    await runtime._restore_agent_core_state()

    restored_state = get_agent_state(scope_id)
    restored_queue = get_event_queue(scope_id)
    assert restored_state.wait is not None
    assert restored_state.event_seq == 1
    assert restored_queue.pending_count(scope_id) == 1
