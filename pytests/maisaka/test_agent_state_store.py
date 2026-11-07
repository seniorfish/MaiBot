"""Agent state checkpoint and event queue restore."""

from src.maisaka.agent_core import AgentState, EventQueue
from src.maisaka.agent_core.store import AgentStateStore


def test_state_store_roundtrip(tmp_path) -> None:
    store = AgentStateStore("group:test", root=tmp_path)
    state = AgentState()
    state.next_seq()
    state.start_wait(session_id="a", seconds=30)
    store.save(state.to_dict(), [{"event_id": "e1", "kind": "message", "scope_id": "group:test", "session_id": "a", "dedupe_key": "d1"}])

    payload = store.load()

    assert payload["state"]["event_seq"] == 1
    assert payload["state"]["wait"]["session_id"] == "a"
    assert payload["events"][0]["dedupe_key"] == "d1"


def test_store_ignores_other_scope(tmp_path) -> None:
    store = AgentStateStore("group:a", root=tmp_path)
    store.save({"event_seq": 1}, [])
    other = AgentStateStore("group:b", root=tmp_path)

    assert other.load() == {}


def test_restore_into_merges_events() -> None:
    queue = EventQueue()
    added = queue.restore_into(
        [
            {"event_id": "e1", "kind": "message", "scope_id": "s", "session_id": "a", "dedupe_key": "d1"},
            {"event_id": "e2", "kind": "message", "scope_id": "s", "session_id": "a", "dedupe_key": "d1"},
        ]
    )

    assert added == 1
    assert queue.pending_count("s") == 1
