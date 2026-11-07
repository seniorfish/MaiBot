"""Event queue and agent state behavior for the event-driven core."""

from src.maisaka.agent_core import AgentState, Event, EventKind, EventQueue


def _event(key: str, session: str = "s1", *, is_self: bool = False) -> Event:
    return Event(
        kind=EventKind.MESSAGE,
        scope_id="scope",
        session_id=session,
        payload={"message_id": key},
        dedupe_key=key,
        is_self=is_self,
    )


def test_duplicate_and_self_events_are_not_enqueued() -> None:
    queue = EventQueue()
    assert queue.put(_event("m1")) is True
    assert queue.put(_event("m1")) is False
    assert queue.put(_event("m2", is_self=True)) is False
    assert queue.pending_count("scope") == 1


def test_drain_packs_same_session_and_keeps_others() -> None:
    queue = EventQueue()
    queue.put(_event("a1", "a"))
    queue.put(_event("a2", "a"))
    queue.put(_event("b1", "b"))

    packed = queue.drain_for_sleep("scope")

    assert [event.dedupe_key for event in packed] == ["a1", "a2"]
    assert queue.pending_count("scope") == 1
    assert queue.has_unread("scope", "b") is True
    assert queue.has_unread("scope", "a") is False


def test_unread_for_session_supports_send_interception() -> None:
    queue = EventQueue()
    queue.put(_event("x", "target"))
    queue.put(_event("y", "other"))

    unread = queue.unread_for_session("scope", "target")

    assert [event.dedupe_key for event in unread] == ["x"]


def test_remove_matching_keeps_unrelated_events() -> None:
    queue = EventQueue()
    queue.put(_event("a", "s1"))
    queue.put(_event("b", "s2"))

    removed = queue.remove_matching("scope", lambda event: event.message_id == "a")

    assert [event.message_id for event in removed] == ["a"]
    assert queue.pending_count("scope") == 1
    assert queue.has_unread("scope", "s2") is True


def test_internal_wake_signals_are_not_unread_messages() -> None:
    queue = EventQueue()
    queue.put(Event(kind=EventKind.INTERNAL, scope_id="scope", session_id="a", dedupe_key="wake-1"))

    assert queue.pending_count("scope") == 1
    assert queue.has_unread("scope", "a") is False
    assert queue.unread_for_session("scope", "a") == []


def test_snapshot_and_restore_roundtrip() -> None:
    queue = EventQueue()
    queue.put(_event("r1", "a"))
    queue.put(_event("r2", "b"))

    restored = EventQueue.restore("scope", queue.snapshot("scope"))

    assert restored.pending_count("scope") == 2
    assert restored.has_unread("scope", "a") is True
    # Dedupe keys survive the rebuild.
    assert restored.put(_event("r1", "a")) is False


def test_agent_state_tracks_events_and_wait() -> None:
    state = AgentState()
    assert state.next_seq() == 1
    state.mark_offered("e1")
    state.mark_handled("e1")
    state.start_wait(session_id="a", seconds=60, tool_call_id="c1", logical_turn_id="t1")

    cached = AgentState.from_dict(state.to_dict())

    assert cached.event_seq == 1
    assert cached.event_states["e1"] == "handled"
    assert cached.wait is not None
    assert cached.wait.session_id == "a"
    assert cached.is_waiting("a") is True
    assert cached.is_waiting("b") is False
    cached.clear_wait()
    assert cached.wait is None
