"""AgentLoop drives run_loop wake-ups when the event core is enabled."""

import asyncio
from types import SimpleNamespace

import pytest

from src.config.config import global_config
from src.maisaka.agent_core import AgentLoop, Event, EventKind
from src.maisaka.reasoning_engine import MaisakaReasoningEngine


def _engine(loop: AgentLoop) -> MaisakaReasoningEngine:
    runtime = SimpleNamespace(_get_agent_loop=lambda: loop, _internal_turn_queue=asyncio.Queue())
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime
    return engine


@pytest.mark.asyncio
async def test_event_batch_maps_to_message_trigger(monkeypatch) -> None:
    monkeypatch.setattr(global_config.agent_core, "enabled", True)
    loop = AgentLoop("scope")
    engine = _engine(loop)

    task = asyncio.create_task(engine._next_turn_trigger())
    await asyncio.sleep(0.02)
    loop.enqueue(Event(kind=EventKind.MENTION, scope_id="scope", session_id="a", dedupe_key="m1"))

    assert await task == "message"


@pytest.mark.asyncio
async def test_wait_timeout_maps_to_timeout_trigger(monkeypatch) -> None:
    monkeypatch.setattr(global_config.agent_core, "enabled", True)
    loop = AgentLoop("scope")
    engine = _engine(loop)

    task = asyncio.create_task(engine._next_turn_trigger())
    await asyncio.sleep(0.02)
    loop.enqueue(Event(kind=EventKind.WAIT_TIMEOUT, scope_id="scope", session_id="a", dedupe_key="t1"))

    assert await task == "timeout"


@pytest.mark.asyncio
async def test_disabled_mode_uses_legacy_queue(monkeypatch) -> None:
    monkeypatch.setattr(global_config.agent_core, "enabled", False)
    runtime = SimpleNamespace(_get_agent_loop=lambda: None, _internal_turn_queue=asyncio.Queue())
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime
    runtime._internal_turn_queue.put_nowait("proactive")

    assert await engine._next_turn_trigger() == "proactive"


def test_finish_events_marks_ingested_messages_handled() -> None:
    loop = AgentLoop("scope")
    loop.enqueue(Event(kind=EventKind.MESSAGE, scope_id="scope", session_id="a", dedupe_key="m1", payload={"message_id": "m1"}))
    loop.enqueue(Event(kind=EventKind.MESSAGE, scope_id="scope", session_id="a", dedupe_key="m2", payload={"message_id": "m2"}))

    finished = loop.finish_events(lambda event: event.message_id == "m1")

    assert finished == 1
    assert loop.pending_count() == 1


@pytest.mark.asyncio
async def test_scheduled_event_fires_after_delay() -> None:
    loop = AgentLoop("scope")
    event = Event(kind=EventKind.PROACTIVE, scope_id="scope", session_id="a", dedupe_key="later")

    loop.schedule_event(event, delay_seconds=0.01)

    assert loop.pending_count() == 0
    await asyncio.sleep(0.05)
    assert loop.pending_count() == 1


@pytest.mark.asyncio
async def test_cancel_scheduled_drops_pending_timers() -> None:
    loop = AgentLoop("scope")
    event = Event(kind=EventKind.PROACTIVE, scope_id="scope", session_id="a", dedupe_key="later")

    loop.schedule_event(event, delay_seconds=0.01)
    loop.cancel_scheduled()
    await asyncio.sleep(0.05)

    assert loop.pending_count() == 0
