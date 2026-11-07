"""Runtime enqueues a turn only after the legacy gates pass."""

from types import SimpleNamespace

from src.maisaka import runtime as runtime_module
from src.maisaka.agent_core import AgentLoop
from src.maisaka.runtime import MaisakaHeartFlowChatting


def _runtime_with_loop() -> MaisakaHeartFlowChatting:
    runtime = object.__new__(MaisakaHeartFlowChatting)
    runtime.session_id = "session-1"
    runtime.message_cache = []
    runtime._message_turn_scheduled = False
    runtime._cancel_deferred_message_turn_task = lambda: None
    runtime._agent_loop = AgentLoop("session-1")
    runtime._get_agent_loop = lambda: runtime._agent_loop
    return runtime


def _message(text: str = "你好") -> SimpleNamespace:
    return SimpleNamespace(
        platform="qq",
        message_id="m-1",
        processed_plain_text=text,
        message_info=SimpleNamespace(user_info=SimpleNamespace(user_id="123")),
    )


def test_turn_enqueues_event_after_gates(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "is_bot_self", lambda platform, user_id: False)
    monkeypatch.setattr(runtime_module, "is_mentioned_bot_in_message", lambda message: (False, 0.0, 0.0))
    runtime = _runtime_with_loop()
    runtime.message_cache.append(_message())

    runtime._enqueue_message_turn()

    assert runtime._message_turn_scheduled is True
    assert runtime._agent_loop.pending_count() == 1


def test_self_message_is_not_enqueued(monkeypatch) -> None:
    monkeypatch.setattr(runtime_module, "is_bot_self", lambda platform, user_id: True)
    runtime = _runtime_with_loop()
    runtime.message_cache.append(_message())

    runtime._enqueue_message_turn()

    assert runtime._agent_loop.pending_count() == 0
