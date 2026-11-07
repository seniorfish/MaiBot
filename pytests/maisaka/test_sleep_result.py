"""Sleep duration text and the sleep result content the model sees."""

import time

from src.maisaka.agent_core import Event, EventKind
from src.maisaka.agent_core.timefmt import format_wait_duration
from src.maisaka.reasoning_engine import MaisakaReasoningEngine
from src.maisaka.runtime import MaisakaHeartFlowChatting


def test_short_waits_keep_seconds() -> None:
    assert format_wait_duration(12.34) == "12.3 秒"
    assert format_wait_duration(0) == "0.0 秒"


def test_long_waits_switch_to_human_units() -> None:
    assert format_wait_duration(1800) == "30 分"
    assert format_wait_duration(3600 * 2 + 60 * 13) == "2 小时 13 分"
    assert format_wait_duration(86400 * 3 + 3600 * 4) == "3 天 4 小时"


def _runtime_stub() -> MaisakaHeartFlowChatting:
    runtime = object.__new__(MaisakaHeartFlowChatting)
    runtime._agent_state = runtime._STATE_RUNNING
    runtime._pending_wait_tool_call_id = "sleep-call"
    runtime._pending_wait_logical_turn_id = "turn-sleep"
    runtime._pending_wait_started_at = time.time() - 4000
    runtime._pending_wait_seconds = None
    runtime._last_trigger_events = []
    runtime._wait_timeout_task = None
    runtime._cancel_wait_timeout_task = lambda: None
    return runtime


def _inbound(text: str) -> Event:
    return Event(
        kind=EventKind.MESSAGE,
        scope_id="scope",
        session_id="s",
        payload={"message_id": text, "text": text},
    )


def test_sleep_result_reports_new_messages() -> None:
    runtime = _runtime_stub()
    runtime._last_trigger_events = [_inbound("在吗"), _inbound("看这个")]
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime

    completed = engine._build_wait_completed_message(has_new_messages=True)

    assert completed.tool_name == "sleep"
    assert "实际空闲 1 小时 6 分" in completed.content
    assert "期间收到 2 条新消息" in completed.content
    assert "看这个" in completed.content


def test_sleep_result_reports_background_wake() -> None:
    runtime = _runtime_stub()
    runtime._last_trigger_events = [
        Event(kind=EventKind.SUBAGENT_RESULT, scope_id="scope", session_id="s")
    ]
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime

    completed = engine._build_wait_completed_message(has_new_messages=False)

    assert "没有新的用户消息" in completed.content


def test_sleep_result_reports_timer_wake() -> None:
    runtime = _runtime_stub()
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime

    completed = engine._build_wait_completed_message(has_new_messages=False)

    assert "定时器" in completed.content


def test_sleep_result_falls_back_to_new_message_flag() -> None:
    runtime = _runtime_stub()
    engine = object.__new__(MaisakaReasoningEngine)
    engine._runtime = runtime

    completed = engine._build_wait_completed_message(has_new_messages=True)

    assert "新的用户输入" in completed.content
