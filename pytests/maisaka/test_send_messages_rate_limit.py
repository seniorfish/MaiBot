"""Anti-spam guards in send_messages: per-minute rate limit and duplicate suppression."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.config import global_config
from src.core.tooling import ToolExecutionResult, ToolInvocation
from src.maisaka.agent_core.events import Event, EventKind
from src.maisaka.agent_core.registry import get_event_queue, reset_scope
from src.maisaka.builtin_tool.send_messages import handle_tool, reset_rate_limits


class _Runtime:
    def __init__(self, session_id: str = "sess") -> None:
        self.session_id = session_id
        self.record_planner_reply = Mock()
        self.find_source_message_by_id = Mock(return_value=None)

    def _agent_scope_id(self) -> str:
        return self.session_id


class _Context:
    def __init__(self, runtime: _Runtime) -> None:
        self.runtime = runtime
        self.post_process_calls = []

    async def post_process_reply_message_items_async(self, text, attachments=None, **kwargs):
        self.post_process_calls.append((text, attachments, kwargs))
        return [SimpleNamespace(sequence=text)]

    @staticmethod
    def build_success_result(name, content="", structured_content=None, metadata=None):
        return ToolExecutionResult(
            tool_name=name, success=True, content=content, structured_content=structured_content
        )

    @staticmethod
    def build_failure_result(name, error_message, structured_content=None, metadata=None):
        return ToolExecutionResult(
            tool_name=name, success=False, error_message=error_message, structured_content=structured_content
        )


class _FakeClock:
    """Patchable replacement for send_messages._now()."""

    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _invocation(arguments):
    return ToolInvocation(
        tool_name="send_messages",
        arguments=arguments,
        call_id="call-send",
        session_id="sess",
        stream_id="sess",
    )


def _set_tools_config(**kwargs) -> None:
    tools = global_config.agent_core.tools
    for name, value in kwargs.items():
        setattr(tools, name, value)


@pytest.fixture(autouse=True)
def _restore_state():
    reset_rate_limits()
    tools = global_config.agent_core.tools
    saved = {
        "rate_limit_per_minute": tools.rate_limit_per_minute,
        "duplicate_suppress_seconds": tools.duplicate_suppress_seconds,
        "max_messages_per_turn": tools.max_messages_per_turn,
    }
    yield
    _set_tools_config(**saved)
    reset_rate_limits()


def _send_patch():
    """Patch the actual send call and count deliveries."""

    counter = {"n": 0}

    async def _send(**kwargs):
        counter["n"] += 1
        return SimpleNamespace(message_id=f"m{counter['n']}")

    return patch("src.services.send_service._send_to_target_with_message", new=AsyncMock(side_effect=_send)), counter


@pytest.mark.asyncio
async def test_sends_when_under_rate_limit() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=5, duplicate_suppress_seconds=60)
    context = _Context(_Runtime())
    patcher, counter = _send_patch()

    with patcher:
        result = await handle_tool(context, _invocation({"messages": ["一", "二"]}))

    assert result.success is True
    assert counter["n"] == 2
    assert len(context.post_process_calls) == 2


@pytest.mark.asyncio
async def test_rate_limit_rejects_whole_call_without_sending_anything() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=3, duplicate_suppress_seconds=0)
    context = _Context(_Runtime())

    with patch("src.maisaka.builtin_tool.send_messages._now", new=_FakeClock()):
        patcher, counter = _send_patch()
        with patcher:
            first = await handle_tool(context, _invocation({"messages": ["一", "二"]}))
            second = await handle_tool(context, _invocation({"messages": ["三"]}))
            third = await handle_tool(context, _invocation({"messages": ["四"]}))

    assert first.success is True
    assert second.success is True
    assert third.success is False
    assert "已发 3 条" in third.error_message
    assert "上限 3 条" in third.error_message
    assert "合并消息" in third.error_message
    # The rejected call must not have sent or even post-processed anything.
    assert counter["n"] == 3
    assert len(context.post_process_calls) == 3
    assert third.structured_content is None


@pytest.mark.asyncio
async def test_duplicate_text_is_suppressed() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=20, duplicate_suppress_seconds=60)
    context = _Context(_Runtime())

    with patch("src.maisaka.builtin_tool.send_messages._now", new=_FakeClock()):
        patcher, counter = _send_patch()
        with patcher:
            first = await handle_tool(context, _invocation({"messages": ["在吗"]}))
            repeated = await handle_tool(context, _invocation({"messages": ["换个说法", "在吗"]}))
            fresh = await handle_tool(context, _invocation({"messages": ["那没事了"]}))

    assert first.success is True
    assert repeated.success is False
    assert "刚刚已经发过" in repeated.error_message
    # The whole batch is rejected: the fresh first entry is not sent either.
    assert counter["n"] == 2
    assert fresh.success is True


@pytest.mark.asyncio
async def test_duplicate_inside_one_call_is_suppressed() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=20, duplicate_suppress_seconds=60)
    context = _Context(_Runtime())
    patcher, counter = _send_patch()

    with patcher:
        result = await handle_tool(context, _invocation({"messages": ["复读", "复读"]}))

    assert result.success is False
    assert counter["n"] == 0


@pytest.mark.asyncio
async def test_zero_disables_both_checks() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=0, duplicate_suppress_seconds=0)
    context = _Context(_Runtime())
    patcher, counter = _send_patch()

    with patcher:
        results = [await handle_tool(context, _invocation({"messages": ["重复"]})) for _ in range(5)]

    assert all(result.success for result in results)
    assert counter["n"] == 5


@pytest.mark.asyncio
async def test_sessions_do_not_share_budget() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=2, duplicate_suppress_seconds=0)
    context = _Context(_Runtime())

    with patch("src.maisaka.builtin_tool.send_messages._now", new=_FakeClock()):
        patcher, counter = _send_patch()
        with patcher:
            mine = await handle_tool(context, _invocation({"messages": ["一", "二"]}))
            other = await handle_tool(context, _invocation({"messages": ["一", "二"], "session_id": "sess-other"}))
            mine_again = await handle_tool(context, _invocation({"messages": ["三"]}))
            other_again = await handle_tool(context, _invocation({"messages": ["三"], "session_id": "sess-other"}))

    assert mine.success is True
    assert other.success is True
    assert mine_again.success is False
    assert other_again.success is False
    assert counter["n"] == 4


@pytest.mark.asyncio
async def test_rate_limit_window_expires() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=2, duplicate_suppress_seconds=0)
    context = _Context(_Runtime())
    clock = _FakeClock()

    with patch("src.maisaka.builtin_tool.send_messages._now", new=clock):
        patcher, counter = _send_patch()
        with patcher:
            first = await handle_tool(context, _invocation({"messages": ["一", "二"]}))
            blocked = await handle_tool(context, _invocation({"messages": ["三"]}))
            clock.advance(61)
            recovered = await handle_tool(context, _invocation({"messages": ["三"]}))

    assert first.success is True
    assert blocked.success is False
    assert recovered.success is True
    assert counter["n"] == 3


@pytest.mark.asyncio
async def test_duplicate_window_expires() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=0, duplicate_suppress_seconds=30)
    context = _Context(_Runtime())
    clock = _FakeClock()

    with patch("src.maisaka.builtin_tool.send_messages._now", new=clock):
        patcher, counter = _send_patch()
        with patcher:
            first = await handle_tool(context, _invocation({"messages": ["同一句"]}))
            blocked = await handle_tool(context, _invocation({"messages": ["同一句"]}))
            clock.advance(31)
            recovered = await handle_tool(context, _invocation({"messages": ["同一句"]}))

    assert first.success is True
    assert blocked.success is False
    assert recovered.success is True
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_failed_send_does_not_consume_budget() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=2, duplicate_suppress_seconds=60)
    context = _Context(_Runtime())

    with patch("src.maisaka.builtin_tool.send_messages._now", new=_FakeClock()):
        with patch(
            "src.services.send_service._send_to_target_with_message",
            new=AsyncMock(return_value=None),
        ) as failing_send:
            failed = await handle_tool(context, _invocation({"messages": ["一", "二"]}))

        patcher, counter = _send_patch()
        with patcher:
            retried = await handle_tool(context, _invocation({"messages": ["一", "二"]}))

    assert failed.success is False
    assert failing_send.await_count == 1
    # Nothing was delivered, so the retry still fits the per-minute budget.
    assert retried.success is True
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_unread_message_check_runs_before_anti_spam() -> None:
    reset_scope("sess")
    _set_tools_config(rate_limit_per_minute=1, duplicate_suppress_seconds=60)
    context = _Context(_Runtime())

    get_event_queue("sess").put(
        Event(
            kind=EventKind.MESSAGE,
            scope_id="sess",
            session_id="sess",
            payload={"text": "新消息"},
            dedupe_key="m-new",
        )
    )
    patcher, counter = _send_patch()

    with patcher:
        result = await handle_tool(context, _invocation({"messages": ["一"]}))

    assert result.success is False
    assert "新消息" in result.error_message
    assert counter["n"] == 0
