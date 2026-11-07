"""send_messages sends model-authored text and stops on the first failure."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.core.tooling import ToolExecutionResult, ToolInvocation
from src.maisaka.agent_core.registry import get_event_queue, reset_scope
from src.maisaka.agent_core.events import Event, EventKind
from src.maisaka.builtin_tool.send_messages import get_tool_spec, handle_tool


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


def _invocation(arguments):
    return ToolInvocation(
        tool_name="send_messages",
        arguments=arguments,
        call_id="call-send",
        session_id="sess",
        stream_id="sess",
    )


def test_spec_requires_messages() -> None:
    spec = get_tool_spec()
    assert spec.name == "send_messages"
    assert "messages" in spec.parameters_schema["required"]
    assert spec.parameters_schema["properties"]["messages"]["maxItems"] >= 1


@pytest.mark.asyncio
async def test_sends_all_messages_without_echoing_text() -> None:
    reset_scope("sess")
    runtime = _Runtime()
    context = _Context(runtime)
    sent = [SimpleNamespace(message_id="m1"), SimpleNamespace(message_id="m2")]

    with patch(
        "src.services.send_service._send_to_target_with_message",
        new=AsyncMock(side_effect=sent),
    ) as send_mock:
        result = await handle_tool(context, _invocation({"messages": ["第一条", "第二条"]}))

    assert result.success is True
    assert send_mock.await_count == 2
    assert "第一条" not in result.content
    assert "第二条" not in result.content
    assert result.structured_content["sent_message_ids"] == ["m1", "m2"]
    # Each list entry must stay one message: no automatic sentence splitting.
    assert context.post_process_calls[0][2]["enable_splitter"] is False
    runtime.record_planner_reply.assert_called_once()


@pytest.mark.asyncio
async def test_unread_same_session_blocks_sending() -> None:
    reset_scope("sess")
    queue = get_event_queue("sess")
    queue.put(
        Event(
            kind=EventKind.MESSAGE,
            scope_id="sess",
            session_id="sess",
            payload={"text": "完整句子来了"},
            dedupe_key="m-new",
        )
    )
    context = _Context(_Runtime())

    result = await handle_tool(context, _invocation({"messages": ["急着追问"]}))

    assert result.success is False
    assert "完整句子来了" in result.error_message


@pytest.mark.asyncio
async def test_stops_after_first_failure() -> None:
    reset_scope("sess")
    context = _Context(_Runtime())

    with patch(
        "src.services.send_service._send_to_target_with_message",
        new=AsyncMock(return_value=None),
    ) as send_mock:
        result = await handle_tool(context, _invocation({"messages": ["一", "二", "三"]}))

    assert result.success is False
    assert send_mock.await_count == 1
    assert len(result.structured_content["send_results"]) == 1


def test_attach_emoji_is_hidden_when_new_send_logic_is_disabled(monkeypatch) -> None:
    from src.config.config import global_config

    monkeypatch.setattr(global_config.emoji, "use_new_send_logic", False)
    spec = get_tool_spec()

    assert "attach_emoji" not in spec.parameters_schema["properties"]
    assert "send_emoji" in spec.description


def test_attach_emoji_is_kept_when_new_send_logic_is_enabled(monkeypatch) -> None:
    from src.config.config import global_config

    monkeypatch.setattr(global_config.emoji, "use_new_send_logic", True)
    spec = get_tool_spec()

    assert "attach_emoji" in spec.parameters_schema["properties"]


def test_chinese_typo_injection_is_off_by_default() -> None:
    """发出的正文必须原样落地，不能有后处理再往里塞错别字。

    ``[chinese_typo]`` 是按同音字随机替换真人的错字（什么 -> 神么），
    逐条落库后模型自己都看不出那是程序改的。
    """

    import inspect

    from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext

    signature = inspect.signature(BuiltinToolRuntimeContext.post_process_reply_message_items_async)
    assert signature.parameters["enable_chinese_typo"].default is False


@pytest.mark.asyncio
async def test_permanent_send_failure_reports_reason_and_forbids_retry() -> None:
    """平台拒收时要说明真实原因，并明确告诉模型别再重试。

    曾有一个非好友私聊里发来「？」：发送链把失败压成 None，模型只看到
    「发送失败」，于是换写法、换工具连试五次，白烧十几次模型调用。
    """

    reset_scope("sess")
    runtime = _Runtime()
    context = _Context(runtime)

    with patch(
        "src.services.send_service._send_to_target_with_message", new=AsyncMock(return_value=None)
    ), patch(
        "src.services.send_service.take_last_send_error",
        return_value="driver=gateway status=FAILED error=发送失败，请先添加对方为好友",
    ):
        result = await handle_tool(context, _invocation({"messages": ["咋了喵"]}))

    assert result.success is False
    assert "请先添加对方为好友" in result.error_message
    assert result.structured_content["retryable"] is False
    assert "结束本轮" in result.structured_content["hint"]
