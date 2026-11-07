"""read_image / read_emoji put stored media back into the planner context."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import ValidationError, validate

from src.common.data_models.message_component_data_model import (
    EmojiComponent,
    ImageComponent,
    MessageSequence,
    TextComponent,
)
from src.config.config import global_config
from src.core.tooling import ToolExecutionContext, ToolInvocation
from src.maisaka.agent_core.registry import get_agent_state, reset_scope
from src.maisaka.agent_core.tiering import LifecycleTracker
from src.maisaka.builtin_tool import read_emoji, read_image
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext
from src.maisaka.context.messages import ContentContextMessage, ToolResultMediaMessage

SCOPE = "read-media-tool-scope"


class _Runtime:
    """Minimal runtime: media tools only need the session scope and chat history."""

    def __init__(self, history=None) -> None:
        self.session_id = SCOPE
        self.log_prefix = "[test]"
        self._chat_history = list(history or [])

    def _agent_scope_id(self) -> str:
        return self.session_id


def _tool_ctx(runtime: _Runtime) -> BuiltinToolRuntimeContext:
    return BuiltinToolRuntimeContext(SimpleNamespace(), runtime)


def _history_message(message_id: str, components) -> ContentContextMessage:
    return ContentContextMessage(
        raw_message=MessageSequence(list(components)),
        visible_text="",
        timestamp=datetime.now(),
        message_id=message_id,
    )


def _invocation(tool_name: str, **arguments) -> ToolInvocation:
    return ToolInvocation(tool_name=tool_name, arguments=arguments, call_id="call-1")


def _execution_context() -> ToolExecutionContext:
    return ToolExecutionContext(session_id=SCOPE)


def _expire(message: ContentContextMessage, *, media_kind: str) -> None:
    """Register the message at sequence 0 and move the scope far past its expiry limit."""

    state = get_agent_state(SCOPE)
    LifecycleTracker(state.context_sidecar).register(message.context_item_id, media_kind, 0)
    state.event_seq = 10


@pytest.fixture(autouse=True)
def _media_tool_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(read_image, "resolve_enable_visual_planner", lambda: True)
    monkeypatch.setattr(read_emoji, "resolve_enable_visual_planner", lambda: True)
    monkeypatch.setattr(global_config.agent_core, "enabled", True)
    reset_scope(SCOPE)
    yield
    reset_scope(SCOPE)


def test_read_image_spec_requires_message_id() -> None:
    spec = read_image.get_tool_spec()

    assert spec.name == "read_image"
    assert "session_id" not in spec.parameters_schema["properties"]
    validate({"message_id": "msg-1"}, spec.parameters_schema)
    with pytest.raises(ValidationError):
        validate({}, spec.parameters_schema)


@pytest.mark.asyncio
async def test_read_image_reintroduces_expired_image() -> None:
    image = ImageComponent(binary_hash="", binary_data=b"image-bytes")
    message = _history_message("msg-1", [TextComponent("[msg_id:msg-1][user]看这个"), image])
    _expire(message, media_kind="image")
    runtime = _Runtime(history=[message])

    result = await read_image.handle_tool(
        _tool_ctx(runtime),
        _invocation("read_image", message_id="msg-1"),
        _execution_context(),
    )

    assert result.success
    assert "已重新引入 1 张图片" in result.content
    assert result.metadata["image_count"] == 1
    assert result.metadata["duplicate"] is False
    assert len(result.post_history_messages) == 1

    media_message = result.post_history_messages[0]
    assert isinstance(media_message, ToolResultMediaMessage)
    assert media_message.tool_call_id == "call-1"
    assert media_message.session_id == SCOPE
    reintroduced = [part for part in media_message.raw_message.components if isinstance(part, ImageComponent)]
    assert len(reintroduced) == 1
    assert reintroduced[0].binary_data == b"image-bytes"
    # The reintroduced image must expire again, on the longer reintroduced limit.
    entry = get_agent_state(SCOPE).context_sidecar[media_message.context_item_id]
    assert entry["kind"] == "image"
    assert entry["reintroduced"] is True


@pytest.mark.asyncio
async def test_read_image_skips_image_still_in_context() -> None:
    image = ImageComponent(binary_hash="", binary_data=b"image-bytes")
    message = _history_message("msg-1", [image])
    runtime = _Runtime(history=[message])

    result = await read_image.handle_tool(
        _tool_ctx(runtime),
        _invocation("read_image", message_id="msg-1"),
        _execution_context(),
    )

    assert result.success
    assert "无需重复引入" in result.content
    assert result.metadata["duplicate"] is True
    assert result.post_history_messages == []


@pytest.mark.asyncio
async def test_read_image_uses_database_when_history_is_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    stored_image = ImageComponent(binary_hash="", binary_data=b"stored-bytes")
    stored_message = SimpleNamespace(message_id="msg-db", raw_message=MessageSequence([stored_image]))
    calls = []

    def _find_messages(**kwargs):
        calls.append(kwargs)
        return [stored_message]

    monkeypatch.setattr(read_image, "find_messages", _find_messages)

    result = await read_image.handle_tool(
        _tool_ctx(_Runtime(history=[])),
        _invocation("read_image", message_id="msg-db"),
        _execution_context(),
    )

    assert result.success
    assert calls == [{"session_id": SCOPE, "message_id": "msg-db", "limit": 1}]
    assert result.metadata["message_id"] == "msg-db"
    assert result.metadata["image_hash"] == stored_image.binary_hash


@pytest.mark.asyncio
async def test_read_image_reports_missing_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(read_image, "find_messages", lambda **_kwargs: [])

    result = await read_image.handle_tool(
        _tool_ctx(_Runtime(history=[])),
        _invocation("read_image", message_id="msg-gone"),
        _execution_context(),
    )

    assert not result.success
    assert "找不到消息" in result.error_message


@pytest.mark.asyncio
async def test_read_image_reports_unreadable_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(self) -> None:
        raise ValueError("无法通过 image_hash 加载图片二进制数据: hash-1")

    monkeypatch.setattr(ImageComponent, "load_image_binary", _fail)
    stored_image = ImageComponent(binary_hash="hash-1")
    stored_message = SimpleNamespace(message_id="msg-db", raw_message=MessageSequence([stored_image]))
    monkeypatch.setattr(read_image, "find_messages", lambda **_kwargs: [stored_message])

    result = await read_image.handle_tool(
        _tool_ctx(_Runtime(history=[])),
        _invocation("read_image", message_id="msg-db"),
        _execution_context(),
    )

    assert not result.success
    assert "不可读取" in result.error_message


@pytest.mark.asyncio
async def test_read_image_reports_message_without_image() -> None:
    message = _history_message("msg-text", [TextComponent("只是文字")])

    result = await read_image.handle_tool(
        _tool_ctx(_Runtime(history=[message])),
        _invocation("read_image", message_id="msg-text"),
        _execution_context(),
    )

    assert not result.success
    assert "没有图片" in result.error_message


def test_read_emoji_spec_accepts_either_identifier() -> None:
    spec = read_emoji.get_tool_spec()

    assert spec.name == "read_emoji"
    assert "session_id" not in spec.parameters_schema["properties"]
    validate({"message_id": "msg-2"}, spec.parameters_schema)
    validate({"emoji_id": "hash-1"}, spec.parameters_schema)
    with pytest.raises(ValidationError):
        validate({}, spec.parameters_schema)


@pytest.mark.asyncio
async def test_read_emoji_reintroduces_expired_emoji() -> None:
    emoji = EmojiComponent(binary_hash="", binary_data=b"emoji-bytes", content="[表情包: 笑]")
    message = _history_message("msg-2", [emoji])
    _expire(message, media_kind="emoji")
    runtime = _Runtime(history=[message])

    result = await read_emoji.handle_tool(
        _tool_ctx(runtime),
        _invocation("read_emoji", message_id="msg-2"),
        _execution_context(),
    )

    assert result.success
    assert "已重新引入 1 个表情包" in result.content
    assert result.metadata["image_count"] == 1
    media_message = result.post_history_messages[0]
    assert isinstance(media_message, ToolResultMediaMessage)
    reintroduced = [part for part in media_message.raw_message.components if isinstance(part, EmojiComponent)]
    assert len(reintroduced) == 1
    assert reintroduced[0].binary_data == b"emoji-bytes"
    assert reintroduced[0].content == "[表情包: 笑]"
    assert get_agent_state(SCOPE).context_sidecar[media_message.context_item_id]["kind"] == "emoji"


@pytest.mark.asyncio
async def test_read_emoji_loads_hash_from_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    emoji_file = tmp_path / "emoji.png"
    emoji_file.write_bytes(b"library-emoji")
    stored_emoji = SimpleNamespace(file_hash="hash-1", full_path=str(emoji_file), description="笑")
    monkeypatch.setattr(read_emoji.emoji_manager, "get_emoji_by_hash", lambda _hash: stored_emoji)
    monkeypatch.setattr(read_emoji, "resolve_stored_image_path", lambda path: Path(path))

    result = await read_emoji.handle_tool(
        _tool_ctx(_Runtime(history=[])),
        _invocation("read_emoji", emoji_id="hash-1"),
        _execution_context(),
    )

    assert result.success
    assert "hash-1" in result.content
    media_message = result.post_history_messages[0]
    reintroduced = [part for part in media_message.raw_message.components if isinstance(part, EmojiComponent)]
    assert reintroduced[0].binary_data == b"library-emoji"
    assert reintroduced[0].content == "[表情包: 笑]"


@pytest.mark.asyncio
async def test_read_emoji_requires_one_identifier() -> None:
    result = await read_emoji.handle_tool(
        _tool_ctx(_Runtime(history=[])),
        _invocation("read_emoji"),
        _execution_context(),
    )

    assert not result.success
    assert "message_id" in result.error_message
