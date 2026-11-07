"""query_chat_history 内置工具的回查行为。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from src.common import message_repository
from src.core.tooling import ToolExecutionContext, ToolInvocation
from src.maisaka.builtin_tool import query_chat_history
from src.maisaka.builtin_tool.context import BuiltinToolRuntimeContext

TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _message(
    timestamp: datetime,
    text: str,
    *,
    message_id: str,
    user_id: str = "10001",
    user_nickname: str = "群友",
    user_cardname: Optional[str] = None,
) -> SimpleNamespace:
    """构造工具只读的历史消息字段。"""

    return SimpleNamespace(
        timestamp=timestamp,
        message_id=message_id,
        processed_plain_text=text,
        message_info=SimpleNamespace(
            user_info=SimpleNamespace(
                user_id=user_id,
                user_nickname=user_nickname,
                user_cardname=user_cardname,
            )
        ),
    )


class _FakeRepository:
    """记录调用参数的仓储替身。"""

    def __init__(
        self,
        messages: List[SimpleNamespace],
        *,
        in_range_total: Optional[int] = None,
        session_total: Optional[int] = None,
    ) -> None:
        self.messages = messages
        self.in_range_total = in_range_total
        self.session_total = session_total
        self.find_calls: List[Dict[str, Any]] = []
        self.count_calls: List[Dict[str, Any]] = []

    def find_messages(self, **kwargs: Any) -> List[SimpleNamespace]:
        self.find_calls.append(kwargs)
        limit = kwargs.get("limit") or 0
        return self.messages[-limit:] if limit else list(self.messages)

    def count_messages(self, **kwargs: Any) -> int:
        self.count_calls.append(kwargs)
        if kwargs.get("start_time") is not None or kwargs.get("end_time") is not None:
            return self.in_range_total if self.in_range_total is not None else len(self.messages)
        return self.session_total if self.session_total is not None else len(self.messages)


def _install_repository(monkeypatch: pytest.MonkeyPatch, repository: _FakeRepository) -> None:
    monkeypatch.setattr(message_repository, "find_messages", repository.find_messages)
    monkeypatch.setattr(message_repository, "count_messages", repository.count_messages)


def _tool_ctx(session_id: str = "chat-history") -> BuiltinToolRuntimeContext:
    return BuiltinToolRuntimeContext(SimpleNamespace(), SimpleNamespace(session_id=session_id))


def _invocation(**arguments: Any) -> ToolInvocation:
    return ToolInvocation(tool_name="query_chat_history", arguments=arguments, call_id="call-history")


def _execution_context(session_id: str = "chat-history") -> ToolExecutionContext:
    return ToolExecutionContext(session_id=session_id)


@pytest.mark.asyncio
async def test_time_range_query_returns_messages_in_ascending_order(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    range_start = now - timedelta(hours=2)
    range_end = now - timedelta(hours=1)
    messages = [
        _message(range_end - timedelta(seconds=30), "第二条", message_id="m2", user_cardname="群主"),
        _message(range_start + timedelta(seconds=30), "第一条", message_id="m1", user_id="10002"),
    ]
    repository = _FakeRepository(messages, in_range_total=2)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(start_time=range_start.isoformat(), end_time=range_end.isoformat()),
        _execution_context(),
    )

    assert result.success
    assert result.error_message == ""
    assert f"查询范围：{range_start.strftime(TIME_FORMAT)} ~ {range_end.strftime(TIME_FORMAT)}" in result.content
    assert "本页第 1–2 条 / 区间共 2 条 / 会话累计 2 条" in result.content
    assert result.content.index("第一条") < result.content.index("第二条")
    assert "msg_id=m1" in result.content
    assert "群主" in result.content
    assert result.structured_content["messages"][0]["user_id"] == "10002"
    assert result.structured_content["messages"][0]["sender"] == "群友"

    find_call = repository.find_calls[0]
    assert find_call["session_id"] == "chat-history"
    assert find_call["limit"] == query_chat_history.MAX_LIMIT
    assert find_call["limit_mode"] == "latest"
    assert find_call["time_field"] == "timestamp"
    assert find_call["start_time"] == pytest.approx(range_start.timestamp())
    assert find_call["end_time"] == pytest.approx(range_end.timestamp())
    assert find_call["start_time"] != find_call["end_time"]

    assert result.metadata["message_count"] == 2
    assert result.metadata["start_time"] == pytest.approx(range_start.timestamp())
    assert result.metadata["end_time"] == pytest.approx(range_end.timestamp())
    assert "pause_execution" not in result.metadata
    assert result.structured_content["query_mode"] == "time_range"
    assert result.structured_content["truncated"] is False


@pytest.mark.asyncio
async def test_last_n_query_keeps_the_newest_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    messages = [
        _message(now - timedelta(minutes=offset), f"消息{offset}", message_id=f"id{offset}")
        for offset in range(5, 0, -1)
    ]
    repository = _FakeRepository(messages)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(last_n=3),
        _execution_context(),
    )

    assert result.success
    assert repository.find_calls[0]["limit"] == 3
    assert repository.find_calls[0]["limit_mode"] == "latest"
    assert repository.find_calls[0]["time_field"] == "timestamp"
    assert "start_time" not in repository.find_calls[0]
    assert "查询范围：最近 3 条" in result.content
    assert "本页第 3–5 条 / 会话累计 5 条" in result.content
    assert "消息1" in result.content
    assert "消息2" in result.content
    assert "消息3" in result.content
    assert "消息4" not in result.content
    assert result.structured_content["query_mode"] == "last_n"
    assert result.structured_content["message_count"] == 3
    assert result.metadata["start_time"] == pytest.approx((now - timedelta(minutes=3)).timestamp())
    assert result.metadata["end_time"] == pytest.approx((now - timedelta(minutes=1)).timestamp())


@pytest.mark.asyncio
async def test_default_window_is_fifty_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _FakeRepository([])
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(_tool_ctx(), _invocation(), _execution_context())

    assert result.success
    assert "查询范围：最近 50 条" in result.content
    assert repository.find_calls[0]["limit"] == query_chat_history.DEFAULT_LIMIT


@pytest.mark.asyncio
async def test_empty_result_succeeds_with_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _FakeRepository([], in_range_total=0)
    _install_repository(monkeypatch, repository)
    now = datetime.now()

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(start_time=(now - timedelta(days=1)).isoformat()),
        _execution_context(),
    )

    assert result.success
    assert "本页 0 条 / 区间共 0 条 / 会话累计 0 条" in result.content
    assert "没有匹配的消息记录" in result.content
    assert result.metadata["message_count"] == 0
    assert result.structured_content["messages"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected_hint"),
    [
        ({"start_time": "not-a-time"}, "无法解析为 ISO 8601 时间或 epoch 秒"),
        ({"end_time": "2026-13-45T99:00:00"}, "无法解析为 ISO 8601 时间或 epoch 秒"),
        ({"last_n": 0}, "`last_n` 必须是正整数"),
        ({"last_n": -3}, "`last_n` 必须是正整数"),
        ({"last_n": "abc"}, "`last_n` 必须是正整数"),
        ({"last_n": True}, "`last_n` 必须是正整数"),
        ({"start_time": "2099-01-01T00:00:00"}, "无法回查未来的记录"),
    ],
)
async def test_invalid_arguments_fail_without_querying(
    monkeypatch: pytest.MonkeyPatch,
    arguments: Dict[str, Any],
    expected_hint: str,
) -> None:
    repository = _FakeRepository([])
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(_tool_ctx(), _invocation(**arguments), _execution_context())

    assert not result.success
    assert expected_hint in result.error_message
    assert repository.find_calls == []
    assert repository.count_calls == []


@pytest.mark.asyncio
async def test_reversed_time_range_fails_without_querying(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _FakeRepository([])
    _install_repository(monkeypatch, repository)
    now = datetime.now()

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(
            start_time=(now - timedelta(hours=1)).isoformat(),
            end_time=(now - timedelta(hours=2)).isoformat(),
        ),
        _execution_context(),
    )

    assert not result.success
    assert "请调换两端" in result.error_message
    assert repository.find_calls == []


@pytest.mark.asyncio
async def test_last_n_is_truncated_to_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    messages = [_message(now - timedelta(seconds=offset), f"消息{offset}", message_id=f"id{offset}") for offset in range(3)]
    repository = _FakeRepository(messages)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(_tool_ctx(), _invocation(last_n=500), _execution_context())

    assert result.success
    assert repository.find_calls[0]["limit"] == query_chat_history.MAX_LIMIT
    assert f"`last_n`=500 超过上限 {query_chat_history.MAX_LIMIT}" in result.content


@pytest.mark.asyncio
async def test_over_long_time_range_keeps_the_newest_page_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    messages = [
        _message(now - timedelta(seconds=offset), f"消息{offset}", message_id=f"id{offset}")
        for offset in range(query_chat_history.MAX_LIMIT, 0, -1)
    ]
    repository = _FakeRepository(messages, in_range_total=450, session_total=4580)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(start_time=(now - timedelta(days=30)).isoformat()),
        _execution_context(),
    )

    assert result.success
    assert "本页第 251–450 条 / 区间共 450 条 / 会话累计 4580 条" in result.content
    assert "本次只返回最近的 200 条" in result.content
    assert result.structured_content["truncated"] is True
    assert result.metadata["message_count"] == query_chat_history.MAX_LIMIT


@pytest.mark.asyncio
async def test_time_range_wins_when_last_n_is_also_given(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    repository = _FakeRepository([_message(now - timedelta(minutes=1), "唯一一条", message_id="m1")], in_range_total=1)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(start_time=(now - timedelta(hours=1)).isoformat(), last_n=5),
        _execution_context(),
    )

    assert result.success
    assert "同时提供了 `last_n`=5，已按时间段查询" in result.content
    assert repository.find_calls[0]["limit"] == query_chat_history.MAX_LIMIT
    assert repository.find_calls[0]["start_time"] is not None


@pytest.mark.asyncio
async def test_future_end_time_is_clamped_to_now(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    repository = _FakeRepository([_message(now - timedelta(minutes=1), "刚刚", message_id="m1")], in_range_total=1)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx(),
        _invocation(start_time=(now - timedelta(hours=1)).isoformat(), end_time=(now + timedelta(days=2)).isoformat()),
        _execution_context(),
    )

    assert result.success
    assert "`end_time` 晚于当前时间，已截断为" in result.content
    assert repository.find_calls[0]["end_time"] <= datetime.now().timestamp()


@pytest.mark.asyncio
async def test_execution_context_session_wins_over_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    repository = _FakeRepository([_message(now - timedelta(minutes=1), "焦点群消息", message_id="m1")])
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx("runtime-session"),
        _invocation(last_n=1),
        _execution_context("focus-session"),
    )

    assert result.success
    assert repository.find_calls[0]["session_id"] == "focus-session"
    assert repository.count_calls[0]["session_id"] == "focus-session"


@pytest.mark.asyncio
async def test_missing_session_id_fails_with_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    repository = _FakeRepository([])
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(
        _tool_ctx(""),
        _invocation(last_n=1),
        ToolExecutionContext(),
    )

    assert not result.success
    assert "session_id" in result.error_message
    assert repository.find_calls == []


@pytest.mark.asyncio
async def test_long_text_is_truncated_and_sender_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime.now()
    long_text = "长" * (query_chat_history.MAX_TEXT_LENGTH + 50)
    messages = [
        _message(now - timedelta(minutes=2), long_text, message_id="m1", user_id="10003", user_nickname="昵称"),
        _message(now - timedelta(minutes=1), "短文本", message_id="m2", user_id="10004", user_nickname="", user_cardname=""),
    ]
    repository = _FakeRepository(messages)
    _install_repository(monkeypatch, repository)

    result = await query_chat_history.handle_tool(_tool_ctx(), _invocation(last_n=2), _execution_context())

    assert result.success
    assert "长" * query_chat_history.MAX_TEXT_LENGTH + "…" in result.content
    assert "长" * (query_chat_history.MAX_TEXT_LENGTH + 1) not in result.content
    # 没有群名片时回退昵称，昵称也为空时回退 user_id。
    assert "昵称" in result.content
    assert "10004" in result.content
    assert result.structured_content["messages"][0]["text"].endswith("…")
    assert result.structured_content["messages"][1]["msg_id"] == "m2"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("2026-11-07T10:00:00", datetime(2026, 11, 7, 10, 0, 0)),
        ("2026-11-07 10:00:00", datetime(2026, 11, 7, 10, 0, 0)),
        ("2026-11-07", datetime(2026, 11, 7, 0, 0, 0)),
        ("20261107", datetime(2026, 11, 7, 0, 0, 0)),
        (1762500000, datetime.fromtimestamp(1762500000)),
        (1762500000.5, datetime.fromtimestamp(1762500000.5)),
        ("1762500000", datetime.fromtimestamp(1762500000)),
    ],
)
def test_time_parsing_accepts_iso_and_epoch(raw_value: Any, expected: datetime) -> None:
    assert query_chat_history._parse_time_argument(raw_value, "start_time") == expected


@pytest.mark.parametrize(
    "raw_value",
    ["2026-11-07T10:00:00+08:00", "2026-11-07T10:00:00Z", "2026-11-07T10:00:00-05:00"],
)
def test_time_parsing_converts_aware_input_to_local_naive(raw_value: str) -> None:
    expected = datetime.fromisoformat(raw_value).astimezone().replace(tzinfo=None)

    parsed = query_chat_history._parse_time_argument(raw_value, "end_time")

    assert parsed.tzinfo is None
    assert parsed == expected


@pytest.mark.parametrize("raw_value", ["", "   ", "昨天", "2026-11-07T25:00:00", None])
def test_time_parsing_rejects_invalid_values(raw_value: Any) -> None:
    with pytest.raises(ValueError):
        query_chat_history._parse_time_argument(raw_value, "start_time")


def test_get_tool_spec_has_no_session_id_and_no_required_fields() -> None:
    spec = query_chat_history.get_tool_spec()

    assert spec.name == "query_chat_history"
    properties = spec.parameters_schema["properties"]
    assert set(properties) == {"start_time", "end_time", "last_n"}
    assert "session_id" not in properties
    assert spec.parameters_schema.get("required", []) == []
    # 超出上限的值要能在工具里被截断并解释，所以 schema 不做硬性上限。
    assert "maximum" not in properties["last_n"]
    assert str(query_chat_history.MAX_LIMIT) in properties["last_n"]["description"]
