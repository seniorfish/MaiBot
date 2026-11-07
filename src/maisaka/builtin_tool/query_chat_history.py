"""query_chat_history 内置工具。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import asyncio

from src.common import message_repository
from src.common.logger import get_logger
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.context.message_id_alias import to_display_message_id

from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_query_chat_history")

# 一次回查最多返回多少条：结果最终要进模型输入，超过这个量只会挤占推理空间。
MAX_LIMIT = 200
# 未指定条数时的默认窗口。
DEFAULT_LIMIT = 50
# 单条消息正文的展示上限，避免一条长文吞掉整个结果。
MAX_TEXT_LENGTH = 200
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_tool_spec() -> ToolSpec:
    """获取 query_chat_history 工具声明。"""

    return ToolSpec(
        name="query_chat_history",
        description=(
            "回查当前会话的真实聊天记录。上下文里较早的消息会被压缩或剥离，"
            "需要确认旧消息的原文、时间或发送者时使用。可用 start_time/end_time 指定时间段，"
            "或用 last_n 取最近 N 条；结果按时间正序返回，每条带 msg_id，便于 reply / view_forward_message 引用。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "start_time": {
                    "anyOf": [{"type": "string"}, {"type": "number"}],
                    "description": (
                        "起始时间，ISO 8601（如 2026-11-07T10:00:00 或 2026-11-07 10:00:00）或 epoch 秒；"
                        "省略表示不限起点。"
                    ),
                },
                "end_time": {
                    "anyOf": [{"type": "string"}, {"type": "number"}],
                    "description": "结束时间，格式同 start_time；省略表示到当前为止。",
                },
                "last_n": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        f"取最近 N 条；超过 {MAX_LIMIT} 条时按最近 {MAX_LIMIT} 条返回。"
                        "与 start_time/end_time 互斥；同时给出时以时间段为准。"
                    ),
                },
            },
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


def _from_epoch(value: float, argument_name: str) -> datetime:
    """把 epoch 秒换算成本机本地时间的 naive datetime。"""

    try:
        return datetime.fromtimestamp(value)
    except (OSError, OverflowError, ValueError) as exc:
        raise ValueError(f"`{argument_name}` 的 epoch 秒超出可表示范围：{value}。") from exc


def _parse_time_argument(raw_value: Any, argument_name: str) -> datetime:
    """解析时间参数，统一为本机本地时间的 naive datetime。

    数据库按本地时间保存 naive datetime，所以带时区的输入要先换算到本地，不带时区的按本机本地时间理解。
    """

    if isinstance(raw_value, bool):
        raise ValueError(f"`{argument_name}` 不接受布尔值，请填写 ISO 8601 时间或 epoch 秒。")

    if isinstance(raw_value, (int, float)):
        return _from_epoch(float(raw_value), argument_name)

    text = str(raw_value).strip()
    if not text:
        raise ValueError(f"`{argument_name}` 不能是空字符串；不需要限制时请省略该参数。")

    # 先按 ISO 8601 解析：像 20261107 这种紧凑日期本身也是合法的 ISO 格式，不能被当成 epoch。
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            return _from_epoch(float(text), argument_name)
        except ValueError as exc:
            raise ValueError(
                f"`{argument_name}` 无法解析为 ISO 8601 时间或 epoch 秒：{text!r}。"
                "示例：2026-11-07T10:00:00、2026-11-07 10:00:00、1762500000。"
            ) from exc

    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed


def _parse_optional_time(raw_value: Any, argument_name: str) -> Optional[datetime]:
    """解析可省略的时间参数，缺省或空值返回 None。"""

    if raw_value is None:
        return None
    if isinstance(raw_value, str) and not raw_value.strip():
        return None
    return _parse_time_argument(raw_value, argument_name)


def _parse_last_n(raw_value: Any) -> int:
    """解析 last_n，只接受正整数。"""

    if isinstance(raw_value, bool):
        raise ValueError("`last_n` 必须是正整数。")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, float) and raw_value.is_integer():
        value = int(raw_value)
    elif isinstance(raw_value, str) and raw_value.strip().isdigit():
        value = int(raw_value.strip())
    else:
        raise ValueError(f"`last_n` 必须是正整数，收到的值无效：{raw_value!r}。")
    if value <= 0:
        raise ValueError(f"`last_n` 必须是正整数，收到的值：{value}。")
    return value


def _resolve_session_id(
    tool_ctx: BuiltinToolRuntimeContext,
    context: Optional[ToolExecutionContext],
) -> str:
    """确定要回查的会话；Focus 路由器会在执行上下文中注入目标会话。"""

    if context is not None:
        context_session_id = str(context.session_id or "").strip()
        if context_session_id:
            return context_session_id
    return str(tool_ctx.runtime.session_id or "").strip()


def _format_time(value: datetime) -> str:
    return value.strftime(TIME_FORMAT)


async def _query_time_range(
    session_id: str,
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> tuple[List[Any], int, int]:
    """按时间段回查消息，返回（本页消息，区间总数，会话累计）。"""

    epoch_start = start_time.timestamp() if start_time is not None else None
    epoch_end = end_time.timestamp() if end_time is not None else None
    total_in_range = await asyncio.to_thread(
        message_repository.count_messages,
        session_id=session_id,
        start_time=epoch_start,
        end_time=epoch_end,
        time_field="timestamp",
    )
    session_total = await asyncio.to_thread(
        message_repository.count_messages,
        session_id=session_id,
        time_field="timestamp",
    )
    # 区间超过上限时取最近的一页，模型可以把 end_time 往前挪继续回查。
    messages = await asyncio.to_thread(
        message_repository.find_messages,
        session_id=session_id,
        start_time=epoch_start,
        end_time=epoch_end,
        limit=MAX_LIMIT,
        limit_mode="latest",
        time_field="timestamp",
    )
    return list(messages or []), int(total_in_range or 0), int(session_total or 0)


async def _query_last_n(session_id: str, last_n: Optional[int]) -> tuple[List[Any], int, int]:
    """按最近 N 条回查消息，返回（本页消息，请求条数，会话累计）。"""

    requested_last_n = last_n if last_n is not None else DEFAULT_LIMIT
    effective_last_n = min(requested_last_n, MAX_LIMIT)
    session_total = await asyncio.to_thread(
        message_repository.count_messages,
        session_id=session_id,
        time_field="timestamp",
    )
    messages = await asyncio.to_thread(
        message_repository.find_messages,
        session_id=session_id,
        limit=effective_last_n,
        limit_mode="latest",
        time_field="timestamp",
    )
    return list(messages or []), requested_last_n, int(session_total or 0)


def _build_message_view(message: Any) -> Dict[str, str]:
    """把历史消息转换成结果条目字段，正文按上限截断。"""

    user_info = message.message_info.user_info
    sender = str(user_info.user_cardname or user_info.user_nickname or user_info.user_id or "").strip()
    raw_text = str(message.processed_plain_text or "").strip()
    text = f"{raw_text[:MAX_TEXT_LENGTH]}…" if len(raw_text) > MAX_TEXT_LENGTH else raw_text
    return {
        "time": _format_time(message.timestamp),
        "sender": sender,
        "user_id": str(user_info.user_id or ""),
        "text": text or "（无文本内容）",
        "msg_id": to_display_message_id(message.message_id),
    }


def _build_content(
    *,
    range_text: str,
    page_stats: str,
    notes: List[str],
    message_views: List[Dict[str, str]],
) -> str:
    """拼接工具的可读结果。"""

    lines: List[str] = [f"查询范围：{range_text}；{page_stats}"]
    lines.extend(f"提示：{note}" for note in notes)
    if not message_views:
        lines.append("该范围内没有匹配的消息记录。")
        return "\n".join(lines)

    for index, view in enumerate(message_views, start=1):
        lines.append(f"{index}. [{view['time']}] {view['sender']} (msg_id={view['msg_id']})")
        lines.append(view["text"])
    return "\n".join(lines)


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """执行 query_chat_history 内置工具。"""

    arguments = invocation.arguments
    session_id = _resolve_session_id(tool_ctx, context)
    if not session_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "无法确定要回查的会话：工具执行上下文缺少 session_id。",
        )

    try:
        start_time = _parse_optional_time(arguments.get("start_time"), "start_time")
        end_time = _parse_optional_time(arguments.get("end_time"), "end_time")
        last_n = _parse_last_n(arguments["last_n"]) if arguments.get("last_n") is not None else None
    except ValueError as exc:
        return tool_ctx.build_failure_result(invocation.tool_name, str(exc))

    now = datetime.now()
    notes: List[str] = []
    if end_time is not None and end_time > now:
        notes.append(f"`end_time` 晚于当前时间，已截断为 {_format_time(now)}。")
        end_time = now
    if start_time is not None and start_time > now:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"`start_time`={_format_time(start_time)} 晚于当前时间，无法回查未来的记录。",
        )
    if start_time is not None and end_time is not None and start_time > end_time:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"`start_time`={_format_time(start_time)} 晚于 `end_time`={_format_time(end_time)}，请调换两端或检查参数。",
        )

    has_time_range = start_time is not None or end_time is not None
    if has_time_range and last_n is not None:
        notes.append(f"同时提供了 `last_n`={last_n}，已按时间段查询；只按条数回查时请去掉时间参数。")

    try:
        if has_time_range:
            messages, total_in_range, session_total = await _query_time_range(session_id, start_time, end_time)
            start_text = _format_time(start_time) if start_time is not None else "不限起点"
            end_text = _format_time(end_time) if end_time is not None else "当前"
            range_text = f"{start_text} ~ {end_text}"
            page_total: Optional[int] = total_in_range
        else:
            messages, requested_last_n, session_total = await _query_last_n(session_id, last_n)
            if requested_last_n > MAX_LIMIT:
                notes.append(f"`last_n`={requested_last_n} 超过上限 {MAX_LIMIT}，本次只返回最近 {MAX_LIMIT} 条。")
            range_text = f"最近 {requested_last_n} 条"
            page_total = None
    except Exception as exc:
        logger.exception(f"回查聊天记录失败: session_id={session_id} 异常={exc}")
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"回查聊天记录时发生异常：{exc}",
        )

    messages = sorted(messages, key=lambda item: item.timestamp)
    message_views = [_build_message_view(message) for message in messages]
    returned = len(message_views)

    if returned:
        page_total_for_label = page_total if page_total is not None else session_total
        # 兜住查询与计数之间插入新消息导致 returned > 计数 的情况。
        page_end = max(page_total_for_label, returned)
        page_start = max(1, page_end - returned + 1)
        page_label = f"本页第 {page_start}–{page_end} 条"
    else:
        page_label = "本页 0 条"
    if page_total is not None:
        page_stats = f"{page_label} / 区间共 {page_total} 条 / 会话累计 {session_total} 条"
    else:
        page_stats = f"{page_label} / 会话累计 {session_total} 条"

    if page_total is not None and returned < page_total:
        notes.append(
            f"区间内共 {page_total} 条，本次只返回最近的 {returned} 条；"
            f"需要更早的记录时，把 end_time 设为 {message_views[0]['time']} 之前再查。"
        )

    logger.info(
        f"触发聊天记录回查工具: session_id={session_id} "
        f"mode={'time_range' if has_time_range else 'last_n'} returned={returned} range={range_text}"
    )

    content = _build_content(range_text=range_text, page_stats=page_stats, notes=notes, message_views=message_views)

    effective_start = start_time if start_time is not None else (messages[0].timestamp if messages else None)
    effective_end = end_time if end_time is not None else (messages[-1].timestamp if messages else None)
    metadata: Dict[str, Any] = {
        "message_count": returned,
        "start_time": effective_start.timestamp() if effective_start is not None else None,
        "end_time": effective_end.timestamp() if effective_end is not None else None,
        "session_id": session_id,
        "query_mode": "time_range" if has_time_range else "last_n",
    }
    structured_content: Dict[str, Any] = {
        "session_id": session_id,
        "query_mode": "time_range" if has_time_range else "last_n",
        "start_time": metadata["start_time"],
        "end_time": metadata["end_time"],
        "message_count": returned,
        "total_in_range": page_total,
        "session_total": session_total,
        "truncated": page_total is not None and returned < page_total,
        "messages": [{"index": index, **view} for index, view in enumerate(message_views, start=1)],
    }

    return tool_ctx.build_success_result(
        invocation.tool_name,
        content,
        structured_content=structured_content,
        metadata=metadata,
    )
