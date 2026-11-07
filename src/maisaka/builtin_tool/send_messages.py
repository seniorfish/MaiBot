"""send_messages builtin tool.

The event-driven core lets the model produce the final text directly, so this
tool replaces the replyer generation step while reusing the existing
post-processing, plugin extension, sending and history-writeback pipeline.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import asyncio
import time

from src.config.config import global_config
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.services import send_service

from .context import BuiltinToolRuntimeContext


# ---------------------------------------------------------------------------
# Anti-spam guards
#
# Everything below is process-local on purpose: a restart clears it and no
# record is ever written to disk.
# ---------------------------------------------------------------------------

_RATE_LIMIT_WINDOW_SECONDS = 60.0

# Serializes every read/write of the two registries below.
_rate_limit_lock = asyncio.Lock()
# (scope_id, target_session_id) -> monotonic timestamps of successfully sent messages in the window.
_send_timestamps: Dict[Tuple[str, str], List[float]] = {}
# (scope_id, target_session_id) -> (monotonic timestamp, text) pairs for duplicate suppression.
_recent_texts: Dict[Tuple[str, str], List[Tuple[float, str]]] = {}


def _now() -> float:
    """Return the monotonic clock; tests patch this to simulate window expiry."""

    return time.monotonic()


def reset_rate_limits() -> None:
    """Drop all anti-spam state; equivalent to what a process restart does."""

    _send_timestamps.clear()
    _recent_texts.clear()


def _prune_timestamps(key: Tuple[str, str], now: float) -> List[float]:
    """Drop rate-limit timestamps that fell out of the 60 second window."""

    kept = [ts for ts in _send_timestamps.get(key, []) if now - ts < _RATE_LIMIT_WINDOW_SECONDS]
    if kept:
        _send_timestamps[key] = kept
    else:
        _send_timestamps.pop(key, None)
    return kept


def _prune_texts(key: Tuple[str, str], now: float, window: float) -> List[Tuple[float, str]]:
    """Drop recorded texts older than the duplicate suppression window."""

    kept = [(ts, text) for ts, text in _recent_texts.get(key, []) if now - ts < window]
    if kept:
        _recent_texts[key] = kept
    else:
        _recent_texts.pop(key, None)
    return kept


async def _check_rate_limits(key: Tuple[str, str], messages: List[str]) -> Optional[str]:
    """Validate a whole call before anything is sent.

    A violation rejects the entire call so a partially sent batch never reaches
    the chat. Returns the model-facing error message, or None when the call may
    proceed.
    """

    tools_config = global_config.agent_core.tools
    rate_limit = int(tools_config.rate_limit_per_minute)
    duplicate_window = int(tools_config.duplicate_suppress_seconds)
    if rate_limit <= 0 and duplicate_window <= 0:
        return None

    async with _rate_limit_lock:
        now = _now()
        if rate_limit > 0:
            timestamps = _prune_timestamps(key, now)
            if len(timestamps) + len(messages) > rate_limit:
                # Tell the model how long the oldest message still blocks the window.
                oldest = timestamps[0] if timestamps else now
                wait_seconds = max(1, int(ceil(oldest + _RATE_LIMIT_WINDOW_SECONDS - now)))
                return (
                    f"本会话最近一分钟已发 {len(timestamps)} 条，上限 {rate_limit} 条，"
                    f"请等待约 {wait_seconds} 秒后重试，或改用更长的合并消息。"
                )
        if duplicate_window > 0:
            seen = {text for _, text in _prune_texts(key, now, float(duplicate_window))}
            for text in messages:
                if text in seen:
                    return "这条内容刚刚已经发过，不要重复发送。"
                # A batch that repeats itself is spam as well.
                seen.add(text)
    return None


async def _record_sent_messages(key: Tuple[str, str], texts: List[str]) -> None:
    """Record successfully sent messages so later calls can be rate limited."""

    if not texts:
        return

    tools_config = global_config.agent_core.tools
    rate_limit = int(tools_config.rate_limit_per_minute)
    duplicate_window = int(tools_config.duplicate_suppress_seconds)
    if rate_limit <= 0 and duplicate_window <= 0:
        return

    async with _rate_limit_lock:
        now = _now()
        if rate_limit > 0:
            timestamps = _prune_timestamps(key, now)
            timestamps.extend([now] * len(texts))
            _send_timestamps[key] = timestamps
        if duplicate_window > 0:
            recent = _prune_texts(key, now, float(duplicate_window))
            recent.extend((now, text) for text in texts)
            _recent_texts[key] = recent


def get_tool_spec() -> ToolSpec:
    """Return the tool declaration for send_messages."""

    max_messages = max(1, int(global_config.agent_core.tools.max_messages_per_turn))
    # Attaching an emoji is part of the new send logic (the model picks one from
    # show_emoji_list); with the legacy path the model must call send_emoji.
    emoji_attachment_allowed = bool(global_config.emoji.use_new_send_logic)
    properties = {
        "messages": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": max_messages,
            "description": "要依次发送的消息正文列表。",
        },
        "session_id": {
            "type": "string",
            "description": "目标聊天的 session_id；留空表示当前聊天。",
        },
        "reply_to": {
            "type": "string",
            "description": "要引用的消息 msg_id；省略则不引用。",
        },
        "at": {
            "type": "array",
            "items": {"type": "string"},
            "description": "要在首条消息开头 @ 的目标列表：消息 msg_id、用户 uid 或显示名都可以。",
        },
        "attach_pic": {
            "type": "array",
            "items": {"type": "string"},
            "description": "要附加的图片路径或标识列表。",
        },
    }
    attachments = "attach_pic 附加图片"
    if emoji_attachment_allowed:
        properties["attach_emoji"] = {
            "type": "string",
            "description": "要附加的表情包标题或编号；省略不附加。",
        }
        attachments += "、attach_emoji 附加表情包"
    else:
        attachments += "；表情包用 send_emoji 工具单独发送"
    return ToolSpec(
        name="send_messages",
        description=(
            "向指定聊天发送一条或多条独立消息；消息按顺序发送，任意一条失败即停止。"
            "正文由你直接给出，不再经过其他模型改写。"
            f"可以用 reply_to 引用一条已有消息，用 at 在开头 @ 某人，用 {attachments}。"
        ),
        parameters_schema={
            "type": "object",
            "properties": properties,
            "required": ["messages"],
            "additionalProperties": False,
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
        metadata={"visibility": "visible", "builtin_stage": "action"},
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """Send model-authored messages through the existing send service."""

    del context
    runtime = tool_ctx.runtime
    arguments = invocation.arguments

    raw_messages = arguments.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return tool_ctx.build_failure_result(invocation.tool_name, "messages 必须是非空字符串列表。")
    messages: List[str] = []
    for item in raw_messages:
        if not isinstance(item, str) or not item.strip():
            return tool_ctx.build_failure_result(invocation.tool_name, "messages 中每一项都必须是非空字符串。")
        messages.append(item.strip())
    max_messages = max(1, int(global_config.agent_core.tools.max_messages_per_turn))
    if len(messages) > max_messages:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"本条消息列表有 {len(messages)} 条，超过单轮上限 {max_messages} 条。",
        )

    target_session_id = str(arguments.get("session_id") or runtime.session_id).strip() or runtime.session_id
    reply_to = str(arguments.get("reply_to") or "").strip()
    target_message = runtime.find_source_message_by_id(reply_to) if reply_to else None
    if reply_to and target_message is None:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"未找到要引用的消息，reply_to={reply_to}",
        )

    # A newer message in the same conversation invalidates the planned reply.
    scope_id = runtime._agent_scope_id()
    from src.maisaka.agent_core.registry import get_event_queue

    queue = get_event_queue(scope_id)
    if queue.has_unread(scope_id, target_session_id):
        unread = queue.unread_for_session(scope_id, target_session_id)
        previews = [
            str(event.payload.get("text") or "").strip()
            for event in unread
            if str(event.payload.get("text") or "").strip()
        ]
        notice = "；".join(previews[-3:]) if previews else "（新消息没有可显示的文本）"
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"请阅读最新的消息，再决定如何回复：{notice}",
        )

    # Anti-spam checks run before any message is sent; a rejection sends nothing at all.
    rate_limit_error = await _check_rate_limits((scope_id, target_session_id), messages)
    if rate_limit_error:
        return tool_ctx.build_failure_result(invocation.tool_name, rate_limit_error)

    attachments: Dict[str, Any] = {}
    if arguments.get("at"):
        attachments["attach_at"] = arguments.get("at")
    if arguments.get("attach_pic"):
        attachments["attach_pic"] = arguments.get("attach_pic")
    if arguments.get("attach_emoji"):
        attachments["attach_emoji"] = arguments.get("attach_emoji")

    send_results: List[Dict[str, Any]] = []
    sent_ids: List[str] = []
    sent_texts: List[str] = []
    first_item = True
    for message_text in messages:
        try:
            items = await tool_ctx.post_process_reply_message_items_async(
                message_text,
                attachments if first_item else None,
                enable_splitter=False,
            )
        except Exception as exc:
            send_results.append({"index": len(send_results), "success": False, "error": str(exc)})
            break
        first_item = False
        message_failed = False
        for item in items:
            set_reply = bool(target_message is not None and not send_results)
            try:
                sent = await send_service._send_to_target_with_message(
                    message_sequence=item.sequence,
                    stream_id=target_session_id,
                    processed_plain_text="",
                    set_reply=set_reply,
                    reply_message=target_message if set_reply else None,
                    typing=bool(send_results),
                    sync_to_maisaka_history=True,
                    maisaka_source_kind="guided_reply",
                )
            except Exception as exc:
                send_results.append({"index": len(send_results), "success": False, "error": str(exc)})
                message_failed = True
                break
            if sent is None:
                send_results.append({"index": len(send_results), "success": False, "error": "发送失败"})
                message_failed = True
                break
            message_id = str(getattr(sent, "message_id", "") or "")
            if message_id:
                sent_ids.append(message_id)
            send_results.append({"index": len(send_results), "success": True, "message_id": message_id})
        if message_failed or not items:
            break
        sent_texts.append(message_text)

    # Only successfully delivered messages consume rate-limit and duplicate budget.
    await _record_sent_messages((scope_id, target_session_id), sent_texts)

    if not any(result.get("success") for result in send_results):
        reason = next(
            (
                str(result.get("error") or "").strip()
                for result in send_results
                if not result.get("success") and str(result.get("error") or "").strip()
            ),
            "",
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"消息未能发送：{reason}" if reason else "消息未能发送：没有一条成功，请检查内容和附件参数后重试。",
            structured_content={"send_results": send_results},
        )

    runtime.record_planner_reply()
    receipt = "、".join(
        "成功" if result.get("success") else f"失败（{result.get('error') or '未知原因'}）" for result in send_results
    )
    return tool_ctx.build_success_result(
        invocation.tool_name,
        f"发送结果：({receipt})",
        structured_content={
            "sent_message_ids": sent_ids,
            "send_results": send_results,
            "session_id": target_session_id,
        },
    )
