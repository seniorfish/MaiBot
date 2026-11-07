"""read_emoji builtin tool: reintroduce an original emoji into the context."""

from __future__ import annotations

from typing import Optional
import asyncio

from src.common.data_models.message_component_data_model import EmojiComponent
from src.common.logger import get_logger
from src.common.utils.image_path import resolve_stored_image_path
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.emoji_system.emoji_manager import emoji_manager
from src.maisaka.context.message_id_alias import to_display_message_id
from src.maisaka.visual.mode_utils import resolve_enable_visual_planner

from .context import BuiltinToolRuntimeContext
from .read_image import (
    MEDIA_KIND_EMOJI,
    build_reintroduced_media_message,
    ensure_media_binary,
    is_media_still_visible,
    iter_media_components,
    locate_message,
    register_reintroduced_media,
    resolve_session_id,
    resolve_tool_call_id,
)

logger = get_logger("maisaka_builtin_read_emoji")

_REINTRODUCED_EMOJI_LABEL = "[重新引入的表情包]"


def get_tool_spec() -> ToolSpec:
    """Build the read_emoji tool declaration."""

    return ToolSpec(
        name="read_emoji",
        description=(
            "把历史消息里的表情包原图重新放回上下文查看。表情包会随事件推进过期，"
            "上下文里只剩文字描述、或需要确认某个表情具体长什么样时调用它。"
            "优先填 message_id（消息上标注的 msg_id）；已知表情包哈希时才用 emoji_id。"
            "一次只引入一个表情包；它已经在上下文里且未过期时不会重复引入。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "包含目标表情包的消息 msg_id。",
                },
                "emoji_id": {
                    "type": "string",
                    "description": "表情包的哈希标识（通常来自其它工具输出）；不知道哈希时改用 message_id。",
                },
            },
            "anyOf": [
                {"required": ["message_id"]},
                {"required": ["emoji_id"]},
            ],
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


async def _load_emoji_by_hash(emoji_hash: str) -> EmojiComponent:
    """Load an emoji component from the registered library or the emoji table.

    Raises ValueError when the hash is unknown or the file is gone.
    """

    emoji = emoji_manager.get_emoji_by_hash(emoji_hash)
    if emoji is None:
        emoji = await asyncio.to_thread(emoji_manager.get_emoji_by_hash_from_db, emoji_hash)
    if emoji is None:
        raise ValueError(f"表情包库中没有哈希 {emoji_hash} 的记录")
    emoji_path = resolve_stored_image_path(emoji.full_path)
    emoji_bytes = await asyncio.to_thread(emoji_path.read_bytes)
    description = emoji.description.strip()
    return EmojiComponent(
        binary_hash=emoji.file_hash,
        content=f"[表情包: {description}]" if description else "[表情包]",
        binary_data=emoji_bytes,
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """Execute read_emoji."""

    runtime = tool_ctx.runtime
    target_message_id = str(invocation.arguments.get("message_id") or "").strip()
    target_emoji_id = str(invocation.arguments.get("emoji_id") or "").strip()
    if not target_message_id and not target_emoji_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "查看表情包工具需要提供 `message_id` 或 `emoji_id` 之一。",
        )
    session_id = resolve_session_id(tool_ctx, context)
    if not session_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "查看表情包工具无法确定目标会话，未执行。",
        )
    if not resolve_enable_visual_planner():
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "当前 Planner 未启用视觉能力，无法把表情包放回上下文。",
        )

    if target_message_id:
        located = await locate_message(
            history=runtime._chat_history,
            session_id=session_id,
            message_id=target_message_id,
        )
        if located is None:
            return tool_ctx.build_failure_result(
                invocation.tool_name,
                f"在会话历史中找不到消息 {to_display_message_id(target_message_id)}。"
                "请确认 msg_id 是否正确，且该消息属于当前会话。",
            )

        display_message_id = to_display_message_id(located.message_id)
        emojis = list(iter_media_components(located.sequence.components, MEDIA_KIND_EMOJI))
        if not emojis:
            return tool_ctx.build_failure_result(
                invocation.tool_name,
                f"消息 {display_message_id} 里没有表情包；普通图片请改用 read_image。",
            )

        emoji = emojis[0]
        emoji_hash = emoji.binary_hash
        if is_media_still_visible(
            runtime,
            runtime._chat_history,
            media_hash=emoji_hash,
            media_kind=MEDIA_KIND_EMOJI,
        ):
            return tool_ctx.build_success_result(
                invocation.tool_name,
                f"上下文里已经能看到这个表情包（消息 {display_message_id}），无需重复引入。",
                structured_content={
                    "message_id": located.message_id,
                    "emoji_hash": emoji_hash,
                    "reintroduced": False,
                },
                metadata={
                    "image_count": 0,
                    "duplicate": True,
                    "message_id": located.message_id,
                    "emoji_hash": emoji_hash,
                    "media_kind": MEDIA_KIND_EMOJI,
                },
            )

        try:
            await ensure_media_binary(emoji)
        except Exception as exc:
            logger.warning(
                f"{runtime.log_prefix} 读取消息表情包失败: message_id={located.message_id} "
                f"emoji_hash={emoji_hash} error={exc}"
            )
            return tool_ctx.build_failure_result(
                invocation.tool_name,
                f"消息 {display_message_id} 的表情包数据已不可读取（原始文件可能已被清理）：{exc}",
            )

        header_text = f"{_REINTRODUCED_EMOJI_LABEL} 来自消息 {display_message_id}"
        content_text = f"已重新引入 1 个表情包，来自消息 {display_message_id}。"
        source_message_id = located.message_id
    else:
        emoji_hash = target_emoji_id
        if is_media_still_visible(
            runtime,
            runtime._chat_history,
            media_hash=emoji_hash,
            media_kind=MEDIA_KIND_EMOJI,
        ):
            return tool_ctx.build_success_result(
                invocation.tool_name,
                f"上下文里已经能看到这个表情包（{emoji_hash}），无需重复引入。",
                structured_content={"emoji_hash": emoji_hash, "reintroduced": False},
                metadata={
                    "image_count": 0,
                    "duplicate": True,
                    "emoji_hash": emoji_hash,
                    "media_kind": MEDIA_KIND_EMOJI,
                },
            )

        try:
            emoji = await _load_emoji_by_hash(emoji_hash)
        except Exception as exc:
            logger.warning(f"{runtime.log_prefix} 按哈希读取表情包失败: emoji_hash={emoji_hash} error={exc}")
            return tool_ctx.build_failure_result(
                invocation.tool_name,
                f"无法读取表情包 {emoji_hash}：{exc}。也可以改用 read_emoji(message_id=...) 从消息里取。",
            )

        header_text = f"{_REINTRODUCED_EMOJI_LABEL} {emoji.content}"
        content_text = f"已重新引入 1 个表情包（{emoji_hash}）。"
        source_message_id = ""

    media_message = build_reintroduced_media_message(
        tool_call_id=resolve_tool_call_id(invocation),
        session_id=session_id,
        media_component=emoji,
        header_text=header_text,
    )
    register_reintroduced_media(runtime, media_message, MEDIA_KIND_EMOJI)
    logger.info(
        f"{runtime.log_prefix} read_emoji 重新引入表情包: emoji_hash={emoji_hash} "
        f"message_id={source_message_id or '-'}"
    )

    return tool_ctx.build_success_result(
        invocation.tool_name,
        content_text,
        structured_content={
            "message_id": source_message_id,
            "emoji_hash": emoji_hash,
            "reintroduced": True,
        },
        metadata={
            "image_count": 1,
            "duplicate": False,
            "message_id": source_message_id,
            "emoji_hash": emoji_hash,
            "media_kind": MEDIA_KIND_EMOJI,
        },
        post_history_messages=[media_message],
    )
