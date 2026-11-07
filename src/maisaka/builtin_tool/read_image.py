"""read_image builtin tool: reintroduce an original image from history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Optional, Sequence
import asyncio
import uuid

from src.A_memorix.core.image.component_paths import iter_message_image_components
from src.common.data_models.message_component_data_model import (
    EmojiComponent,
    ImageComponent,
    MessageSequence,
    TextComponent,
)
from src.common.logger import get_logger
from src.common.message_repository import find_messages
from src.config.config import global_config
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec
from src.maisaka.agent_core.registry import get_agent_state
from src.maisaka.agent_core.tiering import LifecycleTracker
from src.maisaka.context.message_id_alias import build_alias_map, to_display_message_id
from src.maisaka.context.messages import ContentContextMessage, LLMContextMessage, ToolResultMediaMessage
from src.maisaka.visual.mode_utils import resolve_enable_visual_planner

from .context import BuiltinToolRuntimeContext

logger = get_logger("maisaka_builtin_read_image")

MEDIA_KIND_IMAGE = "image"
MEDIA_KIND_EMOJI = "emoji"

MediaComponent = ImageComponent | EmojiComponent


@dataclass(slots=True)
class LocatedMessage:
    """A message found either in live history or in the persisted session."""

    message_id: str
    sequence: MessageSequence


def get_tool_spec() -> ToolSpec:
    """Build the read_image tool declaration."""

    return ToolSpec(
        name="read_image",
        description=(
            "把历史消息里的原图重新放回上下文查看。图片会随事件推进过期，"
            "上下文里出现「[图片已过期，可用 read_image 重新查看]」或只剩文字描述时调用它。"
            "message_id 填消息上标注的 msg_id（也可填工具媒体的 msg_id）。"
            "一次只引入一张图；同一张图已经在上下文里且未过期时不会重复引入。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {
                "message_id": {
                    "type": "string",
                    "description": "包含目标图片的消息 msg_id。",
                },
            },
            "required": ["message_id"],
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


def resolve_session_id(
    tool_ctx: BuiltinToolRuntimeContext,
    context: Optional[ToolExecutionContext],
) -> str:
    """Take the target session from the execution context (Focus injects and strips it)."""

    session_id = str(context.session_id or "").strip() if context is not None else ""
    if session_id:
        return session_id
    return str(tool_ctx.runtime.session_id or "").strip()


def resolve_tool_call_id(invocation: ToolInvocation) -> str:
    """Return the call id required by ToolResultMediaMessage; keep it unique when absent."""

    return str(invocation.call_id or "").strip() or f"read_media:{uuid.uuid4().hex}"


def resolve_message_id(raw_message_id: str, history: Sequence[LLMContextMessage]) -> str:
    """Expand the short msg_id alias shown to the model back to the stored id."""

    alias_map = build_alias_map(
        message.message_id
        for message in reversed(history)
        if isinstance(message, ContentContextMessage) and message.message_id
    )
    return alias_map.get(raw_message_id, raw_message_id)


def _find_history_message(
    history: Sequence[LLMContextMessage],
    message_id: str,
) -> Optional[ContentContextMessage]:
    """Find a live history message by its stored message id."""

    for history_message in reversed(history):
        if isinstance(history_message, ContentContextMessage) and history_message.message_id == message_id:
            return history_message
    return None


def _find_stored_message(session_id: str, message_id: str) -> Any:
    """Blocking DB lookup; callers must run it in a worker thread."""

    messages = find_messages(session_id=session_id, message_id=message_id, limit=1)
    return messages[0] if messages else None


async def locate_message(
    *,
    history: Sequence[LLMContextMessage],
    session_id: str,
    message_id: str,
) -> Optional[LocatedMessage]:
    """Locate a message by msg_id, preferring live history and falling back to the database.

    Live history covers tool-result media and messages that are still inside the context
    window; the database covers messages dropped by compaction or restored sessions.
    """

    target_message_id = resolve_message_id(message_id, history)
    history_message = _find_history_message(history, target_message_id)
    if history_message is not None:
        return LocatedMessage(message_id=target_message_id, sequence=history_message.media_sequence)

    stored_message = await asyncio.to_thread(_find_stored_message, session_id, target_message_id)
    if stored_message is None:
        return None
    return LocatedMessage(message_id=str(stored_message.message_id), sequence=stored_message.raw_message)


def iter_media_components(
    components: Sequence[Any],
    media_kind: str,
) -> Iterator[MediaComponent]:
    """Yield media components of one kind in message order, including nested forward content."""

    for _path, component in iter_message_image_components(components, include_emoji=True):
        if media_kind == MEDIA_KIND_EMOJI and isinstance(component, EmojiComponent):
            yield component
        elif media_kind == MEDIA_KIND_IMAGE and isinstance(component, ImageComponent):
            yield component


async def ensure_media_binary(component: MediaComponent) -> bytes:
    """Load the stored binary by hash when the component carries no inline data.

    Raises ValueError when the original file can no longer be resolved.
    """

    if component.binary_data:
        return component.binary_data
    if isinstance(component, EmojiComponent):
        await component.load_emoji_binary()
    else:
        await component.load_image_binary()
    return component.binary_data


def _lifecycle(runtime: Any) -> tuple[Optional[LifecycleTracker], int]:
    """Return the request-time expiry tracker and current event sequence, if enabled."""

    if not global_config.agent_core.enabled:
        return None, 0
    state = get_agent_state(runtime._agent_scope_id())
    return LifecycleTracker(state.context_sidecar), state.event_seq


def is_media_still_visible(
    runtime: Any,
    history: Sequence[LLMContextMessage],
    *,
    media_hash: str,
    media_kind: str,
) -> bool:
    """Check whether the same media is still rendered in the current context.

    A message without a lifecycle entry is never stripped by the request policy, so it
    counts as visible; a tracked message counts as visible until its expiry limit is hit.
    """

    if not media_hash:
        return False
    tracker, current_seq = _lifecycle(runtime)
    for history_message in reversed(history):
        if not isinstance(history_message, ContentContextMessage):
            continue
        if not any(
            component.binary_hash == media_hash
            for component in iter_media_components(history_message.media_sequence.components, media_kind)
        ):
            continue
        if tracker is None or not tracker.should_expire(history_message.context_item_id, current_seq):
            return True
    return False


def _clone_media_component(component: MediaComponent) -> MediaComponent:
    """Copy the component while sharing the immutable bytes object."""

    if isinstance(component, EmojiComponent):
        return EmojiComponent(
            binary_hash=component.binary_hash,
            content=component.content,
            binary_data=component.binary_data,
        )
    return ImageComponent(
        binary_hash=component.binary_hash,
        content=component.content,
        binary_data=component.binary_data,
    )


def build_reintroduced_media_message(
    *,
    tool_call_id: str,
    session_id: str,
    media_component: MediaComponent,
    header_text: str,
) -> ToolResultMediaMessage:
    """Build the media user message appended right after the tool result."""

    sequence = MessageSequence([TextComponent(header_text), _clone_media_component(media_component)])
    return ToolResultMediaMessage(
        tool_call_id=tool_call_id,
        session_id=session_id,
        raw_message=sequence,
        visible_text=f"{header_text}\n[图片]",
        timestamp=datetime.now(),
        message_id=f"tool_result:{tool_call_id}:reintroduced_media",
    )


def register_reintroduced_media(
    runtime: Any,
    message: ToolResultMediaMessage,
    media_kind: str,
) -> None:
    """Track the new message so it expires after IMAGE_REINTRODUCED_EVENTS events."""

    if not global_config.agent_core.enabled:
        return
    state = get_agent_state(runtime._agent_scope_id())
    LifecycleTracker(state.context_sidecar).register(
        message.context_item_id, media_kind, state.event_seq, reintroduced=True
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """Execute read_image."""

    runtime = tool_ctx.runtime
    target_message_id = str(invocation.arguments.get("message_id") or "").strip()
    if not target_message_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "查看图片工具需要提供有效的 `message_id` 参数。",
        )
    session_id = resolve_session_id(tool_ctx, context)
    if not session_id:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "查看图片工具无法确定目标会话，未执行。",
        )
    if not resolve_enable_visual_planner():
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            "当前 Planner 未启用视觉能力，无法把图片放回上下文。",
        )

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
    images = list(iter_media_components(located.sequence.components, MEDIA_KIND_IMAGE))
    if not images:
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"消息 {display_message_id} 里没有图片；表情包请改用 read_emoji。",
        )

    image = images[0]
    if is_media_still_visible(
        runtime,
        runtime._chat_history,
        media_hash=image.binary_hash,
        media_kind=MEDIA_KIND_IMAGE,
    ):
        return tool_ctx.build_success_result(
            invocation.tool_name,
            f"上下文里已经能看到这张图（消息 {display_message_id}），无需重复引入。",
            structured_content={
                "message_id": located.message_id,
                "image_hash": image.binary_hash,
                "reintroduced": False,
            },
            metadata={
                "image_count": 0,
                "duplicate": True,
                "message_id": located.message_id,
                "image_hash": image.binary_hash,
                "media_kind": MEDIA_KIND_IMAGE,
            },
        )

    try:
        await ensure_media_binary(image)
    except Exception as exc:
        logger.warning(
            f"{runtime.log_prefix} 读取消息图片失败: message_id={located.message_id} "
            f"image_hash={image.binary_hash} error={exc}"
        )
        return tool_ctx.build_failure_result(
            invocation.tool_name,
            f"消息 {display_message_id} 的图片数据已不可读取（原始文件可能已被清理）：{exc}",
        )

    header_text = f"[重新引入的图片] 来自消息 {display_message_id}"
    media_message = build_reintroduced_media_message(
        tool_call_id=resolve_tool_call_id(invocation),
        session_id=session_id,
        media_component=image,
        header_text=header_text,
    )
    register_reintroduced_media(runtime, media_message, MEDIA_KIND_IMAGE)
    logger.info(
        f"{runtime.log_prefix} read_image 重新引入图片: message_id={located.message_id} "
        f"image_hash={image.binary_hash}"
    )

    extra_note = f"（该消息共有 {len(images)} 张图，这里只引入第 1 张）" if len(images) > 1 else ""
    return tool_ctx.build_success_result(
        invocation.tool_name,
        f"已重新引入 1 张图片，位于消息 {display_message_id}{extra_note}。",
        structured_content={
            "message_id": located.message_id,
            "image_hash": image.binary_hash,
            "reintroduced": True,
        },
        metadata={
            "image_count": 1,
            "duplicate": False,
            "message_id": located.message_id,
            "image_hash": image.binary_hash,
            "media_kind": MEDIA_KIND_IMAGE,
        },
        post_history_messages=[media_message],
    )
