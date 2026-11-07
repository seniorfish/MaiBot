"""机器人自己发出的消息必须以 assistant 角色回放。

否则模型会在下一轮把自己说过的话当成用户说的——曾出现「我说 A，被你说成 B」
这类张冠李戴。
"""

from datetime import datetime

from src.chat.message_receive.message import SessionMessage
from src.common.data_models.mai_message_data_model import MessageInfo, UserInfo
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.llm_models.payload_content.context_item import (
    AssistantMessageItem,
    ContextImagePart,
    RoleType,
    UserMessageItem,
)
from src.maisaka.context.messages import SessionBackedMessage


def _session_message(text: str, *, image: bool = False) -> SessionMessage:
    message = SessionMessage("m-1", datetime.now(), "qq", received_at=datetime.now())
    message.session_id = "sess"
    message.message_info = MessageInfo(UserInfo("3651887238", "小鲸鱼"))
    sequence = MessageSequence([TextComponent(text)])
    if image:
        sequence = sequence.image(b"\x89PNG\r\n\x1a\n" + b"0" * 32, content="")
    message.raw_message = sequence
    message.processed_plain_text = text
    return message


def _backed(text: str, source_kind: str, *, image: bool = False) -> SessionBackedMessage:
    original = _session_message(text, image=image)
    return SessionBackedMessage.from_session_message(
        original,
        raw_message=original.raw_message,
        visible_text=text,
        source_kind=source_kind,
    )


def test_own_message_is_replayed_as_assistant() -> None:
    item = _backed("我说的话", "guided_reply").to_context_item()
    assert isinstance(item, AssistantMessageItem)
    assert item.role is RoleType.Assistant


def test_other_message_stays_user() -> None:
    item = _backed("别人说的话", "user").to_context_item()
    assert isinstance(item, UserMessageItem)
    assert item.role is RoleType.User


def test_own_message_does_not_carry_image_parts() -> None:
    # 图片只能出现在 user 消息里，自己发过的图退化成文字。
    item = _backed("发了张图", "guided_reply", image=True).to_context_item()
    assert isinstance(item, AssistantMessageItem)
    assert not any(isinstance(part, ContextImagePart) for part in item.parts)
