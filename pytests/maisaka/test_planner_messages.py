from datetime import datetime

from src.maisaka.context.planner_messages import build_planner_prefix


def test_build_planner_prefix_marks_self_message() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="麦麦",
        message_id="self-message",
        is_self_message=True,
    )

    assert 'is_self_message="true"' in prefix


def test_build_planner_prefix_does_not_mark_user_message() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="用户",
        message_id="user-message",
    )

    assert 'is_self_message="true"' not in prefix


def test_build_planner_prefix_carries_chat_type() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="用户",
        message_id="user-message",
        chat_type="群聊",
    )

    assert 'chat_type="群聊"' in prefix


def test_build_planner_prefix_omits_empty_chat_type() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="用户",
        message_id="user-message",
    )

    assert "chat_type=" not in prefix


def test_build_planner_prefix_carries_short_uid() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="用户",
        message_id="user-message",
        uid="ab12cd34",
    )

    assert 'uid="ab12cd34"' in prefix


def test_build_planner_prefix_carries_group_alias() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="zachary",
        message_id="user-message",
        alias="张峻峰",
    )

    assert 'alias="张峻峰"' in prefix


def test_build_planner_prefix_skips_alias_equal_to_user_name() -> None:
    prefix = build_planner_prefix(
        timestamp=datetime(2026, 8, 20, 1, 9, 30),
        user_name="张峻峰",
        message_id="user-message",
        alias="张峻峰",
    )

    assert "alias=" not in prefix
