"""Request-level context lifecycle: hot info, media expiry and dedupe folding."""

from src.llm_models.payload_content.context_item import (
    ContextImagePart,
    ContextItemMeta,
    ContextTextPart,
    FunctionCallOutputItem,
    UserMessageItem,
)
from src.maisaka.agent_core.tiering import LifecycleTracker, apply_request_policy, fold_repeated_text

LONG_TEXT = "搜索结果。" * 80


def _meta() -> ContextItemMeta:
    return ContextItemMeta.create(logical_turn_id="t1")


def test_long_hot_info_is_replaced_after_expiry() -> None:
    tracker = LifecycleTracker()
    tracker.register("call-1", "hot_info", 1)
    item = FunctionCallOutputItem(meta=_meta(), call_id="call-1", output=LONG_TEXT, tool_name="web_search")

    result = apply_request_policy([item], tracker, 10)

    assert "已过期" in result[0].output
    assert item.output == LONG_TEXT  # original frozen item stays untouched


def test_short_hot_info_is_not_replaced() -> None:
    tracker = LifecycleTracker()
    tracker.register("call-2", "hot_info", 1)
    item = FunctionCallOutputItem(meta=_meta(), call_id="call-2", output="短结果", tool_name="web_fetch")

    result = apply_request_policy([item], tracker, 10)

    assert result[0].output == "短结果"


def test_image_becomes_text_after_expiry() -> None:
    tracker = LifecycleTracker()
    item = UserMessageItem(
        meta=_meta(),
        parts=(ContextTextPart("看图"), ContextImagePart(image_format="png", image_base64="aGVsbG8=")),
    )
    tracker.register(item.meta.item_id, "image", 1)

    result = apply_request_policy([item], tracker, 10)

    assert any(isinstance(part, ContextTextPart) and "图片已过期" in part.text for part in result[0].parts)
    assert any(isinstance(part, ContextImagePart) for part in item.parts)


def test_image_is_kept_before_expiry() -> None:
    tracker = LifecycleTracker()
    item = UserMessageItem(meta=_meta(), parts=(ContextImagePart(image_format="png", image_base64="aGVsbG8="),))
    tracker.register(item.meta.item_id, "image", 5)

    result = apply_request_policy([item], tracker, 6)

    assert any(isinstance(part, ContextImagePart) for part in result[0].parts)


def test_repeated_text_is_folded() -> None:
    previous = LONG_TEXT + "甲"
    current = LONG_TEXT + "乙"
    folded = fold_repeated_text(previous, current)

    assert folded is not None
    assert "已折叠" in folded
    assert len(folded) < len(current)


def test_short_overlap_is_not_folded() -> None:
    assert fold_repeated_text("abc" * 10, "abc" * 10 + "x") is None


def _emoji_item(base64_data: str) -> UserMessageItem:
    return UserMessageItem(
        meta=_meta(),
        parts=(ContextImagePart(image_format="gif", image_base64=base64_data),),
    )


def test_repeated_emoji_shows_image_once_and_keeps_description() -> None:
    tracker = LifecycleTracker()
    first = _emoji_item("c2FtZQ==")
    second = _emoji_item("c2FtZQ==")
    tracker.register(first.meta.item_id, "emoji", 1, description="猫猫歪头笑")
    tracker.register(second.meta.item_id, "emoji", 1, description="猫猫歪头笑")

    result = apply_request_policy([first, second], tracker, 2)

    first_parts = result[0].parts
    assert any(isinstance(part, ContextImagePart) for part in first_parts)
    assert any(isinstance(part, ContextTextPart) and "猫猫歪头笑" in part.text for part in first_parts)
    second_parts = result[1].parts
    assert not any(isinstance(part, ContextImagePart) for part in second_parts)
    assert any(isinstance(part, ContextTextPart) and "猫猫歪头笑" in part.text for part in second_parts)


def test_unique_emoji_keeps_only_the_image() -> None:
    tracker = LifecycleTracker()
    item = _emoji_item("dW5pcXVl")
    tracker.register(item.meta.item_id, "emoji", 1, description="猫猫歪头笑")

    result = apply_request_policy([item], tracker, 2)

    parts = result[0].parts
    assert any(isinstance(part, ContextImagePart) for part in parts)
    assert not any(isinstance(part, ContextTextPart) and "猫猫歪头笑" in part.text for part in parts)


def test_repeated_emoji_without_description_falls_back_to_text() -> None:
    tracker = LifecycleTracker()
    first = _emoji_item("bm9kZXNj")
    second = _emoji_item("bm9kZXNj")
    tracker.register(first.meta.item_id, "emoji", 1)
    tracker.register(second.meta.item_id, "emoji", 1)

    result = apply_request_policy([first, second], tracker, 2)

    assert any(isinstance(part, ContextTextPart) and "相同的表情" in part.text for part in result[1].parts)
