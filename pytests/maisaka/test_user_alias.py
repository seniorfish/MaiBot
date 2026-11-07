"""Stable short user ids and the display-name book behind message headers."""

from src.maisaka.agent_core.user_alias import UserAliasBook, short_person_id
from src.person_info.person_info import get_person_id


def test_short_person_id_is_stable_and_prefix_of_full_id() -> None:
    short = short_person_id("qq", "12345")

    assert len(short) == 8
    assert get_person_id("qq", "12345").startswith(short)
    assert short == short_person_id("qq", "12345")
    assert short_person_id("qq", "") == ""


def test_observe_keeps_the_first_display_name(tmp_path) -> None:
    book = UserAliasBook(tmp_path / "aliases.json")

    book.observe("qq", "1001", "小明")
    book.observe("qq", "1001", "小明改了个名")

    assert book.display_name("qq", "1001") == "小明"
    assert book.dirty is True


def test_resolve_person_id_by_name_alias_and_uid_prefix(tmp_path) -> None:
    book = UserAliasBook(tmp_path / "aliases.json")
    person_id = book.set_display_name("qq", "1001", "小明", aliases=["明明"])

    assert book.resolve_person_id("小明") == person_id
    assert book.resolve_person_id("明明") == person_id
    assert book.resolve_person_id(person_id[:8]) == person_id
    assert book.resolve_person_id("不存在") == ""


def test_short_prefix_must_be_unambiguous(tmp_path) -> None:
    book = UserAliasBook(tmp_path / "aliases.json")
    book.observe("qq", "1001", "甲")
    book.observe("qq", "1002", "乙")

    assert book.resolve_person_id("1") == ""


def test_roundtrip_clears_dirty(tmp_path) -> None:
    path = tmp_path / "aliases.json"
    book = UserAliasBook(path)
    book.observe("qq", "1001", "小明")
    book.save()

    assert book.dirty is False
    reloaded = UserAliasBook(path)
    assert reloaded.display_name("qq", "1001") == "小明"


def test_group_card_overrides_earlier_nickname(tmp_path) -> None:
    book = UserAliasBook(tmp_path / "aliases.json")

    book.observe("qq", "1001", "小明的网名")
    book.observe("qq", "1001", "张三", authoritative=True)

    assert book.display_name("qq", "1001") == "张三"


def test_punctuation_card_does_not_become_a_name(tmp_path) -> None:
    book = UserAliasBook(tmp_path / "aliases.json")

    book.observe("qq", "1001", "网名")
    book.observe("qq", "1001", "。", authoritative=True)

    assert book.display_name("qq", "1001") == "网名"


def test_at_target_accepts_uid_and_display_name(monkeypatch, tmp_path) -> None:
    """@ 目标给 uid 或名字也要能解析，不能只认 msg_id。

    消息头里给模型看的是 8 位 uid，模型经常直接把它当 @ 目标传回来。
    曾经因此整条发送失败（「无法解析 at 目标消息」）。
    """

    from src.maisaka.agent_core import user_alias as user_alias_module
    from src.maisaka.builtin_tool.context import _resolve_at_user_id

    book = UserAliasBook(tmp_path / "aliases.json")
    person_id = book.set_display_name("qq", "1001", "杨梓豪")
    monkeypatch.setattr(user_alias_module, "get_user_alias_book", lambda: book)

    assert _resolve_at_user_id(person_id[:8]) == "1001"
    assert _resolve_at_user_id("杨梓豪") == "1001"
    assert _resolve_at_user_id("1001") == "1001"
    assert _resolve_at_user_id("查无此人") == ""
    assert _resolve_at_user_id("") == ""
