"""Profile injection shows the real name and skips unchanged profiles."""

from src.config.config import global_config
from src.maisaka.agent_core.user_alias import UserAliasBook
from src.maisaka.memory import person_profile
from src.person_info.person_info import get_person_id

import pytest


def test_strip_identity_suffix_drops_person_id_hashes() -> None:
    assert person_profile.strip_identity_suffix("小鲸鱼 [59210e0e8b6928d657b160ca80ab71f5]") == "小鲸鱼"
    assert person_profile.strip_identity_suffix("小鲸鱼 [59210e0e]") == "小鲸鱼"
    assert person_profile.strip_identity_suffix("小鲸鱼") == "小鲸鱼"


def test_rewrite_profile_names_prefers_group_remark(monkeypatch, tmp_path) -> None:
    book = UserAliasBook(tmp_path / "aliases.json")
    book.set_display_name("qq", "1785941890", "张峻峰")
    monkeypatch.setattr(person_profile, "get_user_alias_book", lambda: book)
    person_id = get_person_id("qq", "1785941890")

    rewritten = person_profile.rewrite_profile_names(
        "# 人物画像\n主称呼: zachary\n别名: zachary、张峻峰\n内容\n",
        person_id,
    )

    assert "主称呼: 张峻峰（QQ 昵称 zachary）" in rewritten


def test_rewrite_profile_names_without_alias_only_strips_hashes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(person_profile, "get_user_alias_book", lambda: UserAliasBook(tmp_path / "aliases.json"))

    rewritten = person_profile.rewrite_profile_names("主称呼: 小鲸鱼 [59210e0e8b6928d657b160ca80ab71f5]", "unknown")

    assert rewritten == "主称呼: 小鲸鱼"


def _candidate() -> person_profile.PersonProfileCandidate:
    return person_profile.PersonProfileCandidate(person_id="pid-1", person_name="zachary", user_id="1")


@pytest.fixture
def profile_env(monkeypatch, tmp_path):
    monkeypatch.setattr(global_config.a_memorix.integration, "enable_person_profile_injection", True)
    monkeypatch.setattr(person_profile, "get_user_alias_book", lambda: UserAliasBook(tmp_path / "aliases.json"))
    monkeypatch.setattr(person_profile, "collect_person_profile_candidates", lambda *args, **kwargs: [_candidate()])
    return monkeypatch


def _fake_profile(monkeypatch, text: str) -> None:
    async def fake_admin(**kwargs):
        return {"success": True, "profile_text": text}

    monkeypatch.setattr(person_profile.memory_service, "profile_admin", fake_admin)


@pytest.mark.asyncio
async def test_unchanged_profile_is_not_injected_twice(profile_env) -> None:
    _fake_profile(profile_env, "# 人物画像\n主称呼: zachary\n稳定了解")

    blocks, injected = await person_profile.build_person_profile_injection_messages(anchor_message=None)
    assert blocks
    assert "pid-1" in injected

    blocks_again, injected_again = await person_profile.build_person_profile_injection_messages(
        anchor_message=None, known_profiles=injected
    )
    assert blocks_again == []
    assert injected_again == {}


@pytest.mark.asyncio
async def test_refreshed_profile_is_injected_again(profile_env) -> None:
    _fake_profile(profile_env, "# 人物画像\n主称呼: zachary\n稳定了解")
    _, injected = await person_profile.build_person_profile_injection_messages(anchor_message=None)

    _fake_profile(profile_env, "# 人物画像\n主称呼: zachary\n稳定了解\n新增：喜欢打篮球")
    blocks, refreshed = await person_profile.build_person_profile_injection_messages(
        anchor_message=None, known_profiles=injected
    )

    assert blocks
    assert refreshed["pid-1"] != injected["pid-1"]
