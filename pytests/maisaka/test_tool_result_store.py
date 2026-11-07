"""Tool result store keeps ids stable across restarts when persisted."""

from src.maisaka.tool_result_store import ToolResultStore


def test_persisted_results_survive_restart(tmp_path) -> None:
    first = ToolResultStore(tmp_path)
    result_id = first.save("session", "web_search", "raw content")

    second = ToolResultStore(tmp_path)
    restored = second.get("session", result_id)

    assert restored is not None
    assert restored.content == "raw content"
    assert restored.tool_name == "web_search"


def test_ids_continue_after_restart(tmp_path) -> None:
    first = ToolResultStore(tmp_path)
    first.save("session", "web_search", "one")
    first.save("session", "web_search", "two")

    second = ToolResultStore(tmp_path)
    assert second.save("session", "web_fetch", "three") == "r3"


def test_memory_only_store_has_no_files(tmp_path) -> None:
    store = ToolResultStore()
    result_id = store.save("session", "web_search", "content")

    assert store.get("session", result_id) is not None
    assert not list(tmp_path.glob("**/*.json"))


def test_old_files_are_pruned_with_lru(tmp_path) -> None:
    store = ToolResultStore(tmp_path)
    ids = [store.save("session", "web_search", f"content-{index}") for index in range(25)]

    assert store.get("session", ids[0]) is None
    assert store.get("session", ids[-1]) is not None
    assert len(list((tmp_path).glob("*/r*.json"))) <= 20
