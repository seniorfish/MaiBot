"""Context lifecycle policies: hot-info stripping, dedupe and media expiry.

These functions are pure so they can be applied to request items without
touching the frozen ``ContextItem`` model or rewriting history.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Optional

HOT_INFO_MIN_CHARS = 300
HOT_INFO_EVENTS = 3
IMAGE_EVENTS = 3
IMAGE_REINTRODUCED_EVENTS = 5
DEDUPE_MIN_CHARS = 200
DEDUPE_EDGE_CHARS = 30
# A repeated emoji is worth at most a one-line description once it was shown.
EMOJI_DESCRIPTION_CHARS = 50

IMAGE_KINDS = {"image", "emoji"}
HOT_INFO_KINDS = {"web_search", "web_fetch", "tool_result", "hot_info"}


@dataclass(slots=True)
class LifecycleEntry:
    kind: str
    inserted_seq: int
    reintroduced: bool = False
    last_seen_seq: int = 0


class LifecycleTracker:
    """Sidecar bookkeeping keyed by message id or context item id."""

    def __init__(self, data: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._data: Dict[str, Dict[str, Any]] = data if data is not None else {}

    @property
    def data(self) -> Dict[str, Dict[str, Any]]:
        return self._data

    def register(
        self,
        key: str,
        kind: str,
        event_seq: int,
        *,
        reintroduced: bool = False,
        description: str = "",
    ) -> None:
        existing = self._data.get(key)
        if existing is None:
            self._data[key] = {
                "kind": kind,
                "inserted_seq": event_seq,
                "reintroduced": reintroduced,
                "last_seen_seq": event_seq,
            }
            existing = self._data[key]
        else:
            existing["last_seen_seq"] = event_seq
            if reintroduced:
                existing["inserted_seq"] = event_seq
                existing["reintroduced"] = True
        if description:
            existing["description"] = description[:EMOJI_DESCRIPTION_CHARS]

    def entry(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def should_expire(self, key: str, current_seq: int) -> bool:
        entry = self._data.get(key)
        if entry is None:
            return False
        kind = str(entry.get("kind") or "")
        if kind in IMAGE_KINDS:
            limit = IMAGE_REINTRODUCED_EVENTS if entry.get("reintroduced") else IMAGE_EVENTS
        elif kind in HOT_INFO_KINDS:
            limit = HOT_INFO_EVENTS
        else:
            return False
        inserted = int(entry.get("inserted_seq") or 0)
        return current_seq - inserted >= limit

    def expired_keys(self, current_seq: int, kind: Optional[str] = None) -> list[str]:
        keys = []
        for key, entry in self._data.items():
            if kind is not None and entry.get("kind") != kind:
                continue
            if self.should_expire(key, current_seq):
                keys.append(key)
        return keys

    def forget(self, key: str) -> None:
        self._data.pop(key, None)


def should_strip_hot_info(text: str) -> bool:
    """Only long tool output is worth summarizing; short results stay verbatim."""

    return len(text.strip()) >= HOT_INFO_MIN_CHARS


def fold_repeated_text(previous: str, current: str) -> Optional[str]:
    """Fold ``current`` if it repeats >=200 chars of ``previous``.

    Returns the folded text (head/tail plus a marker), or ``None`` when the
    overlap is below the threshold.
    """

    if len(previous.strip()) < DEDUPE_MIN_CHARS or len(current.strip()) < DEDUPE_MIN_CHARS:
        return None
    matcher = SequenceMatcher(None, previous, current, autojunk=False)
    match = matcher.find_longest_match(0, len(previous), 0, len(current))
    if match.size < DEDUPE_MIN_CHARS:
        return None
    head = current[:DEDUPE_EDGE_CHARS].strip()
    tail = current[-DEDUPE_EDGE_CHARS:].strip()
    return f"{head}…[与前文重复 {match.size} 字符，已折叠]…{tail}"


def dedupe_key(role: str, text: str) -> str:
    """Stable identity for identical content in the same role."""

    return f"{role}:{hash(text.strip())}"


def _item_text(item: Any) -> str:
    """Plain text of a message item, used only for duplicate detection."""

    parts = getattr(item, "parts", None)
    if not parts:
        return ""
    return "".join(getattr(part, "text", "") or "" for part in parts)


def apply_request_policy(items: list[Any], tracker: LifecycleTracker, current_seq: int) -> list[Any]:
    """Apply media expiry, hot-info stripping and duplicate folding to request items.

    The input items are frozen dataclasses; replacements use ``dataclasses.replace``.
    Function-call/output pairs are never dropped, so the request stays valid.
    """

    from dataclasses import replace

    from src.llm_models.payload_content.context_item import (
        AssistantMessageItem,
        ContextImagePart,
        ContextTextPart,
        FunctionCallOutputItem,
        UserMessageItem,
    )

    result: list[Any] = []
    previous_text = ""
    # The same emoji often appears several times; show its image once.
    emoji_counts: Dict[int, int] = {}
    for item in items:
        if not isinstance(item, (UserMessageItem, AssistantMessageItem)):
            continue
        item_id = getattr(getattr(item, "meta", None), "item_id", "")
        entry = tracker.entry(item_id) if item_id else None
        if not entry or entry.get("kind") != "emoji":
            continue
        for part in item.parts:
            if isinstance(part, ContextImagePart):
                digest = hash(part.image_base64)
                emoji_counts[digest] = emoji_counts.get(digest, 0) + 1
    seen_emoji: set[int] = set()
    for item in items:
        if (
            isinstance(item, FunctionCallOutputItem)
            and should_strip_hot_info(item.output)
            and tracker.should_expire(item.call_id, current_seq)
        ):
            result.append(replace(item, output=_expired_tool_notice(item.output)))
            continue

        if isinstance(item, (UserMessageItem, AssistantMessageItem)):
            item_id = getattr(getattr(item, "meta", None), "item_id", "")
            replaced = False
            if item_id and tracker.should_expire(item_id, current_seq):
                parts = tuple(
                    ContextTextPart("[图片已过期，可用 read_image 重新查看]")
                    if isinstance(part, ContextImagePart)
                    else part
                    for part in item.parts
                )
                if parts != item.parts:
                    extra = {"replay": None} if isinstance(item, AssistantMessageItem) else {}
                    result.append(replace(item, parts=parts, **extra))
                    replaced = True
            if replaced:
                previous_text = _item_text(item)
                continue

            entry = tracker.entry(item_id) if item_id else None
            if entry is not None and entry.get("kind") == "emoji":
                digest = next(
                    (hash(part.image_base64) for part in item.parts if isinstance(part, ContextImagePart)),
                    None,
                )
                if digest is not None:
                    description = str(entry.get("description") or "").strip()
                    extra = {"replay": None} if isinstance(item, AssistantMessageItem) else {}
                    if digest in seen_emoji:
                        parts = tuple(
                            ContextTextPart(f"[表情：{description}]" if description else "[与前面相同的表情]")
                            if isinstance(part, ContextImagePart)
                            else part
                            for part in item.parts
                        )
                        result.append(replace(item, parts=parts, **extra))
                        previous_text = _item_text(item)
                        continue
                    seen_emoji.add(digest)
                    if description and emoji_counts.get(digest, 0) > 1:
                        # First appearance: image plus a one-line description.
                        result.append(
                            replace(item, parts=(*item.parts, ContextTextPart(f"（表情：{description}）")), **extra)
                        )
                        previous_text = _item_text(item)
                        continue

            text = _item_text(item)
            if previous_text and isinstance(item, UserMessageItem):
                folded = fold_repeated_text(previous_text, text)
                if folded:
                    result.append(replace(item, parts=(ContextTextPart(folded),)))
                    previous_text = text
                    continue
            if text:
                previous_text = text

        result.append(item)
    return result


def _expired_tool_notice(output: str) -> str:
    head = output.strip()[:DEDUPE_EDGE_CHARS]
    return f"{head}…[早期工具结果已过期，如需细节请重新调用工具]"
