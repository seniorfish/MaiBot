"""Event model and in-memory event queues for the agent core.

An event is the basic unit of a wake-up. A single wake-up may contain several
tool calls and any same-session messages that arrive while it is running; those
do not start a new event.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional
from uuid import uuid4


class EventKind(str, Enum):
    """Kinds of events that can wake the agent."""

    MESSAGE = "message"
    MENTION = "mention"
    WAIT_TIMEOUT = "wait_timeout"
    PROACTIVE = "proactive"
    SUBAGENT_RESULT = "subagent_result"
    SENT_MESSAGE = "sent_message"
    FOCUS_SWITCH = "focus_switch"
    # Internal wake signal: it only tells the consumer to look again, it is not
    # an inbound message and must never block sending.
    INTERNAL = "internal"


# Kinds that come from another person and therefore carry unread message content.
INBOUND_KINDS = frozenset({EventKind.MESSAGE, EventKind.MENTION})


@dataclass(slots=True, frozen=True)
class Event:
    """A single wake-up unit.

    ``scope_id`` groups sessions that share one agent loop (a normal chat is a
    single-session scope, Focus is a multi-session scope). ``dedupe_key`` makes
    enqueue idempotent across reconnects and restarts.
    """

    kind: EventKind
    scope_id: str
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    dedupe_key: str = ""
    is_self: bool = False
    source: str = ""
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if not self.scope_id:
            raise ValueError("event scope_id must not be empty")
        if not self.session_id:
            raise ValueError("event session_id must not be empty")
        if not self.dedupe_key:
            object.__setattr__(self, "dedupe_key", self.event_id)

    @property
    def wakes_agent(self) -> bool:
        """Self-originated events update context but never wake the agent."""

        return not self.is_self

    @property
    def message_id(self) -> str:
        value = self.payload.get("message_id")
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)

    @property
    def is_inbound(self) -> bool:
        """Only real inbound messages count as unread conversation content."""

        return self.kind in INBOUND_KINDS


class EventQueue:
    """Per-scope FIFO queues with dedupe and same-session packing."""

    def __init__(self, *, max_dedupe_keys: int = 4096) -> None:
        self._queues: Dict[str, Deque[Event]] = {}
        self._dedupe: Dict[str, float] = {}
        self._max_dedupe_keys = max_dedupe_keys
        self._wake = asyncio.Event()

    def put(self, event: Event) -> bool:
        """Enqueue an event; return False when it is a duplicate or self event."""

        if not event.wakes_agent:
            return False
        if event.dedupe_key in self._dedupe:
            return False
        self._trim_dedupe()
        self._dedupe[event.dedupe_key] = event.created_at
        self._queues.setdefault(event.scope_id, deque()).append(event)
        self._wake.set()
        return True

    def put_many(self, events: Iterable[Event]) -> int:
        return sum(1 for event in events if self.put(event))

    async def wait_until_pending(self, scope_id: str, timeout: Optional[float] = None) -> None:
        """Wait until the scope has at least one event."""

        if self.pending_count(scope_id) > 0:
            return
        self._wake.clear()
        if self.pending_count(scope_id) > 0:
            return
        if timeout is None:
            await self._wake.wait()
        else:
            await asyncio.wait_for(self._wake.wait(), timeout=timeout)

    def pending_count(self, scope_id: str) -> int:
        return len(self._queues.get(scope_id, ()))

    def has_unread(self, scope_id: str, session_id: str = "") -> bool:
        return bool(self.unread_for_session(scope_id, session_id))

    def unread_for_session(self, scope_id: str, session_id: str = "") -> List[Event]:
        """Pending inbound messages; internal wake signals are not unread chat."""

        return [
            event
            for event in self._queues.get(scope_id, ())
            if event.is_inbound and (not session_id or event.session_id == session_id)
        ]

    def drain_for_sleep(self, scope_id: str) -> List[Event]:
        """Take the head event plus every event of the same session.

        Other sessions stay queued until the next sleep, which keeps a single
        wake-up focused on one conversation.
        """

        queue = self._queues.get(scope_id)
        if not queue:
            return []
        head = queue.popleft()
        packed = [head]
        remaining: Deque[Event] = deque()
        while queue:
            event = queue.popleft()
            if event.session_id == head.session_id:
                packed.append(event)
            else:
                remaining.append(event)
        if remaining:
            self._queues[scope_id] = remaining
        else:
            self._queues.pop(scope_id, None)
        return packed

    def drain_all(self, scope_id: str) -> List[Event]:
        queue = self._queues.pop(scope_id, None)
        return list(queue) if queue else []

    def remove_matching(self, scope_id: str, predicate: Callable[[Event], bool]) -> List[Event]:
        """Take only the events matching a predicate; the rest stay queued."""

        queue = self._queues.get(scope_id)
        if not queue:
            return []
        removed = [event for event in queue if predicate(event)]
        if not removed:
            return []
        remaining: Deque[Event] = deque(event for event in queue if not predicate(event))
        if remaining:
            self._queues[scope_id] = remaining
        else:
            self._queues.pop(scope_id, None)
        return removed

    def clear_scope(self, scope_id: str) -> None:
        self._queues.pop(scope_id, None)

    def _trim_dedupe(self) -> None:
        if len(self._dedupe) < self._max_dedupe_keys:
            return
        # Keep the most recent half; insertion order is creation order.
        keep = self._max_dedupe_keys // 2
        for key in list(self._dedupe)[: len(self._dedupe) - keep]:
            self._dedupe.pop(key, None)

    def snapshot(self, scope_id: str) -> List[Dict[str, Any]]:
        """Serialize pending events for a future checkpoint (stage 5)."""

        return [
            {
                "event_id": event.event_id,
                "kind": event.kind.value,
                "scope_id": event.scope_id,
                "session_id": event.session_id,
                "payload": dict(event.payload),
                "dedupe_key": event.dedupe_key,
                "is_self": event.is_self,
                "source": event.source,
                "created_at": event.created_at,
            }
            for event in self._queues.get(scope_id, ())
        ]

    @classmethod
    def restore(cls, scope_id: str, items: Iterable[Dict[str, Any]]) -> "EventQueue":
        queue = cls()
        queue.restore_into(items)
        return queue

    def restore_into(self, items: Iterable[Dict[str, Any]]) -> int:
        """Merge serialized events into this queue; return how many were added."""

        added = 0
        for item in items:
            try:
                event = Event(
                    kind=EventKind(str(item["kind"])),
                    scope_id=str(item["scope_id"]),
                    session_id=str(item["session_id"]),
                    payload=dict(item.get("payload") or {}),
                    dedupe_key=str(item.get("dedupe_key") or ""),
                    is_self=bool(item.get("is_self")),
                    source=str(item.get("source") or ""),
                    event_id=str(item.get("event_id") or uuid4().hex),
                    created_at=float(item.get("created_at") or time()),
                )
            except (KeyError, ValueError):
                continue
            if self.put(event):
                added += 1
        return added

    def earliest_pending_at(self, scope_id: str) -> Optional[float]:
        queue = self._queues.get(scope_id)
        return queue[0].created_at if queue else None
