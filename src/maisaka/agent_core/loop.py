"""Single event consumer shared by normal chats and Focus groups.

Both consumers block on ``wait_for_batch`` and then run their own turn logic,
so one scope still has exactly one place where wake-ups are produced.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from time import time
from typing import Callable, List, Optional

from .events import Event, EventQueue
from .state import EVENT_OFFERED, AgentState


class AgentLoop:
    """Owns one scope's event queue and state."""

    def __init__(
        self,
        scope_id: str,
        *,
        queue: Optional[EventQueue] = None,
        state: Optional[AgentState] = None,
    ) -> None:
        self.scope_id = scope_id
        self.queue = queue or EventQueue()
        self.state = state or AgentState()
        self._running = False
        self._wake = asyncio.Event()
        self._scheduled: set[asyncio.Task[None]] = set()

    def enqueue(self, event: Event) -> bool:
        if event.scope_id != self.scope_id:
            raise ValueError(f"event scope {event.scope_id} does not match loop scope {self.scope_id}")
        accepted = self.queue.put(event)
        if accepted:
            self.state.mark(event.event_id, "received")
            self._wake.set()
        return accepted

    def pending_count(self) -> int:
        return self.queue.pending_count(self.scope_id)

    def schedule_event(self, event: Event, *, delay_seconds: float) -> asyncio.Task[None]:
        """Enqueue an event after a delay (a self-reminder or a timer)."""

        if event.scope_id != self.scope_id:
            raise ValueError(f"event scope {event.scope_id} does not match loop scope {self.scope_id}")

        async def _fire() -> None:
            await asyncio.sleep(max(0.0, delay_seconds))
            # created_at must reflect the wake-up, not the scheduling time.
            self.enqueue(replace(event, created_at=time()))

        task = asyncio.create_task(_fire())
        self._scheduled.add(task)
        task.add_done_callback(self._scheduled.discard)
        return task

    def cancel_scheduled(self) -> None:
        """Drop pending timers; used when the scope goes away."""

        for task in list(self._scheduled):
            task.cancel()
        self._scheduled.clear()

    def has_unread_for(self, session_id: str) -> bool:
        return self.queue.has_unread(self.scope_id, session_id)

    async def wait_for_batch(self, timeout: Optional[float] = None) -> List[Event]:
        """Block until at least one event is queued, then take a batch."""

        if self.queue.pending_count(self.scope_id) == 0:
            await self.queue.wait_until_pending(self.scope_id, timeout=timeout)
        return self.take_next_batch()

    def take_next_batch(self) -> List[Event]:
        """Take the head event and same-session events for one wake-up."""

        batch = self.queue.drain_for_sleep(self.scope_id)
        for event in batch:
            self.state.mark_offered(event.event_id)
        return batch

    def take_all(self) -> List[Event]:
        """Take every pending event; used by consumers that gate readiness themselves."""

        batch = self.queue.drain_all(self.scope_id)
        for event in batch:
            self.state.mark_offered(event.event_id)
        return batch

    def drain_pending(self) -> List[Event]:
        """Take and finish every pending event (they only signalled a wake-up)."""

        batch = self.take_all()
        if batch:
            self.finish_batch(batch)
        return batch

    def finish_events(self, predicate: Callable[[Event], bool]) -> int:
        """Finish events whose work already happened elsewhere (e.g. ingested messages)."""

        removed = self.queue.remove_matching(self.scope_id, predicate)
        for event in removed:
            self.state.mark_handled(event.event_id)
        return len(removed)

    def finish_batch(self, batch: List[Event], *, ignored: bool = False) -> None:
        """Mark a batch as handled or ignored (both end the event)."""

        for event in batch:
            if ignored:
                self.state.mark_ignored(event.event_id)
            else:
                self.state.mark_handled(event.event_id)

    def sleeping(self, session_id: str) -> bool:
        return self.state.is_waiting(session_id)
