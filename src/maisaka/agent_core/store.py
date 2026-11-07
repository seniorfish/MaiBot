"""Atomic checkpoint store for agent state and pending events.

The file layout mirrors ``FocusCheckpointStore``: one JSON file per scope,
written to a temporary file and atomically replaced.
"""

from __future__ import annotations

import json
import os
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.paths import get_instance_root

CHECKPOINT_VERSION = 1


class AgentStateStore:
    """Persist one scope's agent state and queued events."""

    def __init__(self, scope_id: str, root: Optional[Path] = None) -> None:
        self.scope_id = scope_id
        self.root = root or (get_instance_root() / "data" / "agent_core")
        self.path = self.root / (sha256(scope_id.encode()).hexdigest() + ".json")

    def load(self) -> Dict[str, Any]:
        """Return the checkpoint payload, or an empty dict when absent/invalid."""

        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if payload.get("version") != CHECKPOINT_VERSION or payload.get("scope_id") != self.scope_id:
            return {}
        return payload

    def save(self, state: Dict[str, Any], events: List[Dict[str, Any]]) -> None:
        payload = {
            "version": CHECKPOINT_VERSION,
            "scope_id": self.scope_id,
            "state": state,
            "events": events,
        }
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def delete(self) -> None:
        self.path.unlink(missing_ok=True)
