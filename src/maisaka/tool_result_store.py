"""被整理过的工具原始结果的暂存。

超长工具结果写入 Planner 历史时只保留子代理整理后的结论，原文暂存在这里，
Planner 需要整理里没有的细节时可以用 read_tool_result 工具按编号取回。

原文同时保存在内存与磁盘（可按需关闭磁盘）：每个聊天保留最近若干条，
进程重启后仍能按编号取回，编号从已有文件的编号之后继续，避免旧引用指向新内容。
"""

import json
import os
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Optional

from src.common.paths import get_instance_root

# 每个聊天最多暂存的原始结果条数，超出后丢弃最早的
MAX_STORED_RESULTS_PER_SESSION = 20


@dataclass(frozen=True, slots=True)
class StoredToolResult:
    """一条暂存的工具原始结果。"""

    result_id: str
    tool_name: str
    content: str
    created_at: datetime


class ToolResultStore:
    """按聊天分组暂存工具原始结果，可选落盘以支持重启后回查。"""

    def __init__(self, root: Optional[Path] = None) -> None:
        self._results_by_session: Dict[str, "OrderedDict[str, StoredToolResult]"] = {}
        self._next_index_by_session: Dict[str, int] = {}
        self._root = root

    def save(self, session_id: str, tool_name: str, content: str) -> str:
        """暂存一条原始结果，返回在该聊天内唯一的短编号（如 ``r3``）。"""

        next_index = self._next_index_by_session.get(session_id, 0) + 1
        if next_index == 1 and self._root is not None:
            next_index = self._scan_existing_index(session_id) + 1
        self._next_index_by_session[session_id] = next_index
        result_id = f"r{next_index}"
        stored = StoredToolResult(
            result_id=result_id,
            tool_name=tool_name,
            content=content,
            created_at=datetime.now(),
        )
        session_results = self._results_by_session.setdefault(session_id, OrderedDict())
        session_results[result_id] = stored
        while len(session_results) > MAX_STORED_RESULTS_PER_SESSION:
            dropped_id, _ = session_results.popitem(last=False)
            self._delete_file(session_id, dropped_id)
        self._write_file(session_id, stored)
        return result_id

    def get(self, session_id: str, result_id: str) -> Optional[StoredToolResult]:
        """按编号取回原始结果；已被淘汰且不在磁盘上时返回 None。"""

        normalized = result_id.strip()
        cached = self._results_by_session.get(session_id, OrderedDict()).get(normalized)
        if cached is not None:
            return cached
        restored = self._read_file(session_id, normalized)
        if restored is None:
            return None
        session_results = self._results_by_session.setdefault(session_id, OrderedDict())
        session_results[normalized] = restored
        while len(session_results) > MAX_STORED_RESULTS_PER_SESSION:
            session_results.popitem(last=False)
        return restored

    def _session_dir(self, session_id: str) -> Optional[Path]:
        if self._root is None:
            return None
        return self._root / sha256(session_id.encode()).hexdigest()[:16]

    def _path_for(self, session_id: str, result_id: str) -> Optional[Path]:
        directory = self._session_dir(session_id)
        return None if directory is None else directory / f"{result_id}.json"

    def _scan_existing_index(self, session_id: str) -> int:
        directory = self._session_dir(session_id)
        if directory is None or not directory.exists():
            return 0
        highest = 0
        for path in directory.glob("r*.json"):
            try:
                highest = max(highest, int(path.stem[1:]))
            except ValueError:
                continue
        return highest

    def _write_file(self, session_id: str, stored: StoredToolResult) -> None:
        path = self._path_for(session_id, stored.result_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        payload = {
            "result_id": stored.result_id,
            "tool_name": stored.tool_name,
            "content": stored.content,
            "created_at": stored.created_at.isoformat(),
        }
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            pass
        finally:
            temporary.unlink(missing_ok=True)

    def _read_file(self, session_id: str, result_id: str) -> Optional[StoredToolResult]:
        path = self._path_for(session_id, result_id)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return StoredToolResult(
                result_id=str(payload["result_id"]),
                tool_name=str(payload.get("tool_name") or ""),
                content=str(payload.get("content") or ""),
                created_at=datetime.fromisoformat(str(payload["created_at"])),
            )
        except (OSError, ValueError, KeyError):
            return None

    def _delete_file(self, session_id: str, result_id: str) -> None:
        path = self._path_for(session_id, result_id)
        if path is not None:
            path.unlink(missing_ok=True)


tool_result_store = ToolResultStore(get_instance_root() / "data" / "tool_results")
