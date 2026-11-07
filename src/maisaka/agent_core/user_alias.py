"""Stable user identity shown to the model.

``person_info.get_person_id`` already provides a cross-session stable id; this
book only adds the single display name the model should see for that person.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from src.common.logger import get_logger
from src.common.paths import get_instance_root
from src.person_info.person_info import get_person_id

logger = get_logger("maisaka_user_alias")

BOOK_VERSION = 1
# Short stable handle shown in message headers: md5 person ids are too long to
# repeat on every message, and eight characters are enough for one class.
ALIAS_ID_LENGTH = 8
MIN_ALIAS_PREFIX = 6

_BOOK: Optional["UserAliasBook"] = None


def get_user_alias_book() -> "UserAliasBook":
    """Process-wide alias book; the hot path only reads from it."""

    global _BOOK
    if _BOOK is None:
        _BOOK = UserAliasBook()
    return _BOOK


def short_person_id(platform: str, user_id: str) -> str:
    """Short, stable id the model can hand back to query_person_profile."""

    if not str(user_id).strip():
        return ""
    return get_person_id(platform, str(user_id))[:ALIAS_ID_LENGTH]


class UserAliasBook:
    """JSON-backed ``person_id -> display name`` mapping."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (get_instance_root() / "data" / "user_aliases.json")
        self._people: Dict[str, Dict[str, Any]] = {}
        self._dirty = False
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self._seed_from_person_info()
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if payload.get("version") != BOOK_VERSION:
            return
        people = payload.get("people")
        if isinstance(people, dict):
            self._people = {str(key): dict(value) for key, value in people.items() if isinstance(value, dict)}

    def _seed_from_person_info(self) -> None:
        """Seed display names from the people the bot already knows.

        ``person_info.group_cardname`` keeps the card of every group the person
        spoke in, and a class group sets those cards to real names, so the card
        beats the QQ nickname. Only the real instance file is seeded; tests pass
        their own temp path.
        """

        if self.path != get_instance_root() / "data" / "user_aliases.json":
            return
        from sqlmodel import select

        from src.common.data_models.person_info_data_model import parse_group_cardname_json
        from src.common.database.database import get_db_session
        from src.common.database.database_model import PersonInfo

        try:
            with get_db_session() as session:
                rows = session.exec(select(PersonInfo)).all()
        except Exception as exc:
            logger.warning(f"读取人物备注名失败，用户备注名从空开始: {exc}")
            return
        for row in rows:
            person_id = str(row.person_id or "").strip()
            if not person_id:
                continue
            name = ""
            try:
                for card in parse_group_cardname_json(row.group_cardname) or []:
                    if len(card.group_cardname.strip()) >= 2:
                        name = card.group_cardname.strip()
                        break
            except ValueError:
                name = ""
            name = name or str(row.person_name or "").strip()
            if not name:
                continue
            self._people[person_id] = {
                "platform": str(row.platform or ""),
                "user_id": str(row.user_id or ""),
                "aliases": [],
                "display_name": name,
            }
            self._dirty = True
        if self._people:
            logger.info(f"已从人物库载入 {len(self._people)} 个用户备注名")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": BOOK_VERSION, "people": self._people}
        temporary = self.path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
            self._dirty = False
        finally:
            temporary.unlink(missing_ok=True)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def observe(self, platform: str, user_id: str, name: str = "", *, authoritative: bool = False) -> str:
        """Remember a person seen in chat without writing the file per message.

        The first non-empty name becomes the display name. A group card counts
        as authoritative: the group remark is the real name people use, so it
        replaces a nickname picked up earlier in a private chat.
        """

        person_id = get_person_id(platform, str(user_id))
        record = self._people.get(person_id)
        if record is None:
            record = {"platform": platform, "user_id": str(user_id), "aliases": []}
            self._people[person_id] = record
            self._dirty = True
        clean_name = name.strip()
        # A one-character card is usually punctuation or a joke, not a name.
        if authoritative and len(clean_name) < 2:
            return person_id
        if not clean_name:
            return person_id
        current = str(record.get("display_name") or "").strip()
        if not current or (authoritative and current != clean_name):
            record["display_name"] = clean_name
            self._dirty = True
        return person_id

    def set_display_name(
        self,
        platform: str,
        user_id: str,
        name: str,
        *,
        aliases: Optional[Iterable[str]] = None,
    ) -> str:
        person_id = get_person_id(platform, user_id)
        record = self._people.setdefault(
            person_id,
            {"platform": platform, "user_id": str(user_id), "aliases": []},
        )
        record["display_name"] = name.strip()
        record["platform"] = platform
        record["user_id"] = str(user_id)
        if aliases is not None:
            record["aliases"] = [alias.strip() for alias in aliases if alias.strip()]
        return person_id

    def display_name(self, platform: str, user_id: str) -> str:
        record = self._people.get(get_person_id(platform, user_id))
        if not record:
            return ""
        return str(record.get("display_name") or "")

    def display_name_by_person_id(self, person_id: str) -> str:
        record = self._people.get(str(person_id))
        if not record:
            return ""
        return str(record.get("display_name") or "")

    def resolve_person_id(self, name_or_id: str) -> str:
        query = name_or_id.strip()
        if not query:
            return ""
        for person_id, record in self._people.items():
            if query == person_id:
                return person_id
            if query == str(record.get("display_name") or ""):
                return person_id
            aliases = record.get("aliases") or []
            if isinstance(aliases, list) and query in {str(alias) for alias in aliases}:
                return person_id
        # Message headers carry only a short handle; accept one unambiguous prefix.
        if len(query) >= MIN_ALIAS_PREFIX:
            prefixed = [person_id for person_id in self._people if person_id.startswith(query)]
            if len(prefixed) == 1:
                return prefixed[0]
        return ""

    def people(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._people.items()}
