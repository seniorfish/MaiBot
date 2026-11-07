from contextlib import contextmanager
from typing import Any, Dict, Iterator, Set

import pytest
from fastapi import HTTPException
from sqlmodel import Session, SQLModel, create_engine

from src.common.database.database_model import Jargon
from src.webui.routers import jargon


@pytest.fixture
def export_database(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine, tables=[Jargon.__table__])
    with Session(engine) as session:
        session.add_all([
            Jargon(id=1, content="a", meaning="a", session_id_dict='{"chat-a": 1}'),
            Jargon(id=2, content="b", meaning="b", session_id_dict='{"chat-a": 1, "chat-b": 2}'),
            Jargon(id=3, content="global", meaning="global", is_global=True),
        ])
        session.commit()

    @contextmanager
    def get_test_session() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    monkeypatch.setattr(jargon, "get_db_session", get_test_session)
    yield
    engine.dispose()


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, {"a", "b", "global"}),
        ({"ids": [1, 3]}, {"a", "global"}),
        ({"session_ids": ["chat-b"]}, {"b"}),
        ({"session_ids": ["chat-a", "chat-b", "chat-a"]}, {"a", "b"}),
        ({"session_ids": ["chat"]}, set()),
        ({"session_ids": []}, set()),
        ({"ids": []}, set()),
    ],
)
def test_jargon_export_scopes(export_database: None, params: Dict[str, Any], expected: Set[str]) -> None:
    result = jargon.export_jargons(jargon.JargonExportRequest(**params))
    assert {item.content for item in result.jargons} == expected
    assert result.count == len(expected)


def test_exports_reject_missing_selection(export_database: None) -> None:
    with pytest.raises(HTTPException, match="999"):
        jargon.export_jargons(jargon.JargonExportRequest(ids=[1, 999]))
