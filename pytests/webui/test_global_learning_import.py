from contextlib import contextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select
import pytest

from src.common.database.database_model import Jargon, JargonCreatedBy
from src.webui.dependencies import require_auth
from src.webui.routers import jargon


@pytest.fixture
def storage(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine, tables=[Jargon.__table__])

    @contextmanager
    def db_session(*_args, **_kwargs):
        with Session(engine) as session:
            yield session
            session.commit()

    monkeypatch.setattr(jargon, "get_db_session", db_session)
    monkeypatch.setattr(jargon._chat_manager, "get_existing_session_by_session_id",
                        lambda value: SimpleNamespace(session_id=value) if value == "real-chat" else None)
    monkeypatch.setattr(jargon, "get_display_name_for_session_id", lambda *_args: "测试群")
    app = FastAPI()
    app.include_router(jargon.router)
    app.dependency_overrides[require_auth] = lambda: True
    with TestClient(app) as client:
        yield engine, client
    engine.dispose()


def jargon_payload(**scope):
    return {"jargons": [{"content": "gkd", "meaning": "搞快点", "is_jargon": True}], **scope}


def test_global_jargon_is_not_blocked_by_local_term_and_does_not_overwrite_it(storage):
    engine, client = storage
    with Session(engine) as session:
        session.add(Jargon(content="gkd", meaning="局部含义", session_id_dict='{"real-chat":1}',
                           is_global=False, created_by=JargonCreatedBy.MANUAL))
        session.commit()
    payload = jargon_payload(is_global=True)
    assert client.post("/jargon/import", json=payload).json()["imported_count"] == 1
    assert client.post("/jargon/import", json=payload).json()["skipped_count"] == 1
    with Session(engine) as session:
        rows = session.exec(select(Jargon).order_by(Jargon.id)).all()
        assert len(rows) == 2
        assert rows[0].meaning == "局部含义"
        assert rows[1].is_global
        assert rows[1].session_id_dict == "{}"


def test_import_rejects_missing_or_mixed_targets_and_unknown_sessions(storage):
    _, client = storage
    payload = jargon_payload()
    assert client.post("/jargon/import", json=payload).status_code == 400
    for chat in ("real-chat", "unknown-chat"):
        assert client.post(
            "/jargon/import", json={**payload, "target_session_ids": [chat], "is_global": True}
        ).status_code == 400
    assert client.post(
        "/jargon/import", json={**payload, "target_session_ids": ["unknown-chat"]}
    ).status_code == 400


def test_local_import_still_uses_existing_chat_scope(storage):
    engine, client = storage
    assert client.post(
        "/jargon/import", json=jargon_payload(target_session_ids=["real-chat"])
    ).json()["imported_count"] == 1
    with Session(engine) as session:
        row = session.exec(select(Jargon)).one()
        assert not row.is_global
        assert "real-chat" in row.session_id_dict


def test_manual_global_jargon_needs_no_chat_and_preserves_same_local_term(storage):
    engine, client = storage
    with Session(engine) as session:
        session.add(Jargon(content="gkd", meaning="局部含义", session_id_dict='{"real-chat":1}',
                           is_global=False, created_by=JargonCreatedBy.MANUAL))
        session.commit()
    payload = {"content": "gkd", "meaning": "搞快点", "is_global": True, "session_ids": []}
    result = client.post("/jargon/", json=payload)
    assert result.status_code == 200, result.text
    assert result.json()["data"]["session_ids"] == []
    assert client.post("/jargon/", json=payload).status_code == 400
    with Session(engine) as session:
        rows = session.exec(select(Jargon).order_by(Jargon.id)).all()
        assert len(rows) == 2
        assert rows[0].meaning == "局部含义"
        assert rows[1].is_global
        assert rows[1].session_id_dict == "{}"


def test_manual_local_create_still_validates_existing_chat(storage):
    engine, client = storage
    payload = {"content": "gkd", "meaning": "搞快点"}
    assert client.post("/jargon/", json=payload).status_code == 400
    assert client.post("/jargon/", json={**payload, "session_ids": ["unknown-chat"]}).status_code == 400
    result = client.post("/jargon/", json={**payload, "session_ids": ["real-chat"]})
    assert result.status_code == 200, result.text
    with Session(engine) as session:
        row = session.exec(select(Jargon)).one()
        assert not row.is_global
        assert "real-chat" in row.session_id_dict
