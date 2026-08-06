"""
Unit tests for db/db_service.py

Uses an in-memory SQLite database so no files are created on disk.
"""
import pytest
from unittest.mock import patch, MagicMock
import json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def use_memory_db(monkeypatch):
    """Provide a fresh in-memory SQLite database for every test.

    We pre-create the connection and inject it so that _init_sqlite_fallback's
    hardcoded 'app.db' path is never reached.
    """
    import sqlite3 as _sqlite3
    import db.db_service as db_mod

    conn = _sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = _sqlite3.Row
    conn.executescript("""
        PRAGMA foreign_keys = ON;
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL,
            screen TEXT,
            field TEXT,
            value TEXT,
            button TEXT,
            nav_from TEXT,
            nav_to TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            payload_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS ix_events_session_time ON events(session_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS ix_events_session_type ON events(session_id, type);
        CREATE INDEX IF NOT EXISTS ix_artifacts_session_kind_version ON artifacts(session_id, kind, version DESC);
    """)
    conn.commit()

    monkeypatch.setattr(db_mod, "SQLALCHEMY_AVAILABLE", False, raising=False)
    monkeypatch.setattr(db_mod, "sqlite3", _sqlite3, raising=False)
    monkeypatch.setattr(db_mod, "_sqlite_conn", conn, raising=False)

    yield

    conn.close()
    db_mod._sqlite_conn = None


@pytest.fixture
def session_id():
    return "test-session-001"


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------

class TestInitDb:

    def test_init_db_runs_without_error(self):
        from db.db_service import init_db
        init_db()  # should not raise

    def test_init_db_is_idempotent(self):
        from db.db_service import init_db
        init_db()
        init_db()  # calling twice must not fail


# ---------------------------------------------------------------------------
# save_artifact / get_latest_artifact
# ---------------------------------------------------------------------------

class TestArtifactPersistence:

    def test_save_and_retrieve_requirements_artifact(self, session_id):
        from db.db_service import init_db, save_artifact, get_latest_artifact
        init_db()
        payload = {"project_name": "Flower App", "features": []}
        save_artifact(session_id, "requirements", payload)
        result = get_latest_artifact(session_id, "requirements")
        assert result is not None
        assert result.get("project_name") == "Flower App"

    def test_save_and_retrieve_design_artifact(self, session_id):
        from db.db_service import init_db, save_artifact, get_latest_artifact
        init_db()
        payload = {"design_overview": "Modern UI"}
        save_artifact(session_id, "design", payload)
        result = get_latest_artifact(session_id, "design")
        assert result is not None
        assert result.get("design_overview") == "Modern UI"

    def test_get_latest_returns_most_recent(self, session_id):
        from db.db_service import init_db, save_artifact, get_latest_artifact
        init_db()
        save_artifact(session_id, "requirements", {"version": 1})
        save_artifact(session_id, "requirements", {"version": 2})
        result = get_latest_artifact(session_id, "requirements")
        assert result["version"] == 2

    def test_get_latest_returns_none_when_no_artifact(self, session_id):
        from db.db_service import init_db, get_latest_artifact
        init_db()
        result = get_latest_artifact(session_id, "nonexistent_kind")
        assert result is None

    def test_artifacts_isolated_by_session(self, session_id):
        from db.db_service import init_db, save_artifact, get_latest_artifact
        init_db()
        save_artifact("session-A", "requirements", {"owner": "A"})
        save_artifact("session-B", "requirements", {"owner": "B"})
        result_a = get_latest_artifact("session-A", "requirements")
        result_b = get_latest_artifact("session-B", "requirements")
        assert result_a["owner"] == "A"
        assert result_b["owner"] == "B"


# ---------------------------------------------------------------------------
# save_input / get_screen_state
# ---------------------------------------------------------------------------

class TestInputPersistence:

    def test_save_and_retrieve_input(self, session_id):
        from db.db_service import init_db, save_input, get_screen_state
        init_db()
        save_input(session_id, "discovery", "project_name", "Flower App")
        state = get_screen_state(session_id)
        assert "discovery" in state
        assert state["discovery"].get("project_name") == "Flower App"

    def test_multiple_fields_on_same_screen(self, session_id):
        from db.db_service import init_db, save_input, get_screen_state
        init_db()
        save_input(session_id, "discovery", "project_name", "Flower App")
        save_input(session_id, "discovery", "audience", "Individual customers")
        state = get_screen_state(session_id)
        assert state["discovery"]["project_name"] == "Flower App"
        assert state["discovery"]["audience"] == "Individual customers"


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------

class TestClearSession:

    def test_clear_session_removes_artifacts(self, session_id):
        from db.db_service import init_db, save_artifact, get_latest_artifact, clear_session
        init_db()
        save_artifact(session_id, "requirements", {"project_name": "Test"})
        clear_session(session_id)
        result = get_latest_artifact(session_id, "requirements")
        assert result is None

    def test_clear_session_only_affects_target_session(self):
        from db.db_service import init_db, save_artifact, get_latest_artifact, clear_session
        init_db()
        save_artifact("keep-session", "requirements", {"project_name": "Keep"})
        save_artifact("clear-session", "requirements", {"project_name": "Clear"})
        clear_session("clear-session")
        assert get_latest_artifact("keep-session", "requirements") is not None
        assert get_latest_artifact("clear-session", "requirements") is None


# ---------------------------------------------------------------------------
# create_table_from_schema
# ---------------------------------------------------------------------------

class TestCreateTableFromSchema:

    def test_create_valid_table(self):
        from db.db_service import init_db, create_table_from_schema, table_exists
        init_db()
        columns = [
            {"name": "id", "type": "integer", "pk": True, "nullable": False},
            {"name": "name", "type": "varchar", "pk": False, "nullable": True},
        ]
        success, message = create_table_from_schema("flowers", columns)
        assert success is True
        assert table_exists("flowers")

    def test_create_table_is_idempotent(self):
        from db.db_service import init_db, create_table_from_schema
        init_db()
        columns = [{"name": "id", "type": "integer", "pk": True, "nullable": False}]
        success1, _ = create_table_from_schema("idempotent_table", columns)
        success2, _ = create_table_from_schema("idempotent_table", columns)
        assert success1 is True
        # Second call should not raise (IF NOT EXISTS semantics)
        assert isinstance(success2, bool)

    def test_table_exists_returns_false_for_missing_table(self):
        from db.db_service import init_db, table_exists
        init_db()
        assert table_exists("this_table_does_not_exist_xyz") is False
