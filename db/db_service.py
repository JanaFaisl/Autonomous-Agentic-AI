"""
Database helper for prototype state persistence.
Convex-Chef-inspired schema: sessions + events + artifacts.
"""
import logging
import os
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_sqlite_init_lock = threading.Lock()

# SQLite type mapping used by create_table_from_schema (defined once, not rebuilt per call)
_SQLITE_TYPE_MAP = {
    "UUID": "TEXT", "VARCHAR": "TEXT", "CHAR": "TEXT", "STRING": "TEXT", "TEXT": "TEXT",
    "INT": "INTEGER", "INTEGER": "INTEGER", "BIGINT": "INTEGER", "SMALLINT": "INTEGER",
    "BOOLEAN": "INTEGER", "BOOL": "INTEGER",
    "TIMESTAMP": "TEXT", "DATE": "TEXT", "DATETIME": "TEXT", "TIME": "TEXT",
    "JSON": "TEXT", "JSONB": "TEXT",
    "FLOAT": "REAL", "DOUBLE": "REAL", "DECIMAL": "REAL", "NUMERIC": "REAL", "REAL": "REAL",
}


def _quote_identifier(name: str) -> str:
    """Return a safely double-quoted SQL identifier with embedded quotes stripped."""
    return '"' + name.replace('"', '') + '"'

# Check if Streamlit is available (for error logging)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a dummy st object for compatibility
    class _DummySt:
        session_state = type('obj', (object,), {})()
    st = _DummySt()

try:
    from sqlalchemy import (
        create_engine, Column, String, DateTime, Text, Integer,
        ForeignKey, Index
    )
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.orm import declarative_base, sessionmaker, relationship
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    # Fallback to simple SQLite if SQLAlchemy not available
    import sqlite3
    from typing import Dict, List, Optional

Base = None
_ENGINE = None
_SessionLocal = None

if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

def _now() -> datetime:
    return datetime.now(timezone.utc)

def get_database_url() -> str:
    """Get database URL from env or use SQLite fallback."""
    return os.getenv("DATABASE_URL", "sqlite:///./app.db")

def init_db() -> None:
    """Initialize database connection and create tables."""
    global _ENGINE, _SessionLocal
    if not SQLALCHEMY_AVAILABLE:
        # Fallback: use simple SQLite
        _init_sqlite_fallback()
        return
    
    if _ENGINE is not None:
        return
    
    url = get_database_url()
    # pool_pre_ping avoids stale connections in long-running Streamlit sessions
    _ENGINE = create_engine(url, pool_pre_ping=True, future=True)
    _SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(_ENGINE)

def db_session():
    """Get a database session."""
    if not SQLALCHEMY_AVAILABLE:
        return None
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()

if SQLALCHEMY_AVAILABLE:
    class Session(Base):
        __tablename__ = "sessions"
        id = Column(String(64), primary_key=True)  # your session_id
        created_at = Column(DateTime, default=_now, nullable=False)
        updated_at = Column(DateTime, default=_now, onupdate=_now, nullable=False)

        events = relationship("Event", back_populates="session", cascade="all, delete-orphan")
        artifacts = relationship("Artifact", back_populates="session", cascade="all, delete-orphan")

    class Event(Base):
        __tablename__ = "events"
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

        # INPUT / CLICK / NAV
        type = Column(String(32), nullable=False)  # "input" | "click" | "navigation"
        screen = Column(String(128), nullable=True)
        field = Column(String(128), nullable=True)
        value = Column(Text, nullable=True)
        button = Column(String(256), nullable=True)
        nav_from = Column(String(128), nullable=True)
        nav_to = Column(String(128), nullable=True)

        created_at = Column(DateTime, default=_now, nullable=False)

        session = relationship("Session", back_populates="events")

    Index("ix_events_session_time", Event.session_id, Event.created_at.desc())
    Index("ix_events_session_type", Event.session_id, Event.type)

    class Artifact(Base):
        __tablename__ = "artifacts"
        id = Column(Integer, primary_key=True, autoincrement=True)
        session_id = Column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)

        # "requirements" | "design" | "database_schema"
        kind = Column(String(32), nullable=False)
        version = Column(Integer, nullable=False, default=1)

        # Use JSONB when on Postgres; SQLAlchemy will handle it. SQLite will store as TEXT via fallback.
        payload_json = Column(JSONB().with_variant(Text, "sqlite"), nullable=False)

        created_at = Column(DateTime, default=_now, nullable=False)

        session = relationship("Session", back_populates="artifacts")

    Index("ix_artifacts_session_kind_version", Artifact.session_id, Artifact.kind, Artifact.version.desc())

# Fallback SQLite implementation (simple, no SQLAlchemy)
_sqlite_conn = None

def _init_sqlite_fallback():
    """Initialize simple SQLite fallback if SQLAlchemy not available."""
    global _sqlite_conn
    if _sqlite_conn is not None:
        return
    with _sqlite_init_lock:
        if _sqlite_conn is not None:  # re-check after acquiring lock
            return
    
    _sqlite_conn = sqlite3.connect("app.db", check_same_thread=False)
    _sqlite_conn.row_factory = sqlite3.Row
    cursor = _sqlite_conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
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
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            payload_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_events_session_time 
        ON events(session_id, created_at DESC)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_events_session_type 
        ON events(session_id, type)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS ix_artifacts_session_kind_version 
        ON artifacts(session_id, kind, version DESC)
    """)
    
    _sqlite_conn.commit()

def _ensure_session(sid: str) -> None:
    """Ensure a session exists in the database."""
    if not SQLALCHEMY_AVAILABLE:
        # SQLite fallback
        if _sqlite_conn is None:
            _init_sqlite_fallback()
        cursor = _sqlite_conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO sessions (id) VALUES (?)", (sid,))
        _sqlite_conn.commit()
        return
    
    with db_session() as s:
        obj = s.get(Session, sid)
        if obj is None:
            s.add(Session(id=sid))
        s.commit()

def save_input(session_id: str, screen: str, field: str, value: str) -> None:
    """Save an input event."""
    try:
        _ensure_session(session_id)
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO events (session_id, type, screen, field, value)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, "input", screen, field, value))
            _sqlite_conn.commit()
            return
        
        with db_session() as s:
            s.add(Event(
                session_id=session_id,
                type="input",
                screen=screen,
                field=field,
                value=value
            ))
            s.commit()
    except Exception as e:
        logger.error("save_input error: %s", e, exc_info=True)
        return

def save_click(session_id: str, screen: str, button: str) -> None:
    """Save a click event."""
    try:
        _ensure_session(session_id)
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO events (session_id, type, screen, button)
                VALUES (?, ?, ?, ?)
            """, (session_id, "click", screen, button))
            _sqlite_conn.commit()
            return

        with db_session() as s:
            s.add(Event(
                session_id=session_id,
                type="click",
                screen=screen,
                button=button
            ))
            s.commit()
    except Exception as e:
        logger.error("save_click error: %s", e, exc_info=True)
        return

def save_navigation(session_id: str, from_screen: str, to_screen: str) -> None:
    """Save a navigation event."""
    try:
        _ensure_session(session_id)
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                INSERT INTO events (session_id, type, nav_from, nav_to)
                VALUES (?, ?, ?, ?)
            """, (session_id, "navigation", from_screen, to_screen))
            _sqlite_conn.commit()
            return

        with db_session() as s:
            s.add(Event(
                session_id=session_id,
                type="navigation",
                nav_from=from_screen,
                nav_to=to_screen
            ))
            s.commit()
    except Exception as e:
        logger.error("save_navigation error: %s", e, exc_info=True)
        return

def save_artifact(session_id: str, kind: str, payload: Dict[str, Any]) -> int:
    """Save a versioned JSON artifact. Returns the new version number, or 0 on error."""
    try:
        _ensure_session(session_id)
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                SELECT MAX(version) as max_version FROM artifacts
                WHERE session_id = ? AND kind = ?
            """, (session_id, kind))
            row = cursor.fetchone()
            next_version = (row[0] + 1) if row and row[0] else 1
            cursor.execute("""
                INSERT INTO artifacts (session_id, kind, version, payload_json)
                VALUES (?, ?, ?, ?)
            """, (session_id, kind, next_version, json.dumps(payload)))
            _sqlite_conn.commit()
            return next_version

        with db_session() as s:
            latest = (
                s.query(Artifact)
                .filter(Artifact.session_id == session_id, Artifact.kind == kind)
                .order_by(Artifact.version.desc())
                .first()
            )
            next_version = (latest.version + 1) if latest else 1
            s.add(Artifact(
                session_id=session_id,
                kind=kind,
                version=next_version,
                payload_json=payload
            ))
            s.commit()
            return next_version
    except Exception as e:
        logger.error("save_artifact error (session=%s, kind=%s): %s", session_id, kind, e, exc_info=True)
        return 0

def get_latest_artifact(session_id: str, kind: str) -> Optional[Dict[str, Any]]:
    """Get the latest version of an artifact. Returns None if not found or on error."""
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                SELECT payload_json FROM artifacts
                WHERE session_id = ? AND kind = ?
                ORDER BY version DESC
                LIMIT 1
            """, (session_id, kind))
            row = cursor.fetchone()
            if not row:
                return None
            return json.loads(row[0])

        with db_session() as s:
            latest = (
                s.query(Artifact)
                .filter(Artifact.session_id == session_id, Artifact.kind == kind)
                .order_by(Artifact.version.desc())
                .first()
            )
            if not latest:
                return None
            if isinstance(latest.payload_json, str):
                try:
                    return json.loads(latest.payload_json)
                except Exception:
                    logger.error("get_latest_artifact: failed to decode JSON for session=%s kind=%s", session_id, kind)
                    return None
            return latest.payload_json
    except Exception as e:
        logger.error("get_latest_artifact error (session=%s, kind=%s): %s", session_id, kind, e, exc_info=True)
        return None

def get_all_session_state(session_id: str) -> Dict[str, Any]:
    """Get all events for a session."""
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                SELECT * FROM events
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (session_id,))
            rows = cursor.fetchall()
            return {
                "session_id": session_id,
                "events": [
                    {
                        "type": row["type"],
                        "screen": row["screen"],
                        "field": row["field"],
                        "value": row["value"],
                        "button": row["button"],
                        "from": row["nav_from"],
                        "to": row["nav_to"],
                        "created_at": row["created_at"]
                    }
                    for row in rows
                ]
            }
        
        with db_session() as s:
            events = (
                s.query(Event)
                .filter(Event.session_id == session_id)
                .order_by(Event.created_at.asc())
                .all()
            )
            return {
                "session_id": session_id,
                "events": [
                    {
                        "type": e.type,
                        "screen": e.screen,
                        "field": e.field,
                        "value": e.value,
                        "button": e.button,
                        "from": e.nav_from,
                        "to": e.nav_to,
                        "created_at": e.created_at.isoformat()
                    }
                    for e in events
                ]
            }
    except Exception as e:
        logger.error("get_all_session_state error (session=%s): %s", session_id, e, exc_info=True)
        return {"session_id": session_id, "events": []}

def get_screen_state(session_id: str) -> Dict[str, Dict[str, str]]:
    """
    Reconstruct latest inputs per screen/field from event log.
    """
    state: Dict[str, Dict[str, str]] = {}
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("""
                SELECT screen, field, value FROM events
                WHERE session_id = ? AND type = 'input'
                ORDER BY created_at ASC
            """, (session_id,))
            rows = cursor.fetchall()
            for row in rows:
                if row["screen"] and row["field"]:
                    state.setdefault(row["screen"], {})
                    state[row["screen"]][row["field"]] = row["value"] or ""
            return state
        
        with db_session() as s:
            rows = (
                s.query(Event)
                .filter(Event.session_id == session_id, Event.type == "input")
                .order_by(Event.created_at.asc())
                .all()
            )
            for r in rows:
                if not r.screen or not r.field:
                    continue
                state.setdefault(r.screen, {})
                state[r.screen][r.field] = r.value or ""
        return state
    except Exception as e:
        logger.error("get_screen_state error (session=%s): %s", session_id, e, exc_info=True)
        return {}

def clear_session(session_id: str) -> None:
    """Clear all data for a session."""
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cursor = _sqlite_conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            _sqlite_conn.commit()
            return
        
        with db_session() as s:
            obj = s.get(Session, session_id)
            if obj:
                s.delete(obj)
                s.commit()
    except Exception as e:
        logger.error("clear_session error (session=%s): %s", session_id, e, exc_info=True)


# ============================================================================
# Domain Data Layer: Dynamic table creation and CRUD operations
# ============================================================================

def _get_sqlite_conn():
    """Get SQLite connection (works with both SQLAlchemy and fallback)."""
    if not SQLALCHEMY_AVAILABLE:
        # Fallback: use direct SQLite
        if _sqlite_conn is None:
            _init_sqlite_fallback()
        return _sqlite_conn
    else:
        # SQLAlchemy: get connection from engine
        if _ENGINE is None:
            init_db()
        return _ENGINE.raw_connection()


def create_table_from_schema(table_name: str, columns: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Create a domain table from schema definition.
    
    Args:
        table_name: Name of the table to create
        columns: List of column definitions with keys: name, type, pk, nullable, unique, default
    
    Returns:
        Tuple of (success: bool, error_message: str)
    """
    try:
        # Check if table already exists
        if table_exists(table_name):
            return False, "Table already exists"
        
        if not columns or len(columns) == 0:
            return False, "No columns defined in schema"
        
        # First pass: collect primary key columns
        pk_columns = []  # List of (col_name_escaped, sqlite_type, original_col_name) tuples
        
        for col in columns:
            if not isinstance(col, dict):
                continue
            col_name = col.get("name", "").strip()
            if not col_name:
                continue
            
            col_type = str(col.get("type", "TEXT")).strip().upper()
            base_type = col_type.split("(")[0].strip()
            sqlite_type = _SQLITE_TYPE_MAP.get(base_type, "TEXT")

            col_name_escaped = _quote_identifier(col_name)

            is_pk = col.get("pk") or col.get("primary_key") or False
            if is_pk:
                pk_columns.append((col_name_escaped, sqlite_type, col_name))

        # Build CREATE TABLE statement
        col_defs = []
        for col in columns:
            if not isinstance(col, dict):
                continue

            col_name = col.get("name", "").strip()
            if not col_name:
                continue

            col_type = str(col.get("type", "TEXT")).strip().upper()
            base_type = col_type.split("(")[0].strip()
            sqlite_type = _SQLITE_TYPE_MAP.get(base_type, "TEXT")
            
            col_name_escaped = _quote_identifier(col_name)

            col_def = f"{col_name_escaped} {sqlite_type}"
            
            # Check if this column should be a primary key
            is_pk = col.get("pk") or col.get("primary_key") or False
            
            # Add PRIMARY KEY inline ONLY if it's the single PK (not composite)
            if is_pk and len(pk_columns) == 1:
                col_def += " PRIMARY KEY"
                # If it's INTEGER PRIMARY KEY, make it AUTOINCREMENT
                if sqlite_type == "INTEGER":
                    col_def += " AUTOINCREMENT"
            
            # Add NOT NULL for primary keys or explicitly non-nullable columns
            nullable = col.get("nullable", True)
            if not nullable or is_pk:
                col_def += " NOT NULL"
            
            # Add UNIQUE (but not if it's already a PK)
            is_unique = col.get("unique") or False
            if is_unique and not is_pk:
                col_def += " UNIQUE"
            
            # Add DEFAULT
            default_val = col.get("default")
            if default_val is not None and default_val != "":
                if isinstance(default_val, bool):  # must come before int — bool is a subclass of int
                    col_def += f" DEFAULT {1 if default_val else 0}"
                elif isinstance(default_val, (int, float)):
                    col_def += f" DEFAULT {default_val}"
                elif isinstance(default_val, str):
                    default_val_escaped = default_val.replace("'", "''")
                    col_def += f" DEFAULT '{default_val_escaped}'"
                else:
                    default_val_escaped = str(default_val).replace("'", "''")
                    col_def += f" DEFAULT '{default_val_escaped}'"
            
            col_defs.append(col_def)
        
        if not col_defs:
            return False, "No valid column definitions found"
        
        # Handle composite primary keys (multiple PK columns)
        if len(pk_columns) > 1:
            # Composite primary key: add as separate constraint
            pk_col_names = [name for name, _, _ in pk_columns]
            pk_constraint = f', PRIMARY KEY ({", ".join(pk_col_names)})'
        else:
            pk_constraint = ""
        
        # Create table SQL
        create_sql = f'CREATE TABLE IF NOT EXISTS {_quote_identifier(table_name)} ({", ".join(col_defs)}{pk_constraint})'
        
        # Execute using appropriate connection method
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            try:
                cur.execute(create_sql)
                _sqlite_conn.commit()
                return True, ""
            except Exception as e:
                return False, f"SQL Error: {str(e)}"
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute(create_sql)
                conn.commit()
                cur.close()
                conn.close()
                return True, ""
            except Exception as e:
                conn.close()
                return False, f"SQL Error: {str(e)}"
        
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def table_exists(table_name: str) -> bool:
    """Check if a domain table exists in the database."""
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            cur.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            exists = cur.fetchone() is not None
            return exists
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table_name,))
                exists = cur.fetchone() is not None
                cur.close()
                conn.close()
                return exists
            except Exception:
                conn.close()
                return False
    except Exception:
        return False


def list_domain_tables() -> List[str]:
    """List all domain tables (excludes system tables: sessions, events, artifacts,
    project_tables)."""
    try:
        system_tables = {"sessions", "events", "artifacts", "project_tables"}
        
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            cur.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table'
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cur.fetchall() if row[0] not in system_tables]
            return tables
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table'
                    AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
                tables = [row[0] for row in cur.fetchall() if row[0] not in system_tables]
                cur.close()
                conn.close()
                return tables
            except Exception:
                conn.close()
                return []
    except Exception:
        return []


def fetch_objects(table_name: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch rows from a domain table as list of dictionaries."""
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            cur.execute(f'SELECT * FROM {_quote_identifier(table_name)} LIMIT ?', (limit,))
            rows = cur.fetchall()
            # Get column names
            col_names = [description[0] for description in cur.description]
            return [dict(zip(col_names, row)) for row in rows]
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute(f'SELECT * FROM {_quote_identifier(table_name)} LIMIT ?', (limit,))
                rows = cur.fetchall()
                # Get column names
                col_names = [description[0] for description in cur.description]
                result = [dict(zip(col_names, row)) for row in rows]
                cur.close()
                conn.close()
                return result
            except Exception as e:
                conn.close()
                return []
    except Exception:
        return []


def insert_object(table_name: str, data: Dict[str, Any]) -> None:
    """Insert a new row into a domain table."""
    try:
        cols = list(data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        cols_sql = ", ".join([_quote_identifier(col) for col in cols])
        values = [data[c] for c in cols]

        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            cur.execute(f'INSERT INTO {_quote_identifier(table_name)} ({cols_sql}) VALUES ({placeholders})', values)
            _sqlite_conn.commit()
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute(f'INSERT INTO {_quote_identifier(table_name)} ({cols_sql}) VALUES ({placeholders})', values)
                conn.commit()
                cur.close()
                conn.close()
            except Exception:
                conn.close()
                raise
    except Exception as e:
        if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            if "db_errors" not in st.session_state:
                st.session_state.db_errors = []
            st.session_state.db_errors.append(f"insert_object error: {str(e)}")
        raise


def get_table_columns(table_name: str) -> List[str]:
    """Get column names for a domain table."""
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            cur.execute(f'PRAGMA table_info({_quote_identifier(table_name)})')
            columns = [row[1] for row in cur.fetchall()]
            return columns
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute(f'PRAGMA table_info({_quote_identifier(table_name)})')
                columns = [row[1] for row in cur.fetchall()]
                cur.close()
                conn.close()
                return columns
            except Exception:
                conn.close()
                return []
    except Exception:
        return []


def count_rows(table_name: str) -> int:
    """Return the number of rows in a domain table, or -1 if it can't be counted."""
    sql = f'SELECT COUNT(*) FROM {_quote_identifier(table_name)}'
    try:
        if not SQLALCHEMY_AVAILABLE:
            # SQLite fallback
            if _sqlite_conn is None:
                _init_sqlite_fallback()
            cur = _sqlite_conn.cursor()
            cur.execute(sql)
            return int(cur.fetchone()[0])
        else:
            # SQLAlchemy: use raw connection
            if _ENGINE is None:
                init_db()
            conn = _ENGINE.raw_connection()
            try:
                cur = conn.cursor()
                cur.execute(sql)
                total = int(cur.fetchone()[0])
                cur.close()
                conn.close()
                return total
            except Exception:
                conn.close()
                return -1
    except Exception:
        return -1


# ----------------------------------------------------------------------
# Project table registry
#
# The database accumulates domain tables across every session, so listing the
# raw table names shows tables from unrelated past projects. This registry
# records which tables were created for which session, letting the UI show the
# current project's tables instead of everything ever created.
# ----------------------------------------------------------------------

_PROJECT_TABLES = "project_tables"


def _run(sql: str, params: tuple = (), fetch: bool = False):
    """Execute a single statement on whichever connection backend is available."""
    if not SQLALCHEMY_AVAILABLE:
        if _sqlite_conn is None:
            _init_sqlite_fallback()
        cur = _sqlite_conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        _sqlite_conn.commit()
        return rows

    if _ENGINE is None:
        init_db()
    conn = _ENGINE.raw_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        cur.close()
        return rows
    finally:
        conn.close()


def _ensure_project_registry() -> None:
    _run(
        f'CREATE TABLE IF NOT EXISTS {_quote_identifier(_PROJECT_TABLES)} ('
        "session_id TEXT NOT NULL, table_name TEXT NOT NULL, source TEXT, "
        "created_at TEXT, PRIMARY KEY (session_id, table_name))"
    )


def register_project_table(session_id: str, table_name: str, source: str = "") -> None:
    """Record that `table_name` belongs to this session's project.

    The registry is advisory: a failure here must never prevent the table itself
    from being created, so errors are swallowed.
    """
    if not session_id or not table_name:
        return
    try:
        _ensure_project_registry()
        _run(
            f'INSERT OR REPLACE INTO {_quote_identifier(_PROJECT_TABLES)} '
            "(session_id, table_name, source, created_at) VALUES (?, ?, ?, ?)",
            (session_id, table_name, source, _now().isoformat()),
        )
    except Exception as e:
        logger.debug("register_project_table failed for %s: %s", table_name, e)


def list_project_tables(session_id: str) -> List[str]:
    """Table names registered to this session, most recently added first."""
    if not session_id:
        return []
    try:
        _ensure_project_registry()
        rows = _run(
            f'SELECT table_name FROM {_quote_identifier(_PROJECT_TABLES)} '
            "WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,),
            fetch=True,
        ) or []
        return [r[0] for r in rows]
    except Exception:
        return []
