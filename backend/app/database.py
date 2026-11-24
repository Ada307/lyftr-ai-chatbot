"""SQLite database initialization, connection management, and transactional operations."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Union

from app.config import get_settings


def _resolve_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve database path from argument or application settings."""
    if db_path is not None:
        if isinstance(db_path, str):
            if db_path == ":memory:":
                return Path(":memory:")
            return Path(db_path).resolve()
        return db_path.resolve() if str(db_path) != ":memory:" else Path(":memory:")
    return get_settings().db_path


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Create a configured SQLite connection with foreign keys and timeout enabled."""
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    if str(db_path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")

    return conn


@contextmanager
def get_db(
    db_path: Optional[Union[str, Path]] = None
) -> Generator[sqlite3.Connection, None, None]:
    """Yield a short-lived SQLite connection and ensure it is properly closed."""
    resolved = _resolve_path(db_path)
    conn = get_connection(resolved)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initialize SQLite schema: sessions table, messages table, and index."""
    with get_db(db_path) as conn:
        with conn:
            # Create sessions table before messages table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    system_prompt TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

            # Create messages table referencing sessions
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                );
                """
            )

            # Index for fast ordered message retrieval per session
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages (session_id, id);
                """
            )


def check_db_connection(db_path: Optional[Union[str, Path]] = None) -> bool:
    """Check database connectivity by executing a lightweight query."""
    try:
        with get_db(db_path) as conn:
            row = conn.execute("SELECT 1;").fetchone()
            return row is not None and row[0] == 1
    except Exception:
        return False


def session_exists(session_id: str, db_path: Optional[Union[str, Path]] = None) -> bool:
    """Return True if a session exists with the given session_id, False otherwise."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ? LIMIT 1;",
            (session_id,),
        )
        return cursor.fetchone() is not None


def get_session_system_prompt(
    session_id: str, db_path: Optional[Union[str, Path]] = None
) -> Optional[str]:
    """Retrieve the stored system prompt for a session, or None if not set / nonexistent."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT system_prompt FROM sessions WHERE session_id = ?;",
            (session_id,),
        )
        row = cursor.fetchone()
        if row is not None:
            return row["system_prompt"]
        return None


def get_session_messages(
    session_id: str, db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Retrieve all messages for a session in oldest-first order without system prompt.

    Assistant message metadata is returned as a parsed dictionary.
    User messages omit metadata.
    """
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT role, content, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC;
            """,
            (session_id,),
        )
        rows = cursor.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        role = row["role"]
        content = row["content"]
        raw_meta = row["metadata"]

        item: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and raw_meta is not None:
            try:
                item["metadata"] = json.loads(raw_meta)
            except Exception:
                item["metadata"] = {}
        results.append(item)

    return results


def save_chat_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
    assistant_metadata: Union[Dict[str, Any], Any],
    system_prompt: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Atomically persist a user message and assistant reply with metadata in one transaction.

    - Creates the session if it does not exist.
    - Persists or updates the system prompt when an explicit override is supplied.
    - Saves user message with NULL metadata.
    - Saves assistant message with JSON-serialized metadata.
    - Records timezone-aware UTC ISO-8601 timestamps.
    - Automatically rolls back the entire transaction if any operation fails.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Normalize assistant metadata to a JSON string
    if hasattr(assistant_metadata, "model_dump"):
        meta_dict = assistant_metadata.model_dump()
    elif isinstance(assistant_metadata, dict):
        meta_dict = assistant_metadata
    else:
        meta_dict = dict(assistant_metadata)
    metadata_json = json.dumps(meta_dict)

    with get_db(db_path) as conn:
        with conn:  # Transaction context manager: commits on exit, rolls back on error
            # 1. Manage session record
            cursor = conn.execute(
                "SELECT session_id, system_prompt FROM sessions WHERE session_id = ?;",
                (session_id,),
            )
            session_row = cursor.fetchone()

            if session_row is None:
                conn.execute(
                    """
                    INSERT INTO sessions (session_id, system_prompt, created_at)
                    VALUES (?, ?, ?);
                    """,
                    (session_id, system_prompt, now_iso),
                )
            elif system_prompt is not None:
                conn.execute(
                    """
                    UPDATE sessions
                    SET system_prompt = ?
                    WHERE session_id = ?;
                    """,
                    (system_prompt, session_id),
                )

            # 2. Insert user message with NULL metadata
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?);
                """,
                (session_id, user_message, now_iso),
            )

            # 3. Insert assistant message with JSON-serialized metadata
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?);
                """,
                (session_id, assistant_message, metadata_json, now_iso),
            )
