"""Unit tests for SQLite database operations and schema management."""

from pathlib import Path
import sqlite3
import pytest

from app.database import (
    check_db_connection,
    get_connection,
    get_db,
    get_session_messages,
    get_session_system_prompt,
    init_db,
    save_chat_turn,
    session_exists,
)
from app.schemas import ChatMetadata


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Fixture providing an isolated temporary SQLite database path."""
    test_file = tmp_path / "test_app.db"
    init_db(test_file)
    return test_file


def test_schema_creation_and_idempotency(db_path: Path):
    """Test that schema initializes successfully and repeated calls are safe."""
    # Repeated calls should not raise any errors
    init_db(db_path)
    init_db(db_path)

    assert check_db_connection(db_path) is True

    with get_db(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "sessions" in table_names
        assert "messages" in table_names

        indices = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_messages_session_id_id';"
        ).fetchall()
        assert len(indices) == 1


def test_session_creation_and_message_saving(db_path: Path):
    """Test saving a chat turn creates the session and saves both user and assistant messages."""
    session_id = "sess-001"
    assert session_exists(session_id, db_path) is False

    meta = ChatMetadata(
        resolution_source="LIVE_INFERENCE",
        model="gpt-test",
        latency_ms=120,
        input_tokens=15,
        output_tokens=30,
        key_id="key_1",
        fallback_attempts=0,
    )

    save_chat_turn(
        session_id=session_id,
        user_message="Hello, AI!",
        assistant_message="Hello, Human!",
        assistant_metadata=meta,
        system_prompt="You are a helpful assistant.",
        db_path=db_path,
    )

    assert session_exists(session_id, db_path) is True
    assert get_session_system_prompt(session_id, db_path) == "You are a helpful assistant."

    messages = get_session_messages(session_id, db_path)
    assert len(messages) == 2

    # User message check
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, AI!"
    assert "metadata" not in messages[0]

    # Assistant message check
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello, Human!"
    assert messages[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert messages[1]["metadata"]["model"] == "gpt-test"
    assert messages[1]["metadata"]["latency_ms"] == 120
    assert messages[1]["metadata"]["input_tokens"] == 15
    assert messages[1]["metadata"]["output_tokens"] == 30
    assert messages[1]["metadata"]["key_id"] == "key_1"
    assert messages[1]["metadata"]["fallback_attempts"] == 0


def test_metadata_json_serialization_and_deserialization(db_path: Path):
    """Test metadata is stored as JSON in DB and deserialized to a dict in history."""
    session_id = "sess-meta"
    meta_dict = {
        "resolution_source": "LLM_CACHE",
        "model": "claude-cached",
        "latency_ms": 5,
        "input_tokens": 100,
        "output_tokens": 50,
        "key_id": None,
        "fallback_attempts": 1,
    }

    save_chat_turn(
        session_id=session_id,
        user_message="Cached query",
        assistant_message="Cached answer",
        assistant_metadata=meta_dict,
        db_path=db_path,
    )

    # Check raw SQL table representation
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT metadata FROM messages WHERE session_id = ? AND role = 'assistant';",
            (session_id,),
        ).fetchone()
        assert isinstance(row["metadata"], str)
        assert '"resolution_source": "LLM_CACHE"' in row["metadata"]

    # Check parsed representation from get_session_messages
    messages = get_session_messages(session_id, db_path)
    assert isinstance(messages[1]["metadata"], dict)
    assert messages[1]["metadata"]["resolution_source"] == "LLM_CACHE"
    assert messages[1]["metadata"]["key_id"] is None


def test_message_ordering_oldest_first(db_path: Path):
    """Test messages are returned chronologically oldest-first."""
    session_id = "sess-order"
    meta = {"resolution_source": "LIVE_INFERENCE", "model": "m", "latency_ms": 1, "input_tokens": 1, "output_tokens": 1, "key_id": "key_1", "fallback_attempts": 0}

    save_chat_turn(
        session_id=session_id,
        user_message="Message 1",
        assistant_message="Reply 1",
        assistant_metadata=meta,
        db_path=db_path,
    )
    save_chat_turn(
        session_id=session_id,
        user_message="Message 2",
        assistant_message="Reply 2",
        assistant_metadata=meta,
        db_path=db_path,
    )

    messages = get_session_messages(session_id, db_path)
    assert len(messages) == 4
    contents = [m["content"] for m in messages]
    assert contents == ["Message 1", "Reply 1", "Message 2", "Reply 2"]


def test_session_isolation(db_path: Path):
    """Test that messages from one session do not leak into another session."""
    meta = {"resolution_source": "LIVE_INFERENCE", "model": "m", "latency_ms": 1, "input_tokens": 1, "output_tokens": 1, "key_id": "key_1", "fallback_attempts": 0}

    save_chat_turn(
        session_id="sess-A",
        user_message="User A",
        assistant_message="Reply A",
        assistant_metadata=meta,
        db_path=db_path,
    )
    save_chat_turn(
        session_id="sess-B",
        user_message="User B",
        assistant_message="Reply B",
        assistant_metadata=meta,
        db_path=db_path,
    )

    msgs_a = get_session_messages("sess-A", db_path)
    msgs_b = get_session_messages("sess-B", db_path)

    assert len(msgs_a) == 2
    assert [m["content"] for m in msgs_a] == ["User A", "Reply A"]

    assert len(msgs_b) == 2
    assert [m["content"] for m in msgs_b] == ["User B", "Reply B"]


def test_system_prompt_persistence_and_update(db_path: Path):
    """Test system prompt persistence on creation, retention when omitted, and explicit update."""
    session_id = "sess-sys"
    meta = {"resolution_source": "LIVE_INFERENCE", "model": "m", "latency_ms": 1, "input_tokens": 1, "output_tokens": 1, "key_id": "key_1", "fallback_attempts": 0}

    # Turn 1: with system prompt
    save_chat_turn(
        session_id=session_id,
        user_message="U1",
        assistant_message="A1",
        assistant_metadata=meta,
        system_prompt="Initial prompt",
        db_path=db_path,
    )
    assert get_session_system_prompt(session_id, db_path) == "Initial prompt"

    # Turn 2: without system prompt override (should retain existing)
    save_chat_turn(
        session_id=session_id,
        user_message="U2",
        assistant_message="A2",
        assistant_metadata=meta,
        system_prompt=None,
        db_path=db_path,
    )
    assert get_session_system_prompt(session_id, db_path) == "Initial prompt"

    # Turn 3: with explicit override (should update)
    save_chat_turn(
        session_id=session_id,
        user_message="U3",
        assistant_message="A3",
        assistant_metadata=meta,
        system_prompt="Updated prompt",
        db_path=db_path,
    )
    assert get_session_system_prompt(session_id, db_path) == "Updated prompt"


def test_transaction_rollback_on_failed_insert(db_path: Path):
    """Test that if an insert fails mid-transaction, all changes roll back atomically."""
    session_id = "sess-rollback"

    # Cause an error during the assistant insert by passing invalid non-serializable object
    class Unserializable:
        pass

    with pytest.raises(Exception):
        save_chat_turn(
            session_id=session_id,
            user_message="This should not persist",
            assistant_message="Nor should this",
            assistant_metadata=Unserializable(),
            db_path=db_path,
        )

    # Verify session was NOT created
    assert session_exists(session_id, db_path) is False

    # Verify no messages were persisted
    assert len(get_session_messages(session_id, db_path)) == 0


def test_database_persistence_across_reopen(db_path: Path):
    """Test that data persists to disk after closing connections and reopening the database."""
    session_id = "sess-persist"
    meta = {"resolution_source": "LIVE_INFERENCE", "model": "m", "latency_ms": 10, "input_tokens": 5, "output_tokens": 5, "key_id": "key_1", "fallback_attempts": 0}

    save_chat_turn(
        session_id=session_id,
        user_message="Persistent message",
        assistant_message="Persistent reply",
        assistant_metadata=meta,
        system_prompt="Be helpful",
        db_path=db_path,
    )

    # Open completely new connection to verify disk persistence
    new_conn = get_connection(db_path)
    try:
        session_row = new_conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?;", (session_id,)
        ).fetchone()
        assert session_row is not None
        assert session_row["system_prompt"] == "Be helpful"

        msg_rows = new_conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC;", (session_id,)
        ).fetchall()
        assert len(msg_rows) == 2
        assert msg_rows[0]["content"] == "Persistent message"
        assert msg_rows[1]["content"] == "Persistent reply"
    finally:
        new_conn.close()


def test_foreign_key_enforcement(db_path: Path):
    """Test that foreign key constraint prevents inserting messages for nonexistent sessions."""
    with get_db(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES ('nonexistent-session', 'user', 'Orphan message', NULL, '2026-01-01T00:00:00Z');
                """
            )
