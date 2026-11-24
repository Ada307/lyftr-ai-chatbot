"""Tests for FastAPI endpoints, request validation, and response contracts."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.database import init_db, save_chat_turn
from app.main import app
from app.schemas import ChatMetadata, ChatRequest


@pytest.fixture
def client_with_temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture providing a TestClient with an isolated database and lifespan triggered."""
    test_db = tmp_path / "api_test.db"
    test_url = f"sqlite:///{test_db.as_posix()}"

    monkeypatch.setenv("DATABASE_URL", test_url)
    monkeypatch.setenv("LLM_API_KEYS", "secret_key_1, secret_key_2")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")

    # Override get_settings dependency to use test settings
    test_settings = Settings()
    app.dependency_overrides[get_settings] = lambda: test_settings

    # Using context manager ensures the lifespan (init_db) runs
    with TestClient(app) as test_client:
        yield test_client, test_db

    app.dependency_overrides.clear()


def test_health_endpoint_readiness(client_with_temp_db):
    """Test GET /health returns 200 and reports successful readiness."""
    client, _ = client_with_temp_db
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["keys_configured"] == 2
    assert data["cache_enabled"] is True


def test_unknown_session_returns_404(client_with_temp_db):
    """Test GET /sessions/{session_id}/messages returns 404 with exact detail."""
    client, _ = client_with_temp_db
    response = client.get("/sessions/unknown-session-123/messages")

    assert response.status_code == 404
    assert response.json() == {"detail": "session not found"}


def test_existing_session_history_contract(client_with_temp_db):
    """Test session history matches the exact specification contract."""
    client, test_db = client_with_temp_db
    session_id = "s-example"

    metadata = ChatMetadata(
        resolution_source="LIVE_INFERENCE",
        model="example-model",
        latency_ms=100,
        input_tokens=10,
        output_tokens=5,
        key_id="key_1",
        fallback_attempts=0,
    )

    save_chat_turn(
        session_id=session_id,
        user_message="Hello",
        assistant_message="Hi!",
        assistant_metadata=metadata,
        system_prompt="You are a friendly bot.",
        db_path=test_db,
    )

    response = client.get(f"/sessions/{session_id}/messages")
    assert response.status_code == 200

    data = response.json()
    assert data["session_id"] == "s-example"
    assert len(data["messages"]) == 2

    # 1. User message omits metadata
    user_msg = data["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "Hello"
    assert "metadata" not in user_msg

    # 2. Assistant message includes metadata
    asst_msg = data["messages"][1]
    assert asst_msg["role"] == "assistant"
    assert asst_msg["content"] == "Hi!"
    assert asst_msg["metadata"] == {
        "resolution_source": "LIVE_INFERENCE",
        "model": "example-model",
        "latency_ms": 100,
        "input_tokens": 10,
        "output_tokens": 5,
        "key_id": "key_1",
        "fallback_attempts": 0,
    }

    # 3. System prompt is absent from message history
    all_contents = [m["content"] for m in data["messages"]]
    assert "You are a friendly bot." not in all_contents


def test_cors_headers_support(client_with_temp_db):
    """Test CORS headers allow Content-Type and X-Simulate-Failure headers."""
    client, _ = client_with_temp_db
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "content-type,x-simulate-failure",
    }
    response = client.options("/health", headers=headers)
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ==============================================================================
# Pydantic ChatRequest Validation Tests
# ==============================================================================


def test_chat_request_valid():
    """Test valid ChatRequest creates successfully and preserves session_id."""
    req = ChatRequest(
        session_id="session-42",
        message="  Hello world!  ",
        system_prompt="Be brief",
    )
    assert req.session_id == "session-42"
    assert req.message == "Hello world!"  # Trimmed
    assert req.system_prompt == "Be brief"


def test_chat_request_whitespace_only_message_rejected():
    """Test that whitespace-only messages are rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="   \n\t  ")


def test_chat_request_empty_message_rejected():
    """Test that empty messages are rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message="")


def test_chat_request_message_length_validation_after_trimming():
    """Test that message length check applies after trimming."""
    # 4000 characters + surrounding whitespace should succeed
    msg_4000 = " " * 10 + "a" * 4000 + " " * 10
    req = ChatRequest(session_id="s1", message=msg_4000)
    assert len(req.message) == 4000

    # 4001 characters after trimming should fail
    msg_4001 = "a" * 4001
    with pytest.raises(ValidationError):
        ChatRequest(session_id="s1", message=msg_4001)


def test_chat_request_session_id_length_validation():
    """Test that session_id over 128 characters is rejected."""
    valid_id = "s" * 128
    req = ChatRequest(session_id=valid_id, message="Hi")
    assert req.session_id == valid_id

    invalid_id = "s" * 129
    with pytest.raises(ValidationError):
        ChatRequest(session_id=invalid_id, message="Hi")


def test_chat_request_whitespace_only_session_id_rejected():
    """Test that whitespace-only session_id is rejected."""
    with pytest.raises(ValidationError):
        ChatRequest(session_id="   ", message="Hi")


def test_chat_metadata_validation():
    """Test ChatMetadata validation of types, bounds, and nullability."""
    meta = ChatMetadata(
        resolution_source="LLM_CACHE",
        model="gpt-4o",
        latency_ms=0,
        input_tokens=0,
        output_tokens=0,
        key_id=None,
        fallback_attempts=0,
    )
    assert meta.key_id is None

    # Negative latency should fail
    with pytest.raises(ValidationError):
        ChatMetadata(
            resolution_source="LIVE_INFERENCE",
            model="m",
            latency_ms=-1,
            input_tokens=0,
            output_tokens=0,
            key_id="k",
            fallback_attempts=0,
        )

    # Invalid resolution source should fail
    with pytest.raises(ValidationError):
        ChatMetadata(
            resolution_source="UNKNOWN_SOURCE",  # type: ignore
            model="m",
            latency_ms=10,
            input_tokens=0,
            output_tokens=0,
            key_id=None,
            fallback_attempts=0,
        )


# ==============================================================================
# C. CHAT API ENDPOINT TESTS (POST /chat)
# ==============================================================================

from unittest.mock import AsyncMock, MagicMock, patch
from app.database import get_session_messages, get_session_system_prompt, session_exists


def _make_mock_litellm_resp(content="Test reply", model="test-model", p_tok=25, c_tok=12):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = p_tok
    mock_usage.completion_tokens = c_tok
    mock_usage.total_tokens = p_tok + c_tok
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model = model
    mock_resp.usage = mock_usage
    return mock_resp


def test_post_chat_successful_contract(client_with_temp_db):
    """Test successful POST /chat returns expected JSON contract and persists turn."""
    client, test_db = client_with_temp_db
    session_id = "s-contract-test"
    mock_resp = _make_mock_litellm_resp(content="The capital of France is Paris.", model="gemini/gemini-3.5-flash-lite", p_tok=42, c_tok=9)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_resp

        payload = {"session_id": session_id, "message": "What is the capital of France?"}
        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == session_id
        assert data["reply"] == "The capital of France is Paris."
        metadata = data["metadata"]
        assert metadata["resolution_source"] == "LIVE_INFERENCE"
        assert metadata["model"] == "gemini/gemini-3.5-flash-lite"
        assert isinstance(metadata["latency_ms"], int)
        assert metadata["latency_ms"] >= 0
        assert metadata["input_tokens"] == 42
        assert metadata["output_tokens"] == 9
        assert metadata["key_id"] == "key_1"
        assert metadata["fallback_attempts"] == 0

        # Verify turn was persisted to SQLite
        messages = get_session_messages(session_id, db_path=test_db)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the capital of France?"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "The capital of France is Paris."


def test_post_chat_multi_turn_context(client_with_temp_db):
    """Test that multi-turn context accumulates and is sent to LiteLLM in exact order."""
    client, test_db = client_with_temp_db
    session_id = "s-multiturn"

    resp1 = _make_mock_litellm_resp(content="Nice to meet you.")
    resp2 = _make_mock_litellm_resp(content="Your favourite colour is purple.")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [resp1, resp2]

        # Turn 1
        r1 = client.post("/chat", json={"session_id": session_id, "message": "My favourite colour is purple."})
        assert r1.status_code == 200

        # Turn 2
        r2 = client.post("/chat", json={"session_id": session_id, "message": "What is my favourite colour?"})
        assert r2.status_code == 200

        # Inspect messages sent to LiteLLM on Turn 2
        call_turn_2_messages = mock_litellm.call_args_list[1][1]["messages"]
        assert len(call_turn_2_messages) == 4
        assert call_turn_2_messages[0]["role"] == "system"
        assert call_turn_2_messages[1] == {"role": "user", "content": "My favourite colour is purple."}
        assert call_turn_2_messages[2] == {"role": "assistant", "content": "Nice to meet you."}
        assert call_turn_2_messages[3] == {"role": "user", "content": "What is my favourite colour?"}


def test_post_chat_system_prompt_precedence_and_override(client_with_temp_db):
    """Test custom system prompt persistence and precedence."""
    client, test_db = client_with_temp_db
    session_id = "s-sys-override"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = _make_mock_litellm_resp(content="OK")

        # Turn 1: with explicit system prompt
        client.post("/chat", json={"session_id": session_id, "message": "Hi", "system_prompt": "You are a poet."})
        assert get_session_system_prompt(session_id, db_path=test_db) == "You are a poet."
        assert mock_litellm.call_args_list[0][1]["messages"][0]["content"] == "You are a poet."

        # Turn 2: without system prompt (should retain "You are a poet.")
        client.post("/chat", json={"session_id": session_id, "message": "Hi again"})
        assert get_session_system_prompt(session_id, db_path=test_db) == "You are a poet."
        assert mock_litellm.call_args_list[1][1]["messages"][0]["content"] == "You are a poet."

        # Turn 3: with new override
        client.post("/chat", json={"session_id": session_id, "message": "Hi 3", "system_prompt": "You are a chef."})
        assert get_session_system_prompt(session_id, db_path=test_db) == "You are a chef."
        assert mock_litellm.call_args_list[2][1]["messages"][0]["content"] == "You are a chef."


def test_post_chat_session_isolation(client_with_temp_db):
    """Test that Session B does not receive history from Session A."""
    client, test_db = client_with_temp_db

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = _make_mock_litellm_resp(content="Reply")

        # Turn in Session A
        client.post("/chat", json={"session_id": "session-A", "message": "Secret message from A"})

        # Turn in Session B
        client.post("/chat", json={"session_id": "session-B", "message": "Hello from B"})

        session_b_messages = mock_litellm.call_args_list[1][1]["messages"]
        all_text = " ".join(m["content"] for m in session_b_messages)
        assert "Secret message from A" not in all_text


def test_post_chat_failed_request_does_not_persist(client_with_temp_db):
    """Test that a failed chat request rolls back and creates no session or messages."""
    client, test_db = client_with_temp_db
    session_id = "s-failed-turn"

    # Simulate all providers failing
    response = client.post(
        "/chat",
        json={"session_id": session_id, "message": "This will fail"},
        headers={"X-Simulate-Failure": "*=429"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}

    # Verify session was NOT created
    assert session_exists(session_id, db_path=test_db) is False
    assert len(get_session_messages(session_id, db_path=test_db)) == 0


def test_post_chat_malformed_simulation_header_returns_400(client_with_temp_db):
    """Test that invalid X-Simulate-Failure header returns 400 validation error."""
    client, _ = client_with_temp_db

    response = client.post(
        "/chat",
        json={"session_id": "s-bad-header", "message": "Hello"},
        headers={"X-Simulate-Failure": "key_1=not_a_number"},
    )

    assert response.status_code == 400
    assert "invalid X-Simulate-Failure header" in response.json()["detail"]


def test_post_chat_validation_errors(client_with_temp_db):
    """Test that invalid payloads return 422."""
    client, _ = client_with_temp_db

    # Empty message
    r1 = client.post("/chat", json={"session_id": "s1", "message": ""})
    assert r1.status_code == 422

    # Whitespace message
    r2 = client.post("/chat", json={"session_id": "s1", "message": "   "})
    assert r2.status_code == 422

    # Message > 4000 characters
    r3 = client.post("/chat", json={"session_id": "s1", "message": "x" * 4001})
    assert r3.status_code == 422

    # Session ID > 128 characters
    r4 = client.post("/chat", json={"session_id": "s" * 129, "message": "Hi"})
    assert r4.status_code == 422
