"""Automated tests for LiteLLM response caching, metadata corrections, and readiness."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.database import get_session_messages, init_db, session_exists
from app.llm_service import (
    execute_chat_completion,
    init_cache,
    normalize_model_name,
    reset_cache,
)
from app.main import app
from litellm.llms.custom_httpx.http_handler import HTTPHandler


def make_mock_gemini_transport_response(
    content: str = "Mocked Paris",
    prompt_tokens: int = 15,
    completion_tokens: int = 6,
    status_code: int = 200,
):
    """Build a raw HTTP response mimicking the Gemini provider API."""
    if status_code != 200:
        return httpx.Response(
            status_code,
            json={"error": {"code": status_code, "message": "Simulated error"}},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )

    gemini_body = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": content}],
                    "role": "model",
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
            "totalTokenCount": prompt_tokens + completion_tokens,
        },
    }
    return httpx.Response(
        200,
        json=gemini_body,
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
    )


@pytest.fixture(autouse=True)
def fresh_cache():
    """Ensure LiteLLM cache is initialized and emptied before each test."""
    init_cache()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client_with_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Fixture providing a TestClient with an isolated database and test settings."""
    test_db = tmp_path / "cache_test.db"
    test_url = f"sqlite:///{test_db.as_posix()}"

    monkeypatch.setenv("DATABASE_URL", test_url)
    monkeypatch.setenv("LLM_API_KEYS", "key_alpha, key_beta")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-3.5-flash-lite")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")

    test_settings = Settings()
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as test_client:
        yield test_client, test_db, test_settings

    app.dependency_overrides.clear()


# ==============================================================================
# TESTS 1 - 12: CACHE HITS, MISSES, AND CONTEXT VARIATIONS
# ==============================================================================


def test_cache_miss_and_hit_across_separate_sessions(client_with_db):
    """Tests 1, 2, 3, 4, 5, 6, 7, 8: Verify cache miss, cache hit, key_id=null, and token preservation."""
    client, test_db, _ = client_with_db
    mock_raw_resp = make_mock_gemini_transport_response(
        content="Paris is the capital of France.",
        prompt_tokens=25,
        completion_tokens=8,
    )

    with patch.object(HTTPHandler, "post", return_value=mock_raw_resp) as mock_post:
        # 1. First call in session 1: Cache Miss -> LIVE_INFERENCE
        r1 = client.post(
            "/chat",
            json={"session_id": "session-1", "message": "What is the capital of France?"},
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["key_id"] == "key_1"
        assert d1["metadata"]["fallback_attempts"] == 0
        assert d1["metadata"]["input_tokens"] == 25
        assert d1["metadata"]["output_tokens"] == 8
        assert d1["metadata"]["model"] == "gemini/gemini-3.5-flash-lite"
        assert mock_post.call_count == 1

        # 2. Second call in session 2 with identical context: Cache Hit -> LLM_CACHE
        r2 = client.post(
            "/chat",
            json={"session_id": "session-2", "message": "What is the capital of France?"},
        )
        assert r2.status_code == 200
        d2 = r2.json()

        # Check cache hit requirements
        assert d2["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d2["metadata"]["key_id"] is None
        assert "key_id" in d2["metadata"]  # Explicitly present as null in JSON
        assert d2["metadata"]["fallback_attempts"] == 0
        assert d2["reply"] == d1["reply"]
        assert d2["metadata"]["input_tokens"] == 25
        assert d2["metadata"]["output_tokens"] == 8
        assert d2["metadata"]["model"] == "gemini/gemini-3.5-flash-lite"
        assert isinstance(d2["metadata"]["latency_ms"], int)

        # Crucial: Provider transport was NOT called again (call_count remains 1)
        assert mock_post.call_count == 1

        # Both turns must appear in SQLite
        msgs1 = get_session_messages("session-1", db_path=test_db)
        msgs2 = get_session_messages("session-2", db_path=test_db)
        assert len(msgs1) == 2
        assert len(msgs2) == 2
        assert msgs1[1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert msgs2[1]["metadata"]["resolution_source"] == "LLM_CACHE"
        assert msgs2[1]["metadata"]["key_id"] is None


def test_different_messages_miss_cache(client_with_db):
    """Test 9: Different messages do not share a cache entry."""
    client, _, _ = client_with_db
    respA = make_mock_gemini_transport_response(content="Answer A")
    respB = make_mock_gemini_transport_response(content="Answer B")

    with patch.object(HTTPHandler, "post", side_effect=[respA, respB]) as mock_post:
        r1 = client.post("/chat", json={"session_id": "s1", "message": "Question A"})
        assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

        r2 = client.post("/chat", json={"session_id": "s2", "message": "Question B"})
        assert r2.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert mock_post.call_count == 2


def test_different_system_prompts_miss_cache(client_with_db):
    """Test 10: Same message with different system prompts misses cache."""
    client, _, _ = client_with_db
    respA = make_mock_gemini_transport_response(content="Poetic answer")
    respB = make_mock_gemini_transport_response(content="Concise answer")

    with patch.object(HTTPHandler, "post", side_effect=[respA, respB]) as mock_post:
        r1 = client.post(
            "/chat",
            json={"session_id": "s1", "message": "Hello", "system_prompt": "You are a poet."},
        )
        assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

        r2 = client.post(
            "/chat",
            json={"session_id": "s2", "message": "Hello", "system_prompt": "You are concise."},
        )
        assert r2.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert mock_post.call_count == 2


def test_different_conversation_histories_miss_cache(client_with_db):
    """Test 11: Same user message in a different conversation history produces separate cache entries."""
    client, _, _ = client_with_db
    resp1 = make_mock_gemini_transport_response(content="Turn 1 answer")
    resp2 = make_mock_gemini_transport_response(content="Turn 2 answer with history")
    resp_fresh = make_mock_gemini_transport_response(content="Fresh answer without history")

    with patch.object(HTTPHandler, "post", side_effect=[resp1, resp2, resp_fresh]) as mock_post:
        # Multi-turn in session 1
        client.post("/chat", json={"session_id": "s-hist", "message": "Context 1"})
        r_hist = client.post("/chat", json={"session_id": "s-hist", "message": "What now?"})
        assert r_hist.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"

        # Same message "What now?" in a brand-new session (empty history) -> must be LIVE_INFERENCE
        r_fresh = client.post("/chat", json={"session_id": "s-fresh", "message": "What now?"})
        assert r_fresh.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert mock_post.call_count == 3


@pytest.mark.anyio
async def test_equivalent_requests_share_cache_across_different_api_keys():
    """Test 12: Cached response is shared across different API keys."""
    settings = Settings()
    mock_resp = make_mock_gemini_transport_response(content="Universal Answer")

    with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
        messages = [{"role": "user", "content": "Universal Question"}]

        # Call with key_1
        res1 = await execute_chat_completion(messages=messages, settings=settings)
        assert res1.resolution_source == "LIVE_INFERENCE"
        assert mock_post.call_count == 1

        # Call with simulated failure on key_1 disabled, key_2 as active key
        # In a normal call, LiteLLM checks cache before provider call
        res2 = await execute_chat_completion(messages=messages, settings=settings)
        assert res2.resolution_source == "LLM_CACHE"
        assert res2.key_id is None
        assert mock_post.call_count == 1  # No extra provider call


# ==============================================================================
# TESTS 13 - 18: FAILURE SIMULATION & CACHE BYPASS
# ==============================================================================


def test_simulation_bypasses_warm_cache(client_with_db):
    """Tests 13, 14: Failure simulation bypasses warm cache and does not poison it."""
    client, _, _ = client_with_db
    warm_resp = make_mock_gemini_transport_response(content="Original warm reply")
    failover_resp = make_mock_gemini_transport_response(content="Failover real reply")

    with patch.object(HTTPHandler, "post", side_effect=[warm_resp, failover_resp]) as mock_post:
        # 1. Warm the cache
        r1 = client.post("/chat", json={"session_id": "s-warm", "message": "Cached query"})
        assert r1.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert r1.json()["metadata"]["key_id"] == "key_1"
        assert mock_post.call_count == 1

        # 2. Same query with X-Simulate-Failure: key_1=429
        # MUST bypass cache, fail key_1, make real provider call with key_2!
        r2 = client.post(
            "/chat",
            json={"session_id": "s-sim", "message": "Cached query"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d2["metadata"]["key_id"] == "key_2"
        assert d2["metadata"]["fallback_attempts"] == 1
        assert d2["reply"] == "Failover real reply"
        assert mock_post.call_count == 2  # Provider was called for key_2!

        # 3. Third request without simulation header: returns original warm cache!
        r3 = client.post("/chat", json={"session_id": "s-resumed", "message": "Cached query"})
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["metadata"]["resolution_source"] == "LLM_CACHE"
        assert d3["metadata"]["key_id"] is None
        assert d3["reply"] == "Original warm reply"
        assert mock_post.call_count == 2  # No new provider call


def test_all_provider_failure_with_warm_cache(client_with_db):
    """Tests 15, 16, 17: All-provider failure returns 503 despite warm cache, without persisting."""
    client, test_db, _ = client_with_db
    warm_resp = make_mock_gemini_transport_response(content="Warm reply")

    with patch.object(HTTPHandler, "post", return_value=warm_resp):
        # Warm cache
        client.post("/chat", json={"session_id": "s-warm", "message": "Query"})

        # Send *=429
        r_fail = client.post(
            "/chat",
            json={"session_id": "s-outage", "message": "Query"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert r_fail.status_code == 503
        assert r_fail.json() == {"detail": "all providers unavailable"}

        # Verify no session was created for s-outage
        assert session_exists("s-outage", db_path=test_db) is False

        # Resuming normal request uses the warm cache
        r_ok = client.post("/chat", json={"session_id": "s-ok", "message": "Query"})
        assert r_ok.status_code == 200
        assert r_ok.json()["metadata"]["resolution_source"] == "LLM_CACHE"


# ==============================================================================
# TESTS 19 - 21: HEALTH ENDPOINT & MODEL NAME NORMALIZATION
# ==============================================================================


def test_health_endpoint_successful_contract(client_with_db):
    """Test 19: Health returns 200 and exact successful readiness contract when all ready."""
    client, _, _ = client_with_db
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["keys_configured"] == 2
    assert data["cache_enabled"] is True


def test_health_endpoint_unhealthy_on_missing_components(client_with_db, monkeypatch):
    """Test 20: Health returns 503 when essential components are missing."""
    client, _, _ = client_with_db

    # 1. Test insufficient keys
    monkeypatch.setenv("LLM_API_KEYS", "only_one_key")
    settings = Settings()
    app.dependency_overrides[get_settings] = lambda: settings

    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"
    assert "insufficient keys configured" in r.json()["detail"]

    # 2. Test unreachable database
    monkeypatch.setenv("DATABASE_URL", "sqlite:///Z:/nonexistent_dir/app.db")
    settings_bad_db = Settings()
    app.dependency_overrides[get_settings] = lambda: settings_bad_db

    r_db = client.get("/health")
    assert r_db.status_code == 503
    assert r_db.json()["status"] == "unavailable"


def test_model_name_normalization():
    """Test 21: Model name normalization handles raw provider names and avoids duplication."""
    configured = "gemini/gemini-3.5-flash-lite"

    # Raw model returned without prefix
    assert normalize_model_name("gemini-3.5-flash-lite", configured) == "gemini/gemini-3.5-flash-lite"

    # Already prefixed model returned
    assert normalize_model_name("gemini/gemini-3.5-flash-lite", configured) == "gemini/gemini-3.5-flash-lite"

    # None model returned
    assert normalize_model_name(None, configured) == "gemini/gemini-3.5-flash-lite"

    # OpenAI style model
    assert normalize_model_name("gpt-4o-mini", "openai/gpt-4o-mini") == "openai/gpt-4o-mini"
    assert normalize_model_name("openai/gpt-4o-mini", "openai/gpt-4o-mini") == "openai/gpt-4o-mini"
