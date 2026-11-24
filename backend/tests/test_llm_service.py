"""Unit tests for LiteLLM Gateway, ordered fallback, and failure simulation."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.config import ApiKeyEntry, Settings
from app.llm_service import (
    AllProvidersUnavailableException,
    InvalidSimulationHeaderException,
    build_llm_messages,
    execute_chat_completion,
    parse_failure_simulation_header,
)


def make_mock_completion(
    content: str = "Test response",
    model: str = "gemini/gemini-3.5-flash-lite",
    prompt_tokens: int = 15,
    completion_tokens: int = 7,
):
    """Create a mock LiteLLM completion response object."""
    mock_choice = MagicMock()
    mock_choice.message.content = content

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = prompt_tokens
    mock_usage.completion_tokens = completion_tokens
    mock_usage.total_tokens = prompt_tokens + completion_tokens

    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_resp.model = model
    mock_resp.usage = mock_usage
    return mock_resp


def get_mock_settings(keys=None, simulation_enabled=True):
    """Create test settings with custom keys."""
    settings = Settings()
    if keys is not None:
        settings.api_keys = [ApiKeyEntry(key_id=k, key_value=f"val_{k}") for k in keys]
    else:
        settings.api_keys = [
            ApiKeyEntry(key_id="key_1", key_value="val_1"),
            ApiKeyEntry(key_id="key_2", key_value="val_2"),
        ]
    settings.enable_failure_simulation = simulation_enabled
    settings.llm_model = "gemini/gemini-3.5-flash-lite"
    return settings


# ==============================================================================
# A. GATEWAY TESTS
# ==============================================================================


@pytest.mark.anyio
async def test_primary_key_succeeds():
    """Verify that key_1 succeeds on first attempt with 0 fallback attempts."""
    settings = get_mock_settings()
    mock_resp = make_mock_completion(content="Success 1", prompt_tokens=20, completion_tokens=10)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_resp

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            settings=settings,
        )

        assert result.reply == "Success 1"
        assert result.key_id == "key_1"
        assert result.fallback_attempts == 0
        assert result.input_tokens == 20
        assert result.output_tokens == 10
        assert result.model == "gemini/gemini-3.5-flash-lite"
        assert mock_litellm.call_count == 1
        assert mock_litellm.call_args[1]["api_key"] == "val_1"


@pytest.mark.anyio
async def test_primary_returns_429_backup_succeeds():
    """Verify failover to key_2 when key_1 encounters HTTP 429."""
    settings = get_mock_settings()
    err_429 = Exception("Rate limit exceeded")
    setattr(err_429, "status_code", 429)

    mock_resp_2 = make_mock_completion(content="Success from backup")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [err_429, mock_resp_2]

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            settings=settings,
        )

        assert result.reply == "Success from backup"
        assert result.key_id == "key_2"
        assert result.fallback_attempts == 1
        assert mock_litellm.call_count == 2
        assert mock_litellm.call_args_list[0][1]["api_key"] == "val_1"
        assert mock_litellm.call_args_list[1][1]["api_key"] == "val_2"


@pytest.mark.anyio
async def test_primary_returns_500_backup_succeeds():
    """Verify failover to key_2 when key_1 encounters HTTP 500."""
    settings = get_mock_settings()
    err_500 = Exception("Internal Server Error")
    setattr(err_500, "status_code", 500)

    mock_resp_2 = make_mock_completion(content="Success 500 backup")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [err_500, mock_resp_2]

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            settings=settings,
        )

        assert result.reply == "Success 500 backup"
        assert result.key_id == "key_2"
        assert result.fallback_attempts == 1


@pytest.mark.anyio
@pytest.mark.parametrize("status_code", [502, 503])
async def test_primary_returns_502_or_503_backup_succeeds(status_code):
    """Verify failover to key_2 when key_1 encounters HTTP 502 or 503."""
    settings = get_mock_settings()
    err = Exception(f"Server error {status_code}")
    setattr(err, "status_code", status_code)

    mock_resp_2 = make_mock_completion(content=f"Recovered from {status_code}")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [err, mock_resp_2]

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            settings=settings,
        )

        assert result.reply == f"Recovered from {status_code}"
        assert result.key_id == "key_2"
        assert result.fallback_attempts == 1


@pytest.mark.anyio
async def test_quota_error_triggers_failover():
    """Verify provider quota error triggers failover to next key."""
    settings = get_mock_settings()
    quota_err = Exception("Resource has been exhausted (e.g. check quota)")

    mock_resp_2 = make_mock_completion(content="Quota recovered")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [quota_err, mock_resp_2]

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            settings=settings,
        )

        assert result.reply == "Quota recovered"
        assert result.key_id == "key_2"
        assert result.fallback_attempts == 1


@pytest.mark.anyio
async def test_all_configured_keys_fail_raises_exception():
    """Verify AllProvidersUnavailableException is raised when all keys fail."""
    settings = get_mock_settings()
    err_503 = Exception("Unavailable")
    setattr(err_503, "status_code", 503)

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [err_503, err_503]

        with pytest.raises(AllProvidersUnavailableException):
            await execute_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                settings=settings,
            )

        assert mock_litellm.call_count == 2


@pytest.mark.anyio
async def test_three_or_more_keys_handling():
    """Verify ordered progression through 3 keys until success on key_3."""
    settings = get_mock_settings(keys=["key_1", "key_2", "key_3"])
    err_429 = Exception("Rate limit")
    setattr(err_429, "status_code", 429)
    err_500 = Exception("Server error")
    setattr(err_500, "status_code", 500)

    mock_resp_3 = make_mock_completion(content="Key 3 succeeded")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [err_429, err_500, mock_resp_3]

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            settings=settings,
        )

        assert result.reply == "Key 3 succeeded"
        assert result.key_id == "key_3"
        assert result.fallback_attempts == 2
        assert mock_litellm.call_count == 3


# ==============================================================================
# B. FAILURE SIMULATION TESTS
# ==============================================================================


@pytest.mark.anyio
async def test_simulate_key1_429_uses_key2():
    """Verify simulation header key_1=429 skips key_1 without provider call and uses key_2."""
    settings = get_mock_settings()
    mock_resp_2 = make_mock_completion(content="Key 2 via simulation")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_resp_2

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            simulate_header="key_1=429",
            settings=settings,
        )

        assert result.key_id == "key_2"
        assert result.fallback_attempts == 1
        assert result.reply == "Key 2 via simulation"
        # Only ONE real provider call was made (for key_2)
        assert mock_litellm.call_count == 1
        assert mock_litellm.call_args[1]["api_key"] == "val_2"


@pytest.mark.anyio
async def test_simulate_key1_500_uses_key2():
    """Verify simulation header key_1=500 skips key_1 and uses key_2."""
    settings = get_mock_settings()
    mock_resp_2 = make_mock_completion(content="Key 2 via 500 simulation")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_resp_2

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            simulate_header="key_1=500",
            settings=settings,
        )

        assert result.key_id == "key_2"
        assert result.fallback_attempts == 1
        assert mock_litellm.call_count == 1


@pytest.mark.anyio
async def test_simulate_wildcard_429_fails_all():
    """Verify simulation header *=429 triggers AllProvidersUnavailableException without provider calls."""
    settings = get_mock_settings()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        with pytest.raises(AllProvidersUnavailableException):
            await execute_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                simulate_header="*=429",
                settings=settings,
            )

        # Zero real provider calls should have been made
        assert mock_litellm.call_count == 0


@pytest.mark.anyio
async def test_simulate_multiple_keys_fail_all():
    """Verify simulation header key_1=429,key_2=500 raises AllProvidersUnavailableException."""
    settings = get_mock_settings()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        with pytest.raises(AllProvidersUnavailableException):
            await execute_chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                simulate_header="key_1=429,key_2=500",
                settings=settings,
            )

        assert mock_litellm.call_count == 0


@pytest.mark.anyio
async def test_disabled_simulation_ignores_header():
    """Verify simulation header is ignored when ENABLE_FAILURE_SIMULATION=false."""
    settings = get_mock_settings(simulation_enabled=False)
    mock_resp_1 = make_mock_completion(content="Key 1 responded")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.return_value = mock_resp_1

        result = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            simulate_header="key_1=429",  # Should be ignored
            settings=settings,
        )

        assert result.key_id == "key_1"
        assert result.fallback_attempts == 0
        assert mock_litellm.call_count == 1
        assert mock_litellm.call_args[1]["api_key"] == "val_1"


@pytest.mark.anyio
async def test_simulation_does_not_affect_subsequent_requests():
    """Verify that simulating a failure on request 1 does not affect request 2."""
    settings = get_mock_settings()
    mock_resp_2 = make_mock_completion(content="Key 2")
    mock_resp_1 = make_mock_completion(content="Key 1 back")

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_litellm:
        mock_litellm.side_effect = [mock_resp_2, mock_resp_1]

        # Request 1 with simulation
        res1 = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            simulate_header="key_1=429",
            settings=settings,
        )
        assert res1.key_id == "key_2"

        # Request 2 without simulation: key_1 must be tried and succeed
        res2 = await execute_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            simulate_header=None,
            settings=settings,
        )
        assert res2.key_id == "key_1"
        assert res2.fallback_attempts == 0


def test_simulation_header_parsing_validation():
    """Test validation errors for malformed simulation headers."""
    configured = ["key_1", "key_2"]

    # Valid
    assert parse_failure_simulation_header("key_1=429", True, configured) == {"key_1": 429}
    assert parse_failure_simulation_header("*=500", True, configured) == {"key_1": 500, "key_2": 500}

    # Invalid status code
    with pytest.raises(InvalidSimulationHeaderException):
        parse_failure_simulation_header("key_1=404", True, configured)

    # Missing status
    with pytest.raises(InvalidSimulationHeaderException):
        parse_failure_simulation_header("key_1=", True, configured)

    # Missing key
    with pytest.raises(InvalidSimulationHeaderException):
        parse_failure_simulation_header("=429", True, configured)

    # Unknown key
    with pytest.raises(InvalidSimulationHeaderException):
        parse_failure_simulation_header("key_99=429", True, configured)
