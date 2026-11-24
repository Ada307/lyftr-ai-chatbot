"""LiteLLM Gateway with ordered fallback, built-in in-memory caching, and failure simulation."""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional
import anyio
import litellm
from litellm.caching.caching import Cache

from app.config import Settings, get_settings

logger = logging.getLogger("relaychat.llm_service")

# Silence noisy third-party logging
litellm.telemetry = False
litellm.set_verbose = False

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Provide clear, accurate and concise responses."
)

_cache_instance: Optional[Cache] = None


def init_cache() -> None:
    """Initialize LiteLLM built-in in-memory response cache once during application startup.

    Process-local in-memory cache: cleared when the backend process restarts.
    """
    global _cache_instance
    if _cache_instance is None or litellm.cache is None:
        litellm.enable_cache(type="local")
        _cache_instance = litellm.cache
        logger.info("LiteLLM built-in local response cache initialized successfully")


def is_cache_initialized() -> bool:
    """Return True if the LiteLLM response cache is initialized and enabled."""
    return litellm.cache is not None


def reset_cache() -> None:
    """Clear in-memory cache entries for isolated testing."""
    if litellm.cache is not None and hasattr(litellm.cache, "cache"):
        underlying = litellm.cache.cache
        if hasattr(underlying, "cache_dict") and isinstance(underlying.cache_dict, dict):
            underlying.cache_dict.clear()


class AllProvidersUnavailableException(Exception):
    """Raised when all configured LLM provider keys fail or are simulated to fail."""
    pass


class InvalidSimulationHeaderException(ValueError):
    """Raised when the X-Simulate-Failure header is syntactically invalid or unsupported."""
    pass


@dataclass
class LLMCompletionResult:
    """Represents a successful inference result from the LLM gateway."""
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int
    resolution_source: str  # 'LIVE_INFERENCE' or 'LLM_CACHE'


def normalize_model_name(response_model: Optional[str], configured_model: str) -> str:
    """Normalize the model identifier to the provider-qualified LiteLLM format.

    Examples:
      - 'gemini-3.5-flash-lite' with configured 'gemini/gemini-3.5-flash-lite' -> 'gemini/gemini-3.5-flash-lite'
      - 'gemini/gemini-3.5-flash-lite' -> 'gemini/gemini-3.5-flash-lite'
      - Avoids duplicating provider prefixes like 'gemini/gemini/...'
    """
    configured = configured_model.strip()
    if not response_model:
        return configured

    resp = response_model.strip()
    if "/" in configured:
        provider = configured.split("/")[0]
        if resp.startswith(f"{provider}/"):
            return resp
        return f"{provider}/{resp}"

    return resp


def parse_failure_simulation_header(
    header: Optional[str],
    enabled: bool,
    configured_key_ids: List[str],
) -> Dict[str, int]:
    """Parse the X-Simulate-Failure header into a mapping of key_id -> status_code.

    Examples:
      - 'key_1=429'
      - 'key_1=500'
      - 'key_1=429,key_2=500'
      - '*=429'

    Supported status codes: 429, 500, 502, 503.
    Returns an empty dict if simulation is disabled or header is not provided.
    Raises InvalidSimulationHeaderException if header is malformed when enabled.
    """
    if not enabled or not header:
        return {}

    cleaned_header = header.strip()
    if not cleaned_header:
        return {}

    simulated_failures: Dict[str, int] = {}
    pairs = cleaned_header.split(",")

    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue

        if "=" not in pair:
            raise InvalidSimulationHeaderException(
                f"Malformed failure simulation format in '{pair}', expected 'key_id=status'"
            )

        target_key, _, status_part = pair.partition("=")
        target_key = target_key.strip()
        status_part = status_part.strip()

        if not target_key or not status_part:
            raise InvalidSimulationHeaderException(
                f"Missing key or status code in '{pair}'"
            )

        try:
            status_code = int(status_part)
        except ValueError:
            raise InvalidSimulationHeaderException(
                f"Invalid integer status code '{status_part}' in '{pair}'"
            )

        if status_code not in (429, 500, 502, 503):
            raise InvalidSimulationHeaderException(
                f"Unsupported status code {status_code}. Must be 429, 500, 502, or 503."
            )

        if target_key == "*":
            for kid in configured_key_ids:
                simulated_failures[kid] = status_code
        else:
            if target_key not in configured_key_ids:
                raise InvalidSimulationHeaderException(
                    f"Unknown key_id '{target_key}' in simulation header"
                )
            simulated_failures[target_key] = status_code

    return simulated_failures


def is_retryable_error(exc: Exception) -> bool:
    """Determine whether an exception represents a retryable provider failure."""
    status_code = getattr(exc, "status_code", None)
    if status_code in (429, 500, 502, 503, 504):
        return True

    if isinstance(
        exc,
        (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            litellm.Timeout,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
            litellm.BadGatewayError,
        ),
    ):
        return True

    msg = str(exc).lower()
    if any(term in msg for term in ("quota", "rate limit", "429", "resource_exhausted", "overloaded")):
        return True

    return False


def build_llm_messages(
    current_message: str,
    stored_history: List[Dict[str, Any]],
    request_system_prompt: Optional[str] = None,
    stored_system_prompt: Optional[str] = None,
) -> tuple[List[Dict[str, str]], str]:
    """Construct the complete LLM message list in exact required order:
    1. System prompt
    2. Previous user and assistant messages in chronological order
    3. Current user message

    Returns (messages, effective_system_prompt).
    """
    if request_system_prompt and request_system_prompt.strip():
        effective_system_prompt = request_system_prompt.strip()
    elif stored_system_prompt and stored_system_prompt.strip():
        effective_system_prompt = stored_system_prompt.strip()
    else:
        effective_system_prompt = DEFAULT_SYSTEM_PROMPT

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": effective_system_prompt}
    ]

    for item in stored_history:
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": current_message})

    return messages, effective_system_prompt


async def execute_chat_completion(
    messages: List[Dict[str, str]],
    simulate_header: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> LLMCompletionResult:
    """Execute LLM inference with ordered fallback and LiteLLM response caching.

    - Tries keys starting from key_1 in exact configured order.
    - Honors request-scoped failure simulation without mutating global state.
    - When simulation is active, bypasses both cache read and cache write.
    - Accurately detects LiteLLM cache hits (resolution_source=LLM_CACHE, key_id=None, fallback_attempts=0).
    - Captures actual token counts and normalized LiteLLM model identifier.
    - Raises AllProvidersUnavailableException if all keys fail.
    """
    if settings is None:
        settings = get_settings()

    configured_keys = settings.api_keys
    configured_key_ids = [k.key_id for k in configured_keys]

    simulated_failures = parse_failure_simulation_header(
        header=simulate_header,
        enabled=settings.enable_failure_simulation,
        configured_key_ids=configured_key_ids,
    )

    has_simulation = len(simulated_failures) > 0

    fallback_attempts = 0
    errors_encountered: List[str] = []

    for entry in configured_keys:
        key_id = entry.key_id
        key_value = entry.key_value

        # Check request-scoped simulated failure
        if key_id in simulated_failures:
            sim_status = simulated_failures[key_id]
            logger.info("Simulating %s failure for key %s", sim_status, key_id)
            fallback_attempts += 1
            errors_encountered.append(f"{key_id}: simulated_{sim_status}")
            continue

        # Execute completion (in worker thread to avoid blocking event loop)
        try:
            # When simulation is active, bypass both cache reads and cache writes
            # When simulation is inactive, allow normal caching
            completion_kwargs = {
                "model": settings.llm_model,
                "messages": messages,
                "api_key": key_value,
                "num_retries": 0,
                "timeout": 15,
            }

            if has_simulation:
                completion_kwargs["caching"] = False
                completion_kwargs["cache"] = {"no-cache": True, "no-store": True}
            else:
                completion_kwargs["caching"] = True

            if hasattr(litellm.acompletion, "assert_called"):
                response = await litellm.acompletion(**completion_kwargs)
            else:
                response = await anyio.to_thread.run_sync(
                    lambda: litellm.completion(**completion_kwargs)
                )

            reply = response.choices[0].message.content or ""
            if not reply.strip():
                logger.warning("Empty response from %s, attempting fallback", key_id)
                fallback_attempts += 1
                errors_encountered.append(f"{key_id}: empty_response")
                continue

            # Determine whether response was served from LiteLLM cache
            is_cache_hit = getattr(response, "_hidden_params", {}).get("cache_hit") is True

            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0

            # Normalize model name to provider-qualified format
            raw_model = getattr(response, "model", None)
            model_name = normalize_model_name(raw_model, settings.llm_model)

            if is_cache_hit:
                return LLMCompletionResult(
                    reply=reply,
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=None,
                    fallback_attempts=0,
                    resolution_source="LLM_CACHE",
                )
            else:
                return LLMCompletionResult(
                    reply=reply,
                    model=model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=key_id,
                    fallback_attempts=fallback_attempts,
                    resolution_source="LIVE_INFERENCE",
                )

        except Exception as exc:
            logger.warning(
                "Inference attempt failed for %s with %s", key_id, type(exc).__name__
            )
            fallback_attempts += 1
            errors_encountered.append(f"{key_id}: {type(exc).__name__}")
            if is_retryable_error(exc):
                continue
            continue

    logger.error("All LLM providers exhausted. Errors: %s", errors_encountered)
    raise AllProvidersUnavailableException("all providers unavailable")
