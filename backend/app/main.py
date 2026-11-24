"""FastAPI application for RelayChat."""

from contextlib import asynccontextmanager
import logging
import time
from typing import AsyncGenerator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.database import (
    check_db_connection,
    get_session_messages,
    get_session_system_prompt,
    init_db,
    save_chat_turn,
    session_exists,
)
from app.llm_service import (
    AllProvidersUnavailableException,
    InvalidSimulationHeaderException,
    build_llm_messages,
    execute_chat_completion,
    init_cache,
    is_cache_initialized,
)
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionHistoryResponse,
)

logger = logging.getLogger("relaychat")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager initializing database and in-memory response cache."""
    init_db()
    init_cache()
    yield


app = FastAPI(
    title="RelayChat",
    description="LLM chatbot with API key failover, response caching, and SQLite session storage.",
    version="0.1.0",
    lifespan=lifespan,
)

# Apply CORS configuration supporting frontend origins and failure simulation headers
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely catch unhandled exceptions without leaking stack traces or credentials."""
    logger.error("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Backend readiness check",
)
def get_health(current_settings: Settings = Depends(get_settings)) -> JSONResponse:
    """Readiness endpoint.

    Verifies SQLite connectivity, at least 2 configured keys, and LiteLLM response cache readiness.
    """
    db_ok = check_db_connection(current_settings.db_path)
    keys_count = current_settings.keys_configured
    cache_ready = is_cache_initialized()

    is_ready = db_ok and (keys_count >= 2) and cache_ready

    if is_ready:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "ok",
                "keys_configured": keys_count,
                "cache_enabled": True,
            },
        )
    else:
        unmet = []
        if not db_ok:
            unmet.append("database unavailable")
        if keys_count < 2:
            unmet.append(f"insufficient keys configured ({keys_count}/2)")
        if not cache_ready:
            unmet.append("response cache uninitialized")

        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "keys_configured": keys_count,
                "cache_enabled": cache_ready,
                "detail": f"Service not ready: {', '.join(unmet)}",
            },
        )


@app.get(
    "/sessions/{session_id}/messages",
    response_model=SessionHistoryResponse,
    summary="Retrieve session message history",
)
def get_messages(
    session_id: str,
    current_settings: Settings = Depends(get_settings),
) -> SessionHistoryResponse:
    """Retrieve message history for an existing session in oldest-first order.

    Returns HTTP 404 if the session does not exist.
    """
    if not session_exists(session_id, db_path=current_settings.db_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="session not found",
        )

    messages = get_session_messages(session_id, db_path=current_settings.db_path)
    return SessionHistoryResponse(session_id=session_id, messages=messages)


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Send chat message with multi-turn context, key failover, and caching",
)
async def post_chat(
    request: ChatRequest,
    x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
    current_settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """Multi-turn chat endpoint with automatic key failover, response caching, and metadata."""
    start_time = time.perf_counter()

    # 1. Retrieve session context from SQLite
    stored_history = get_session_messages(
        request.session_id, db_path=current_settings.db_path
    )
    stored_system_prompt = get_session_system_prompt(
        request.session_id, db_path=current_settings.db_path
    )

    # 2. Construct messages with system prompt precedence
    llm_messages, _ = build_llm_messages(
        current_message=request.message,
        stored_history=stored_history,
        request_system_prompt=request.system_prompt,
        stored_system_prompt=stored_system_prompt,
    )

    # 3. Perform inference via LLM Gateway with ordered fallback and caching
    try:
        completion_result = await execute_chat_completion(
            messages=llm_messages,
            simulate_header=x_simulate_failure,
            settings=current_settings,
        )
    except InvalidSimulationHeaderException as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid X-Simulate-Failure header: {str(exc)}",
        )
    except AllProvidersUnavailableException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="all providers unavailable",
        )

    # 4. Measure end-to-end latency in milliseconds independently for this request
    latency_ms = int(round((time.perf_counter() - start_time) * 1000))

    # 5. Assemble metadata (null key_id and 0 fallback_attempts on cache hits)
    metadata = ChatMetadata(
        resolution_source=completion_result.resolution_source,
        model=completion_result.model,
        latency_ms=latency_ms,
        input_tokens=completion_result.input_tokens,
        output_tokens=completion_result.output_tokens,
        key_id=completion_result.key_id,
        fallback_attempts=completion_result.fallback_attempts,
    )

    # 6. Atomically persist conversation turn in SQLite
    save_chat_turn(
        session_id=request.session_id,
        user_message=request.message,
        assistant_message=completion_result.reply,
        assistant_metadata=metadata,
        system_prompt=request.system_prompt,
        db_path=current_settings.db_path,
    )

    return ChatResponse(
        session_id=request.session_id,
        reply=completion_result.reply,
        metadata=metadata,
    )
