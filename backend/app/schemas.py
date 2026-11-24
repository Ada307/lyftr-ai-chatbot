"""Pydantic schemas for request and response validation."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_serializer


class ChatRequest(BaseModel):
    """Schema for chat requests."""
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(...)
    system_prompt: Optional[str] = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("session_id must be a string")
        if not v.strip():
            raise ValueError("session_id cannot be empty or whitespace-only")
        if len(v) > 128:
            raise ValueError("session_id cannot exceed 128 characters")
        return v  # Preserved exactly

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("message must be a string")
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("message cannot be empty or whitespace-only")
        if len(trimmed) > 4000:
            raise ValueError("message length cannot exceed 4000 characters after trimming")
        return trimmed

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            if not isinstance(v, str):
                raise ValueError("system_prompt must be a string")
            trimmed = v.strip()
            return trimmed if trimmed else None
        return None


class ChatMetadata(BaseModel):
    """Per-request metadata tracking resolution, tokens, and fallback."""
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int = Field(..., ge=0)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    key_id: Optional[str] = None
    fallback_attempts: int = Field(..., ge=0)


class ChatResponse(BaseModel):
    """Chat exchange response echoing session_id and metadata."""
    session_id: str
    reply: str = Field(..., min_length=1)
    metadata: ChatMetadata


class HistoryMessageItem(BaseModel):
    """Single message in a session history conversation."""
    role: str
    content: str
    metadata: Optional[ChatMetadata] = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        """Omit metadata field entirely for user messages where metadata is None."""
        res = handler(self)
        if res.get("metadata") is None:
            res.pop("metadata", None)
        return res


class SessionHistoryResponse(BaseModel):
    """Ordered message history for a specific session."""
    session_id: str
    messages: List[HistoryMessageItem]


class HealthResponse(BaseModel):
    """System readiness check response model."""
    status: str
    keys_configured: int
    cache_enabled: bool
    detail: Optional[str] = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        res = handler(self)
        if res.get("detail") is None:
            res.pop("detail", None)
        return res
