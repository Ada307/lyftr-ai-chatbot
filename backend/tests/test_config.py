"""Tests for environment configuration and SQLite path parsing."""

from pathlib import Path
import pytest

from app.config import ApiKeyEntry, Settings, parse_sqlite_path


def test_api_key_entry_confidentiality():
    """Verify ApiKeyEntry string representation does not expose raw secret."""
    entry = ApiKeyEntry(key_id="key_1", key_value="super_secret_token_123")
    rep = repr(entry)
    assert "super_secret_token_123" not in rep
    assert "key_1" in rep
    assert "***" in rep


def test_settings_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Verify settings parses comma-separated keys, origins, and boolean flags."""
    monkeypatch.setenv("LLM_API_KEYS", "key_alpha, key_beta,  , key_gamma  ")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000, http://example.com")
    monkeypatch.setenv("ENABLE_FAILURE_SIMULATION", "true")
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./data/test.db")

    settings = Settings()

    assert settings.keys_configured == 3
    assert [k.key_id for k in settings.api_keys] == ["key_1", "key_2", "key_3"]
    assert settings.get_key_by_id("key_1") == "key_alpha"
    assert settings.get_key_by_id("key_2") == "key_beta"
    assert settings.get_key_by_id("key_3") == "key_gamma"
    assert settings.get_key_by_id("key_99") is None

    # Verify settings repr does not leak raw keys
    assert "key_alpha" not in repr(settings)

    assert settings.allowed_origins == ["http://localhost:3000", "http://example.com"]
    assert settings.enable_failure_simulation is True


def test_sqlite_path_parsing(tmp_path: Path):
    """Verify SQLite path parsing for relative, absolute, and memory URLs."""
    base_dir = tmp_path / "project_root"
    base_dir.mkdir()

    # Relative path
    rel_path = parse_sqlite_path("sqlite:///./data/app.db", base_dir=base_dir)
    assert rel_path == (base_dir / "data" / "app.db").resolve()

    # In-memory
    mem_path = parse_sqlite_path("sqlite:///:memory:", base_dir=base_dir)
    assert str(mem_path) == ":memory:"

    # Absolute Windows path
    win_abs = parse_sqlite_path("sqlite:///C:/test_dir/app.db", base_dir=base_dir)
    assert "test_dir" in str(win_abs)
