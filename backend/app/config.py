"""Application configuration loaded from environment variables."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Project root directory: lyftr-ai-chatbot/
# Path resolution: config.py -> app/ -> backend/ -> lyftr-ai-chatbot/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Working directory used to resolve relative SQLite database paths
# Relative paths such as 'sqlite:///./data/app.db' are resolved against PROJECT_ROOT
RELATIVE_DB_BASE_DIR = PROJECT_ROOT

# Load root .env file if it exists, without overriding existing OS environment variables
_DOTENV_PATH = PROJECT_ROOT / ".env"
if _DOTENV_PATH.exists():
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


@dataclass(frozen=True)
class ApiKeyEntry:
    """Represents a configured API key with a stable identifier (key_1, key_2, ...)."""
    key_id: str
    key_value: str

    def __repr__(self) -> str:
        # Never expose actual secrets in logs or representations
        return f"ApiKeyEntry(key_id='{self.key_id}', key_value='***')"


def parse_sqlite_path(database_url: str, base_dir: Path = RELATIVE_DB_BASE_DIR) -> Path:
    """Parse a SQLite database URL into a filesystem Path.

    Supported formats:
      - sqlite:///./data/app.db (relative to base_dir)
      - sqlite:////data/app.db  (absolute path, e.g. Linux/Docker /data/app.db)
      - sqlite:///C:/path/app.db (Windows absolute path)
      - sqlite:///:memory:      (in-memory SQLite database)

    Relative paths are resolved against base_dir (default: PROJECT_ROOT).
    """
    url = database_url.strip()
    if url.startswith("sqlite:///"):
        raw_path = url[10:]
    elif url.startswith("sqlite://"):
        raw_path = url[9:]
    else:
        raw_path = url

    if raw_path == ":memory:":
        return Path(":memory:")

    # Check for absolute Windows path (e.g. C:/ or C:\)
    if len(raw_path) >= 2 and raw_path[1] == ":":
        return Path(raw_path).resolve()

    # Check for absolute Unix/Docker path (e.g. /data/app.db)
    if raw_path.startswith("/") or raw_path.startswith("\\"):
        p = Path(raw_path)
        if os.name != "nt":
            return p.resolve()
        # On Windows, resolve root-relative paths against current drive root
        return Path(os.path.abspath(raw_path))

    # Relative path resolved against base_dir
    return (base_dir / Path(raw_path)).resolve()


class Settings:
    """Application settings parsed from environment variables."""

    def __init__(self) -> None:
        raw_keys = os.getenv("LLM_API_KEYS", "")
        parsed_keys: List[ApiKeyEntry] = []
        idx = 1
        for item in raw_keys.split(","):
            cleaned = item.strip()
            if cleaned:
                parsed_keys.append(ApiKeyEntry(key_id=f"key_{idx}", key_value=cleaned))
                idx += 1
        self.api_keys: List[ApiKeyEntry] = parsed_keys

        self.llm_model: str = os.getenv("LLM_MODEL", "your_litellm_model").strip()

        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/app.db").strip()
        self.db_path: Path = parse_sqlite_path(self.database_url, RELATIVE_DB_BASE_DIR)

        raw_origins = os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
        )
        self.allowed_origins: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]

        raw_fail_sim = os.getenv("ENABLE_FAILURE_SIMULATION", "false").strip().lower()
        self.enable_failure_simulation: bool = raw_fail_sim in (
            "true", "1", "yes", "t", "y", "on"
        )

    @property
    def keys_configured(self) -> int:
        """Return the count of configured API keys."""
        return len(self.api_keys)

    def get_key_by_id(self, key_id: str) -> Optional[str]:
        """Retrieve key secret by its stable key_id (confidential)."""
        for entry in self.api_keys:
            if entry.key_id == key_id:
                return entry.key_value
        return None

    def __repr__(self) -> str:
        # Never print actual keys
        return (
            f"Settings(keys_configured={self.keys_configured}, "
            f"llm_model='{self.llm_model}', "
            f"database_url='{self.database_url}', "
            f"db_path='{self.db_path}', "
            f"allowed_origins={self.allowed_origins}, "
            f"enable_failure_simulation={self.enable_failure_simulation})"
        )


def get_settings() -> Settings:
    """Return a fresh Settings instance from current environment variables."""
    return Settings()
