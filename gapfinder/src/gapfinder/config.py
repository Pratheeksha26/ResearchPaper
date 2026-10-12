from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
GAPFINDER_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = ROOT_DIR / "runs"
DEFAULT_CACHE_DIR = ROOT_DIR / ".cache"
DOTENV_PATH = GAPFINDER_DIR / ".env"
_PREEXISTING_ENV = set(os.environ)
_DOTENV_INITIAL_VALUES: dict[str, str] = {}


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding inherited environment values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = _clean_env_value(value)
        if name:
            os.environ.setdefault(name, value)
            if name not in _PREEXISTING_ENV:
                _DOTENV_INITIAL_VALUES[name] = value


def _clean_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


_load_dotenv(GAPFINDER_DIR / ".env")


def get_secret(name: str) -> str | None:
    """Read a secret with inherited environment precedence and live dotenv refresh."""
    current = os.getenv(name)
    if name in _PREEXISTING_ENV:
        return _clean_env_value(current or "") or None
    if name not in _DOTENV_INITIAL_VALUES:
        return _clean_env_value(current or "") or None
    if current != _DOTENV_INITIAL_VALUES[name]:
        return _clean_env_value(current or "") or None
    if not DOTENV_PATH.is_file():
        return None
    for raw_line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        env_name, value = line.split("=", 1)
        if env_name.strip() == name:
            refreshed = _clean_env_value(value)
            os.environ[name] = refreshed
            _DOTENV_INITIAL_VALUES[name] = refreshed
            return refreshed or None
    return None


def get_secret_source(name: str) -> str:
    """Report where a configured secret resolves, never its contents."""
    if name in _PREEXISTING_ENV:
        return "environment" if os.getenv(name) else "missing"
    current = os.getenv(name)
    if name not in _DOTENV_INITIAL_VALUES:
        return "environment" if current else "missing"
    if current != _DOTENV_INITIAL_VALUES[name]:
        return "environment" if current else "missing"
    return "dotenv" if DOTENV_PATH.is_file() else "missing"


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:  # pragma: no cover - config safety
        raise ValueError(f"Environment variable {name} must be an integer: {value!r}") from exc


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - config safety
        raise ValueError(f"Environment variable {name} must be a float: {value!r}") from exc


MODEL = os.getenv("GAPFINDER_MODEL", "")
API_KEY = os.getenv("GAPFINDER_API_KEY")
OPENALEX_API_KEY = _clean_env_value(
    os.getenv("OPENALEX_API_KEY") or os.getenv("GAPFINDER_OPENALEX_API_KEY") or ""
) or None
EMAIL = os.getenv("OPENALEX_MAILTO") or os.getenv("GAPFINDER_MAILTO") or "mailto:research@example.com"
SEED = getenv_int("GAPFINDER_SEED", 42)
TEMPERATURE = getenv_float("GAPFINDER_TEMPERATURE", 0.0)
DEDUPE_THRESHOLD = getenv_float("GAPFINDER_DEDUPE_THRESHOLD", 0.95)
RUNS_DIR = Path(os.getenv("GAPFINDER_RUNS_DIR", str(DEFAULT_RUNS_DIR)))
CACHE_DIR = Path(os.getenv("GAPFINDER_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
HTTP_CACHE_PATH = CACHE_DIR / "openalex_cache.json"
OPENALEX_BASE_URL = os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org")

__all__ = [
    "ROOT_DIR",
    "DOTENV_PATH",
    "get_secret",
    "get_secret_source",
    "MODEL",
    "API_KEY",
    "OPENALEX_API_KEY",
    "EMAIL",
    "SEED",
    "TEMPERATURE",
    "DEDUPE_THRESHOLD",
    "RUNS_DIR",
    "CACHE_DIR",
    "HTTP_CACHE_PATH",
    "OPENALEX_BASE_URL",
]
