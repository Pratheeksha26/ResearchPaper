from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from gapfinder.config import CACHE_DIR

PACIFIC = ZoneInfo("America/Los_Angeles")


@dataclass(frozen=True)
class ModelLimits:
    rpm: int
    tpm: int
    rpd: int


MODEL_LIMITS: dict[str, ModelLimits] = {
    "gemini-3.8-flash": ModelLimits(rpm=5, tpm=250_000, rpd=20),
    "gemini-3.5-flash-lite": ModelLimits(rpm=15, tpm=250_000, rpd=500),
    "gemini-3.1-flash-lite": ModelLimits(rpm=15, tpm=250_000, rpd=500),
    "gemma-4-31b-it": ModelLimits(rpm=30, tpm=16_000, rpd=14_400),
    "gemma-4-26b-it": ModelLimits(rpm=30, tpm=16_000, rpd=14_400),
    "gemma-4-31b": ModelLimits(rpm=30, tpm=16_000, rpd=14_400),
    "gemma-4-26b": ModelLimits(rpm=30, tpm=16_000, rpd=14_400),
}

_LOCK_PATH = CACHE_DIR / "gemini_rate_limit.lock"


def selected_model() -> str:
    return os.getenv("GAPFINDER_MODEL", "").strip()


def require_selected_model(model: str) -> ModelLimits:
    selected = selected_model()
    if model != selected:
        raise ValueError(
            f"Model {model!r} is blocked; GAPFINDER_MODEL selects {selected!r}."
        )
    try:
        return MODEL_LIMITS[model]
    except KeyError as exc:
        raise ValueError(
            f"No free-tier quota configuration exists for model {model!r}."
        ) from exc


@contextmanager
def _state_lock() -> Iterator[None]:
    _LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_PATH.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                handle.fileno(), fcntl.LOCK_EX  # type: ignore[attr-defined]
            )
            try:
                yield
            finally:
                fcntl.flock(  # type: ignore[attr-defined]
                    handle.fileno(), fcntl.LOCK_UN  # type: ignore[attr-defined]
                )


def _budget_path() -> Path:
    return Path(
        os.getenv(
            "GAPFINDER_LLM_BUDGET_PATH",
            str(CACHE_DIR / "gemini_request_budget.json"),
        )
    )


def _pacific_date() -> str:
    return datetime.now(PACIFIC).date().isoformat()


def _seconds_until_pacific_reset() -> float:
    now = datetime.now(PACIFIC)
    next_day = now.date() + timedelta(days=1)
    reset = datetime.combine(next_day, datetime_time.min, tzinfo=PACIFIC)
    return max(1.0, (reset - now).total_seconds() + 1.0)


def wait_until_pacific_reset(
    model: str,
    *,
    reason: str,
    minimum_wait: float = 0.0,
) -> None:
    wait_seconds = max(_seconds_until_pacific_reset(), minimum_wait)
    reset_at = (datetime.now(PACIFIC) + timedelta(seconds=wait_seconds)).isoformat()
    print(
        f"{model} paused until {reset_at} ({reason}); the run will resume "
        "automatically.",
        flush=True,
    )
    time.sleep(wait_seconds)


def _read_state(path: Path, day: str) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM request-budget file is invalid: {path}") from exc
    if not isinstance(state, dict):
        raise RuntimeError(f"LLM request-budget state is invalid: {path}")
    if state.get("pacific_date") != day:
        return {"pacific_date": day, "models": {}}
    if not isinstance(state.get("models"), dict):
        raise RuntimeError(f"LLM request-budget state is invalid: {path}")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RequestReservation:
    def __init__(
        self,
        *,
        model: str,
        reservation_id: str,
        reserved_tokens: int,
        budget_path: Path,
        quota_bucket: str | None = None,
    ) -> None:
        self.model = model
        self.state_key = _model_state_key(model, quota_bucket)
        self.reservation_id = reservation_id
        self.reserved_tokens = reserved_tokens
        self.budget_path = budget_path
        self._settled = False

    def settle(self, actual_tokens: int | None) -> None:
        if self._settled or actual_tokens is None:
            return
        with _state_lock():
            state = _read_state(self.budget_path, _pacific_date())
            model_state = state["models"].setdefault(
                self.state_key, {"attempts_today": 0, "recent": []}
            )
            for request in model_state["recent"]:
                if request["id"] == self.reservation_id:
                    request["tokens"] = max(0, int(actual_tokens))
                    break
            _write_state(self.budget_path, state)
        self._settled = True

    def refund(self) -> None:
        """Release a reservation when transport failed before an HTTP response."""
        if self._settled:
            return
        with _state_lock():
            state = _read_state(self.budget_path, _pacific_date())
            model_state = state["models"].setdefault(
                self.state_key, {"attempts_today": 0, "recent": []}
            )
            model_state["attempts_today"] = max(
                0, int(model_state["attempts_today"]) - 1
            )
            model_state["recent"] = [
                request
                for request in model_state.get("recent", [])
                if request.get("id") != self.reservation_id
            ]
            _write_state(self.budget_path, state)
        self._settled = True


def _prune_recent(model_state: dict[str, Any], now: float) -> None:
    model_state["recent"] = [
        request
        for request in model_state.get("recent", [])
        if now - float(request["time"]) < 60.0
    ]


def _model_state_key(model: str, quota_bucket: str | None) -> str:
    if quota_bucket is None:
        return model
    return f"{model}::key-{quota_bucket}"


def reserve_request(
    model: str,
    estimated_tokens: int,
    *,
    on_daily_pause: Callable[[], None] | None = None,
    quota_bucket: str | None = None,
) -> RequestReservation:
    """Atomically enforce and persist RPM, rolling TPM, and Pacific-day RPD."""
    limits = require_selected_model(model)
    if estimated_tokens < 0:
        raise ValueError("Estimated token count cannot be negative.")
    if estimated_tokens > limits.tpm:
        raise ValueError(
            f"Request estimate ({estimated_tokens} tokens) exceeds {model} TPM "
            f"limit ({limits.tpm})."
        )
    budget_path = _budget_path()
    state_key = _model_state_key(model, quota_bucket)

    while True:
        wait_seconds = 0.0
        pause_for_daily_reset = False
        with _state_lock():
            now = time.time()
            state = _read_state(budget_path, _pacific_date())
            model_state = state["models"].setdefault(
                state_key, {"attempts_today": 0, "recent": []}
            )
            _prune_recent(model_state, now)
            if int(model_state["attempts_today"]) >= limits.rpd:
                wait_seconds = _seconds_until_pacific_reset()
                pause_for_daily_reset = True
            recent = model_state["recent"]
            recent_tokens = sum(int(request["tokens"]) for request in recent)
            if not pause_for_daily_reset and len(recent) >= limits.rpm:
                wait_seconds = max(
                    wait_seconds,
                    60.0 - (now - float(recent[0]["time"])),
                )
            if not pause_for_daily_reset and recent_tokens + estimated_tokens > limits.tpm:
                tokens_after_expiry = recent_tokens
                for request in recent:
                    tokens_after_expiry -= int(request["tokens"])
                    if tokens_after_expiry + estimated_tokens <= limits.tpm:
                        wait_seconds = max(
                            wait_seconds,
                            60.0 - (now - float(request["time"])),
                        )
                        break
            if wait_seconds <= 0:
                reservation_id = uuid.uuid4().hex
                model_state["attempts_today"] = (
                    int(model_state["attempts_today"]) + 1
                )
                model_state["recent"].append(
                    {
                        "id": reservation_id,
                        "time": now,
                        "tokens": estimated_tokens,
                    }
                )
                _write_state(budget_path, state)
                return RequestReservation(
                    model=model,
                    reservation_id=reservation_id,
                    reserved_tokens=estimated_tokens,
                    budget_path=budget_path,
                    quota_bucket=quota_bucket,
                )
        if pause_for_daily_reset:
            if on_daily_pause is not None:
                on_daily_pause()
            wait_until_pacific_reset(
                model,
                reason="persisted daily request cap reached",
                minimum_wait=wait_seconds,
            )
        else:
            time.sleep(max(0.01, wait_seconds))


def request_token_estimate(
    prompt: str,
    schema: dict[str, Any] | None,
    *,
    output_reserve: int = 512,
) -> int:
    schema_chars = len(json.dumps(schema, ensure_ascii=False)) if schema else 0
    return max(1, (len(prompt) + schema_chars + 3) // 4 + output_reserve)


def refund_logged_transport_attempts(model: str, attempt_ids: list[str]) -> int:
    """Idempotently refund legacy logged transport failures from today's RPD count."""
    if not attempt_ids:
        return 0
    with _state_lock():
        state = _read_state(_budget_path(), _pacific_date())
        model_state = state["models"].setdefault(
            model, {"attempts_today": 0, "recent": []}
        )
        refunded = set(model_state.get("refunded_attempt_ids", []))
        new_ids = [attempt_id for attempt_id in attempt_ids if attempt_id not in refunded]
        refundable = min(len(new_ids), int(model_state["attempts_today"]))
        model_state["attempts_today"] = int(model_state["attempts_today"]) - refundable
        model_state["refunded_attempt_ids"] = sorted(refunded.union(new_ids))
        _write_state(_budget_path(), state)
        return refundable


def reassign_legacy_attempts(
    model: str,
    *,
    count: int,
    quota_bucket: str,
    migration_id: str,
) -> int:
    """Move known post-keying requests from the legacy model-only bucket once."""
    if count < 0:
        raise ValueError("Attempt count cannot be negative.")
    if not migration_id:
        raise ValueError("A migration ID is required.")
    with _state_lock():
        path = _budget_path()
        state = _read_state(path, _pacific_date())
        migrations = state.setdefault("quota_bucket_migrations", [])
        if migration_id in migrations:
            return 0
        models = state["models"]
        source = models.setdefault(model, {"attempts_today": 0, "recent": []})
        target_key = _model_state_key(model, quota_bucket)
        target = models.setdefault(
            target_key, {"attempts_today": 0, "recent": []}
        )
        moved = min(count, int(source["attempts_today"]))
        source["attempts_today"] = int(source["attempts_today"]) - moved
        target["attempts_today"] = int(target["attempts_today"]) + moved
        recent = source.get("recent", [])
        move_recent = min(moved, len(recent))
        if move_recent:
            target.setdefault("recent", []).extend(recent[-move_recent:])
            source["recent"] = recent[:-move_recent]
        migrations.append(migration_id)
        _write_state(path, state)
        return moved


def daily_attempt_count(model: str, *, quota_bucket: str | None = None) -> int:
    with _state_lock():
        state = _read_state(_budget_path(), _pacific_date())
        state_key = _model_state_key(model, quota_bucket)
        return int(state["models"].get(state_key, {}).get("attempts_today", 0))


__all__ = [
    "MODEL_LIMITS",
    "ModelLimits",
    "daily_attempt_count",
    "require_selected_model",
    "request_token_estimate",
    "reserve_request",
    "wait_until_pacific_reset",
]
