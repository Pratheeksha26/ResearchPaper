from __future__ import annotations

import hashlib
import json
import os
import random
import time
import csv
import tempfile
from collections.abc import Mapping
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel

from gapfinder.config import MODEL, RUNS_DIR, SEED, TEMPERATURE, get_secret
from gapfinder.llm.rate_limiter import (
    require_selected_model,
    request_token_estimate,
    reserve_request,
    wait_until_pacific_reset,
)


class LLMAPIError(RuntimeError):
    """Sanitized API failure with its HTTP status for callers and run reports."""

    def __init__(
        self,
        message: str,
        status_code: int | None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def credential_fingerprint(api_key: str) -> str:
    """Return a stable, non-reversible identifier for per-credential quota state."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:20]


class RequestBudget:
    """Count actual outbound API attempts and fail before exceeding a run cap."""

    def __init__(
        self,
        maximum: int | None,
        *,
        on_daily_pause: Callable[[], None] | None = None,
    ) -> None:
        if maximum is not None and maximum <= 0:
            raise ValueError("Maximum request count must be positive.")
        self.maximum = maximum
        self.attempts = 0
        self.on_daily_pause = on_daily_pause

    def check_available(self) -> None:
        if self.maximum is not None and self.attempts >= self.maximum:
            raise RuntimeError(
                f"Run request cap reached ({self.maximum}); refusing another API attempt."
            )

    def record_attempt(self) -> None:
        self.check_available()
        self.attempts += 1

    def refund_attempt(self) -> None:
        if self.attempts <= 0:
            raise RuntimeError("Cannot refund a request that was not reserved.")
        self.attempts -= 1


_MAX_API_ATTEMPTS = 5


def _run_dir(run_id: str) -> Path:
    base = Path(os.getenv("GAPFINDER_RUNS_DIR", str(RUNS_DIR)))
    target = base / run_id
    target.mkdir(parents=True, exist_ok=True)
    return target


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _is_daily_quota_error(exc: Exception) -> bool:
    body = getattr(exc, "body", None)
    message = getattr(exc, "message", None)
    details = " ".join(
        value
        for value in (
            str(exc),
            str(message or ""),
            json.dumps(body, ensure_ascii=True, default=str) if body is not None else "",
        )
        if value
    )
    normalized = re.sub(r"[^a-z0-9]", "", details.casefold())
    return any(
        marker in normalized
        for marker in (
            "requestsperday",
            "requestperday",
            "perdayperproject",
            "dailyquota",
            "dailyrequests",
        )
    )


def _is_pre_http_transport_error(exc: Exception) -> bool:
    """Identify transport failures with no HTTP response to charge against RPD."""
    if getattr(exc, "response", None) is not None:
        return False
    if getattr(exc, "status_code", getattr(exc, "code", None)) is not None:
        return False
    message = str(getattr(exc, "message", None) or exc).casefold()
    markers = (
        "getaddrinfo failed",
        "name or service not known",
        "temporary failure in name resolution",
        "connection reset",
        "connection aborted",
        "connection refused",
        "failed to establish a new connection",
        "network is unreachable",
        "nodename nor servname",
        "timed out",
        "timeout while connecting",
    )
    return any(marker in message for marker in markers)


def _append_usage(
    run_id: str,
    *,
    timestamp: str,
    model: str,
    paper_id: str,
    attempt: int,
    input_tokens: int | None,
    output_tokens: int | None,
    thinking_tokens: int | None,
    latency_ms: int,
    status_code: int | None,
    quota_counted: bool = True,
    credential_id: str = "",
) -> None:
    path = _run_dir(run_id) / "usage.csv"
    exists = path.is_file()
    if exists:
        with path.open(encoding="utf-8", newline="") as source:
            fieldnames = next(csv.reader(source), [])
        for field in ("quota_counted", "credential_id"):
            if field not in fieldnames:
                with path.open(encoding="utf-8", newline="") as source:
                    rows = list(csv.DictReader(source))
                fieldnames.append(field)
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=path.parent,
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    writer = csv.DictWriter(temporary, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                    temporary_path = Path(temporary.name)
                temporary_path.replace(path)
    else:
        fieldnames = [
            "timestamp",
            "model",
            "paper_id",
            "attempt",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
            "latency_ms",
            "status_code",
            "quota_counted",
            "credential_id",
        ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        row = {
                "timestamp": timestamp,
                "model": model,
                "paper_id": paper_id,
                "attempt": attempt,
                "input_tokens": input_tokens if input_tokens is not None else "",
                "output_tokens": output_tokens if output_tokens is not None else "",
                "thinking_tokens": (
                    thinking_tokens if thinking_tokens is not None else ""
                ),
                "latency_ms": latency_ms,
                "status_code": status_code if status_code is not None else "",
                "quota_counted": quota_counted,
                "credential_id": credential_id,
            }
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def _log_call(
    run_id: str,
    *,
    model: str,
    temperature: float,
    seed: int,
    prompt: str,
    response: str,
    fallback: bool = False,
    error: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    thinking_tokens: int | None = None,
    duration_ms: int = 0,
    timestamp: str | None = None,
    paper_id: str = "",
    attempt: int | None = None,
    status_code: int | None = None,
    request_payload: dict[str, Any] | None = None,
    quota_counted: bool = True,
    credential_id: str = "",
) -> None:
    log_path = _run_dir(run_id) / "llm_calls.jsonl"
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_timestamp": timestamp,
        "model": model,
        "paper_id": paper_id,
        "attempt": attempt,
        "status_code": status_code,
        "request_payload": request_payload,
        "quota_counted": quota_counted,
        "credential_id": credential_id,
        "temperature": temperature,
        "seed": seed,
        "prompt": prompt,
        "response": response,
        "fallback": fallback,
        "error": error,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "duration_ms": duration_ms,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fallback_response(
    prompt: str,
    *,
    model: str,
    temperature: float,
    seed: int,
    run_id: str,
    error: str | None = None,
    duration_ms: int = 0,
) -> str:
    response = json.dumps({"fallback": True, "error": error})
    _log_call(
        run_id,
        model=model,
        temperature=temperature,
        seed=seed,
        prompt=prompt,
        response=response,
        fallback=True,
        error=error,
        duration_ms=duration_ms,
    )
    return response


def generate(
    prompt: str,
    schema: Mapping[str, Any] | type[BaseModel] | None = None,
    *,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
    model: str | None = None,
    run_id: str = "manual",
    allow_fallback: bool = False,
    paper_id: str = "",
    thinking_level: str = "low",
    request_budget: RequestBudget | None = None,
    raise_on_rate_limit: bool = False,
    system_instruction: str | None = None,
) -> str:
    """Generate text through Gemini Interactions and log the complete call."""
    model_name = model or os.getenv("GAPFINDER_MODEL") or MODEL
    require_selected_model(model_name)
    started = time.monotonic()
    if os.getenv("GAPFINDER_LLM_MODE") == "stub":
        if not allow_fallback:
            raise RuntimeError(
                "Stub generation requires explicit allow_fallback=True."
            )
        return _fallback_response(
            prompt,
            model=model_name,
            temperature=temperature,
            seed=seed,
            run_id=run_id,
        )

    api_key = (
        get_secret("GEMINI_API_KEY2")
        or get_secret("GEMINI_API_KEY")
        or get_secret("GAPFINDER_API_KEY")
    )
    if not api_key:
        error = "Gemini API key is not configured."
        _log_call(
            run_id,
            model=model_name,
            temperature=temperature,
            seed=seed,
            prompt=prompt,
            response="",
            error=error,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        if allow_fallback:
            return _fallback_response(
                prompt,
                model=model_name,
                temperature=temperature,
                seed=seed,
                run_id=run_id,
                error=error,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        raise RuntimeError(error)
    credential_id = credential_fingerprint(api_key)

    try:
        from google import genai  # type: ignore[import-untyped]
        from google.genai import errors  # type: ignore[import-untyped]
        from google.genai import types  # type: ignore[import-untyped]
        from google.genai._gaos.lib.compat_errors import (  # type: ignore[import-untyped]
            APIError as InteractionsAPIError,
        )
    except ImportError as exc:
        error = "google-genai is required; install gapfinder[llm]."
        _log_call(
            run_id,
            model=model_name,
            temperature=temperature,
            seed=seed,
            prompt=prompt,
            response="",
            error=error,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        if allow_fallback:
            return _fallback_response(
                prompt,
                model=model_name,
                temperature=temperature,
                seed=seed,
                run_id=run_id,
                error=error,
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        raise RuntimeError(error) from exc

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
        }
        if model_name.startswith("gemini-3."):
            generation_config["thinking_level"] = thinking_level
        response_format: dict[str, Any] | None = None
        if schema is not None:
            json_schema = (
                schema.model_json_schema()
                if isinstance(schema, type) and issubclass(schema, BaseModel)
                else dict(schema)
            )
            response_format = {
                "type": "text",
                "mime_type": "application/json",
                "schema": json_schema,
            }
        request_payload = {
            "model": model_name,
            "system_instruction": system_instruction,
            "input": prompt,
            "generation_config": generation_config,
            "response_format": response_format,
        }
        attempt = 0
        while attempt < _MAX_API_ATTEMPTS:
            if request_budget is not None:
                request_budget.check_available()
            reservation = reserve_request(
                model_name,
                request_token_estimate(prompt, response_format),
                quota_bucket=credential_id,
                on_daily_pause=(
                    request_budget.on_daily_pause if request_budget else None
                ),
            )
            if request_budget is not None:
                request_budget.record_attempt()
            call_started = time.monotonic()
            request_timestamp = datetime.now(timezone.utc).isoformat()
            try:
                interaction = client.interactions.create(
                    model=model_name,
                    system_instruction=system_instruction,
                    input=prompt,
                    generation_config=generation_config,
                    response_format=response_format,
                )
                if not hasattr(interaction, "output_text") or not hasattr(
                    interaction, "usage"
                ):
                    raise RuntimeError(
                        "Gemini returned an unexpected streaming response."
                    )
                response_text = interaction.output_text or ""
                usage = interaction.usage
                input_tokens = getattr(usage, "total_input_tokens", None)
                output_tokens = getattr(usage, "total_output_tokens", None)
                thinking_tokens = getattr(usage, "total_thought_tokens", None)
                actual_tokens = (
                    sum(
                        int(token_count or 0)
                        for token_count in (
                            input_tokens,
                            output_tokens,
                            thinking_tokens,
                        )
                    )
                    if input_tokens is not None or output_tokens is not None
                    else None
                )
                reservation.settle(actual_tokens)
                latency_ms = round((time.monotonic() - call_started) * 1000)
                _log_call(
                    run_id,
                    model=model_name,
                    temperature=temperature,
                    seed=seed,
                    prompt=prompt,
                    response=response_text,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    thinking_tokens=thinking_tokens,
                    duration_ms=latency_ms,
                    timestamp=request_timestamp,
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    status_code=200,
                    request_payload=request_payload,
                    credential_id=credential_id,
                )
                _append_usage(
                    run_id,
                    timestamp=request_timestamp,
                    model=model_name,
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    thinking_tokens=thinking_tokens,
                    latency_ms=latency_ms,
                    status_code=200,
                    quota_counted=True,
                    credential_id=credential_id,
                )
                return response_text
            except Exception as exc:
                transport_error = _is_pre_http_transport_error(exc)
                if not isinstance(exc, (errors.APIError, InteractionsAPIError)) and not transport_error:
                    raise
                message = str(getattr(exc, "message", None) or exc)
                if api_key in message:
                    message = message.replace(api_key, "[REDACTED]")
                code = getattr(exc, "status_code", getattr(exc, "code", None))
                status = getattr(exc, "status", type(exc).__name__)
                retry_after = _retry_after_seconds(exc)
                error = (
                    f"Gemini API error HTTP {code or 'unknown'} ({status}): {message}"
                )
                latency_ms = round((time.monotonic() - call_started) * 1000)
                _log_call(
                    run_id,
                    model=model_name,
                    temperature=temperature,
                    seed=seed,
                    prompt=prompt,
                    response="",
                    error=error,
                    duration_ms=latency_ms,
                    timestamp=request_timestamp,
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    status_code=code if isinstance(code, int) else None,
                    request_payload=request_payload,
                    quota_counted=not transport_error,
                    credential_id=credential_id,
                )
                _append_usage(
                    run_id,
                    timestamp=request_timestamp,
                    model=model_name,
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    input_tokens=None,
                    output_tokens=None,
                    thinking_tokens=None,
                    latency_ms=latency_ms,
                    status_code=code if isinstance(code, int) else None,
                    quota_counted=not transport_error,
                    credential_id=credential_id,
                )
                if transport_error:
                    reservation.refund()
                    if request_budget is not None:
                        request_budget.refund_attempt()
                    if attempt + 1 == _MAX_API_ATTEMPTS:
                        raise LLMAPIError(
                            error,
                            None,
                        ) from exc
                    delay = min(
                        60.0,
                        2**attempt + random.uniform(0.0, 1.0),
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                if code not in (408, 429) and not (
                    isinstance(code, int) and code >= 500
                ):
                    retryable = False
                else:
                    retryable = True
                if not retryable:
                    if allow_fallback:
                        return _fallback_response(
                            prompt,
                            model=model_name,
                            temperature=temperature,
                            seed=seed,
                            run_id=run_id,
                            error=error,
                            duration_ms=latency_ms,
                        )
                    raise LLMAPIError(
                        error,
                        code if isinstance(code, int) else None,
                        retry_after,
                    ) from exc
                if code == 429 and _is_daily_quota_error(exc):
                    if raise_on_rate_limit:
                        raise LLMAPIError(
                            error,
                            code if isinstance(code, int) else None,
                            retry_after,
                        ) from exc
                    if request_budget and request_budget.on_daily_pause:
                        request_budget.on_daily_pause()
                    wait_until_pacific_reset(
                        model_name,
                        reason="Google API reported its daily request quota exhausted",
                        minimum_wait=retry_after or 0.0,
                    )
                    attempt = 0
                    continue
                if code == 429 and raise_on_rate_limit:
                    raise LLMAPIError(
                        error,
                        code if isinstance(code, int) else None,
                        retry_after,
                    ) from exc
                if attempt + 1 == _MAX_API_ATTEMPTS:
                    if allow_fallback:
                        return _fallback_response(
                            prompt,
                            model=model_name,
                            temperature=temperature,
                            seed=seed,
                            run_id=run_id,
                            error=error,
                            duration_ms=latency_ms,
                        )
                    raise LLMAPIError(
                        error,
                        code if isinstance(code, int) else None,
                        retry_after,
                    ) from exc
                delay = (
                    retry_after
                    if retry_after is not None
                    else min(60.0, 2**attempt + random.uniform(0.0, 1.0))
                )
                time.sleep(delay)
                attempt += 1
            except RuntimeError as exc:
                latency_ms = round((time.monotonic() - call_started) * 1000)
                _log_call(
                    run_id,
                    model=model_name,
                    temperature=temperature,
                    seed=seed,
                    prompt=prompt,
                    response="",
                    error=str(exc),
                    duration_ms=latency_ms,
                    timestamp=request_timestamp,
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    request_payload=request_payload,
                    credential_id=credential_id,
                )
                _append_usage(
                    run_id,
                    timestamp=request_timestamp,
                    model=model_name,
                    paper_id=paper_id,
                    attempt=attempt + 1,
                    input_tokens=None,
                    output_tokens=None,
                    thinking_tokens=None,
                    latency_ms=latency_ms,
                    status_code=None,
                    quota_counted=False,
                    credential_id=credential_id,
                )
                raise
        raise RuntimeError("Gemini retry loop finished unexpectedly.")
    except (errors.APIError, InteractionsAPIError) as exc:
        raise RuntimeError("Unreachable Gemini API error state.") from exc


__all__ = ["LLMAPIError", "RequestBudget", "generate"]
