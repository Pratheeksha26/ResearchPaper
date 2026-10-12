from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gapfinder.data.prisma import make_prisma_counts
from gapfinder.llm.client import credential_fingerprint, generate


def _keyed_budget_count(
    budget: dict, model: str, api_key: str
) -> int:
    state_key = f"{model}::key-{credential_fingerprint(api_key)}"
    return budget["models"][state_key]["attempts_today"]


def test_filters_remove_missing_abstract_and_old_or_non_english_entries() -> None:
    records = [
        {"title": "Valid English Paper", "year": 2024, "abstract": "This paper presents a robust benchmark using large-scale evidence retrieval."},
        {"title": "Missing Abstract", "year": 2024, "abstract": ""},
        {"title": "Very old paper", "year": 2014, "abstract": "This is older than the minimum allowed range."},
        {"title": "中文论文", "year": 2024, "abstract": "这篇文章研究图论算法和科学发现。"},
    ]
    counts = make_prisma_counts(records, min_year=2015)
    assert counts == {
        "identified": 4,
        "deduped": 4,
        "abstract_missing_removed": 1,
        "after_abstract_missing_removal": 3,
        "language_date_filtered": 2,
        "after_language_date_filtering": 1,
        "included": 1,
    }


def test_generate_raises_without_allow_fallback(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAPFINDER_LLM_MODE", "")
    monkeypatch.setenv("GAPFINDER_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY2", "")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    try:
        with pytest.raises(RuntimeError, match="API key"):
            generate("hello", run_id="test-run")
    finally:
        monkeypatch.undo()


def test_generate_records_fallback_with_flag(tmp_path: Path) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAPFINDER_LLM_MODE", "stub")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    try:
        response = generate("hello", run_id="fallback-test", allow_fallback=True)
        assert json.loads(response)["fallback"] is True
        log_path = tmp_path / "fallback-test" / "llm_calls.jsonl"
        assert log_path.exists(), "expected llm log to be written"
        record = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert record["fallback"] is True
    finally:
        monkeypatch.undo()


def test_generate_reads_interactions_text_and_usage_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genai = pytest.importorskip("google.genai")
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key2")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )

    class FakeInteractions:
        def create(self, **kwargs):
            assert kwargs["generation_config"]["temperature"] == 0.0
            assert kwargs["response_format"]["mime_type"] == "application/json"
            assert kwargs["system_instruction"] == "Use the JSON contract."
            return SimpleNamespace(
                output_text='{"result":"ok"}',
                usage=SimpleNamespace(
                    total_input_tokens=12,
                    total_output_tokens=4,
                    total_thought_tokens=2,
                ),
            )

    class FakeClient:
        interactions = FakeInteractions()

        def __init__(self, *, api_key: str, http_options):
            assert api_key == "test-key2"
            assert http_options.retry_options.attempts == 1

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert generate(
        "hello",
        schema={"type": "object"},
        run_id="sdk-v2-test",
        paper_id="W123",
        system_instruction="Use the JSON contract.",
    ) == '{"result":"ok"}'
    log_path = tmp_path / "sdk-v2-test" / "llm_calls.jsonl"
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["input_tokens"] == 12
    assert record["output_tokens"] == 4
    assert record["thinking_tokens"] == 2
    assert record["request_payload"]["response_format"]["schema"] == {
        "type": "object"
    }
    assert record["request_payload"]["system_instruction"] == "Use the JSON contract."
    with (tmp_path / "sdk-v2-test" / "usage.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        usage = next(csv.DictReader(source))
    assert usage == {
        "timestamp": usage["timestamp"],
        "model": "gemini-3.8-flash",
        "paper_id": "W123",
        "attempt": "1",
        "input_tokens": "12",
        "output_tokens": "4",
        "thinking_tokens": "2",
        "latency_ms": usage["latency_ms"],
        "status_code": "200",
        "quota_counted": "True",
        "credential_id": credential_fingerprint("test-key2"),
    }


def test_generate_records_interactions_api_error_without_leaking_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    from google.genai._gaos.lib.compat_errors import RateLimitError

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )
    monkeypatch.setattr("gapfinder.llm.client.time.sleep", lambda _seconds: None)
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(429, request=request)
    api_error = RateLimitError(
        "Rate limit exceeded; credential test-key is not relevant",
        response=response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    class FakeInteractions:
        def create(self, **_kwargs):
            raise api_error

    class FakeClient:
        interactions = FakeInteractions()

        def __init__(self, *, api_key: str, http_options):
            assert api_key == "test-key"
            assert http_options.retry_options.attempts == 1

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        generate("hello", run_id="sdk-v2-error")
    log_path = tmp_path / "sdk-v2-error" / "llm_calls.jsonl"
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 5
    assert all(record["error"] and "test-key" not in record["error"] for record in records)
    usage_path = tmp_path / "sdk-v2-error" / "usage.csv"
    assert len(usage_path.read_text(encoding="utf-8").splitlines()) == 6
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert _keyed_budget_count(budget, "gemini-3.8-flash", "test-key") == 5


def test_retry_after_header_is_parsed() -> None:
    httpx = pytest.importorskip("httpx")
    from gapfinder.llm.client import _retry_after_seconds

    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(
        429,
        request=request,
        headers={"Retry-After": "7"},
    )
    api_error = RuntimeError("limited")
    api_error.response = response
    assert _retry_after_seconds(api_error) == 7.0


def test_daily_quota_error_is_detected_from_google_quota_details() -> None:
    from gapfinder.llm.client import _is_daily_quota_error

    daily = RuntimeError(
        "RESOURCE_EXHAUSTED",
    )
    daily.body = {
        "error": {
            "details": [
                {
                    "quotaMetric": (
                        "generativelanguage.googleapis.com/"
                        "generate_requests_per_day_per_project_per_model"
                    )
                }
            ]
        }
    }
    rpm = RuntimeError("RESOURCE_EXHAUSTED")
    rpm.body = {
        "error": {
            "details": [{"quotaMetric": "generate_requests_per_minute"}]
        }
    }

    assert _is_daily_quota_error(daily)
    assert not _is_daily_quota_error(rpm)


def test_daily_quota_probe_raises_without_waiting_for_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from google.genai._gaos.lib.compat_errors import RateLimitError

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key2")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    sleeps: list[float] = []
    monkeypatch.setattr("gapfinder.llm.client.time.sleep", sleeps.append)
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(429, request=request)
    error = RateLimitError(
        "Daily project request quota exhausted",
        response=response,
        body={
            "error": {
                "message": "Daily quota exceeded",
                "details": [
                    {"quotaMetric": "generate_requests_per_day_per_project"}
                ],
            }
        },
    )

    class FakeInteractions:
        def create(self, **_kwargs):
            raise error

    class FakeClient:
        interactions = FakeInteractions()

        def __init__(self, *, api_key: str, http_options):
            assert api_key == "test-key2"

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(RuntimeError, match="HTTP 429"):
        generate(
            "Reply with OK only.",
            run_id="daily-quota-probe-test",
            raise_on_rate_limit=True,
        )
    assert sleeps == []


def test_generate_retries_after_retry_after_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    from google.genai._gaos.lib.compat_errors import RateLimitError

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )
    sleeps: list[float] = []
    monkeypatch.setattr(
        "gapfinder.llm.client.time.sleep", lambda seconds: sleeps.append(seconds)
    )
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(
        429,
        request=request,
        headers={"Retry-After": "7"},
    )
    api_error = RateLimitError(
        "Rate limit exceeded",
        response=response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    class FakeInteractions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise api_error
            return SimpleNamespace(
                output_text='{"result":"ok"}',
                usage=SimpleNamespace(
                    total_input_tokens=1,
                    total_output_tokens=1,
                    total_thought_tokens=0,
                ),
            )

    class FakeClient:
        interactions = FakeInteractions()

        def __init__(self, *, api_key: str, http_options):
            assert api_key == "test-key"
            assert http_options.retry_options.attempts == 1

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert generate("hello", run_id="retry-after-test") == '{"result":"ok"}'
    assert 7.0 in sleeps
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert _keyed_budget_count(budget, "gemini-3.8-flash", "test-key") == 2


def test_generate_retries_service_unavailable_503(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from google.genai._gaos.lib.compat_errors import InternalServerError

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setattr("gapfinder.llm.client.time.sleep", lambda _seconds: None)
    request = httpx.Request("POST", "https://example.test")
    response = httpx.Response(503, request=request)
    api_error = InternalServerError(
        "Service unavailable",
        response=response,
        body={"error": {"message": "Service unavailable"}},
    )

    class FakeInteractions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise api_error
            return SimpleNamespace(
                output_text='{"result":"ok"}',
                usage=SimpleNamespace(
                    total_input_tokens=1,
                    total_output_tokens=1,
                    total_thought_tokens=0,
                ),
            )

    fake_interactions = FakeInteractions()

    class FakeClient:
        interactions = fake_interactions

        def __init__(self, *, api_key: str, http_options):
            assert api_key == "test-key"
            assert http_options.retry_options.attempts == 1

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert generate("hello", run_id="retry-503-test") == '{"result":"ok"}'
    assert fake_interactions.calls == 2
    budget = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert _keyed_budget_count(budget, "gemini-3.8-flash", "test-key") == 2


def test_generate_refuses_model_other_than_configured_without_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.5-flash-lite")
    with pytest.raises(ValueError, match="blocked"):
        generate("hello", model="gemini-3.1-flash-lite")


def test_generate_requires_explicit_gapfinder_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GAPFINDER_MODEL", raising=False)
    with pytest.raises(ValueError, match="blocked"):
        generate("hello", model="gemini-3.8-flash")


def test_model_quota_configuration_matches_user_limits() -> None:
    from gapfinder.llm.rate_limiter import MODEL_LIMITS

    assert MODEL_LIMITS["gemini-3.8-flash"] == type(
        MODEL_LIMITS["gemini-3.8-flash"]
    )(rpm=5, tpm=250_000, rpd=20)
    assert MODEL_LIMITS["gemini-3.1-flash-lite"].rpm == 15
    assert MODEL_LIMITS["gemini-3.1-flash-lite"].tpm == 250_000
    assert MODEL_LIMITS["gemini-3.1-flash-lite"].rpd == 500
    assert MODEL_LIMITS["gemma-4-31b-it"].rpm == 30
    assert MODEL_LIMITS["gemma-4-31b-it"].tpm == 16_000
    assert MODEL_LIMITS["gemma-4-31b-it"].rpd == 14_400


def test_daily_attempt_budget_counts_every_reserved_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )
    monkeypatch.setattr(
        rate_limiter,
        "MODEL_LIMITS",
        {"gemini-3.8-flash": rate_limiter.ModelLimits(rpm=5, tpm=100, rpd=2)},
    )
    monkeypatch.setattr(rate_limiter.time, "time", lambda: 1_800_000_000.0)
    monkeypatch.setattr(rate_limiter.time, "sleep", lambda _seconds: None)

    first = rate_limiter.reserve_request("gemini-3.8-flash", 40)
    first.settle(35)
    rate_limiter.reserve_request("gemini-3.8-flash", 40)
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 2


def test_daily_budget_separates_credential_buckets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setattr(
        rate_limiter,
        "MODEL_LIMITS",
        {"gemini-3.8-flash": rate_limiter.ModelLimits(rpm=5, tpm=100, rpd=10)},
    )
    rate_limiter.reserve_request(
        "gemini-3.8-flash", 10, quota_bucket="credential-a"
    )
    rate_limiter.reserve_request(
        "gemini-3.8-flash", 10, quota_bucket="credential-b"
    )
    assert rate_limiter.daily_attempt_count(
        "gemini-3.8-flash", quota_bucket="credential-a"
    ) == 1
    assert rate_limiter.daily_attempt_count(
        "gemini-3.8-flash", quota_bucket="credential-b"
    ) == 1
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 0


def test_legacy_attempt_reassignment_is_exact_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setattr(rate_limiter, "_pacific_date", lambda: "2026-09-29")
    path = tmp_path / "budget.json"
    path.write_text(
        json.dumps(
            {
                "pacific_date": "2026-09-29",
                "models": {
                    "gemini-3.8-flash": {
                        "attempts_today": 40,
                        "recent": [{"id": str(i), "time": i, "tokens": 2} for i in range(4)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    moved = rate_limiter.reassign_legacy_attempts(
        "gemini-3.8-flash",
        count=12,
        quota_bucket="credential-b",
        migration_id="migration-test",
    )
    assert moved == 12
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 28
    assert rate_limiter.daily_attempt_count(
        "gemini-3.8-flash", quota_bucket="credential-b"
    ) == 12
    assert rate_limiter.reassign_legacy_attempts(
        "gemini-3.8-flash",
        count=12,
        quota_bucket="credential-b",
        migration_id="migration-test",
    ) == 0


def test_transport_reservation_and_run_attempt_can_be_refunded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter
    from gapfinder.llm.client import RequestBudget

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setattr(
        rate_limiter,
        "MODEL_LIMITS",
        {"gemini-3.8-flash": rate_limiter.ModelLimits(rpm=5, tpm=100, rpd=2)},
    )
    reservation = rate_limiter.reserve_request("gemini-3.8-flash", 20)
    budget = RequestBudget(1)
    budget.record_attempt()
    reservation.refund()
    budget.refund_attempt()
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 0
    assert budget.attempts == 0


def test_legacy_transport_log_refund_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setattr(rate_limiter, "_pacific_date", lambda: "2026-09-29")
    rate_limiter.reserve_request("gemini-3.8-flash", 10)
    ids = ["legacy-error-1", "legacy-error-2"]
    assert rate_limiter.refund_logged_transport_attempts(
        "gemini-3.8-flash", ids
    ) == 1
    assert rate_limiter.refund_logged_transport_attempts(
        "gemini-3.8-flash", ids
    ) == 0
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 0


def test_transport_failures_retry_without_consuming_daily_or_run_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from gapfinder.llm.client import RequestBudget
    from gapfinder.llm.rate_limiter import daily_attempt_count

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY2", "test-key")
    monkeypatch.setenv("GAPFINDER_RUNS_DIR", str(tmp_path))
    monkeypatch.setenv("GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json"))
    monkeypatch.setattr("gapfinder.llm.client.time.sleep", lambda _seconds: None)
    request = httpx.Request("POST", "https://example.test")

    class FakeInteractions:
        calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("getaddrinfo failed", request=request)
            return SimpleNamespace(
                output_text='{"result":"ok"}',
                usage=SimpleNamespace(
                    total_input_tokens=1,
                    total_output_tokens=1,
                    total_thought_tokens=0,
                ),
            )

    interactions = FakeInteractions()

    class FakeClient:
        def __init__(self, *, api_key: str, http_options):
            assert api_key == "test-key"
            assert http_options.retry_options.attempts == 1

        @property
        def interactions(self):
            return interactions

    monkeypatch.setattr(genai, "Client", FakeClient)
    budget = RequestBudget(1)
    assert generate("hello", run_id="transport-retry", request_budget=budget) == (
        '{"result":"ok"}'
    )
    assert interactions.calls == 2
    assert budget.attempts == 1
    assert daily_attempt_count(
        "gemini-3.8-flash",
        quota_bucket=credential_fingerprint("test-key"),
    ) == 1
    records = [
        json.loads(line)
        for line in (
            tmp_path / "transport-retry" / "llm_calls.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["quota_counted"] is False
    assert records[1]["quota_counted"] is True


def test_tpm_limiter_waits_until_enough_tokens_expire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )
    monkeypatch.setattr(
        rate_limiter,
        "MODEL_LIMITS",
        {"gemini-3.8-flash": rate_limiter.ModelLimits(rpm=5, tpm=100, rpd=10)},
    )
    now = [1_800_000_000.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(rate_limiter.time, "time", lambda: now[0])
    monkeypatch.setattr(rate_limiter.time, "sleep", sleep)
    rate_limiter.reserve_request("gemini-3.8-flash", 60)
    rate_limiter.reserve_request("gemini-3.8-flash", 40)
    rate_limiter.reserve_request("gemini-3.8-flash", 60)

    assert sleeps == [60.0]
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 3


def test_rpm_limiter_waits_for_the_rolling_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )
    monkeypatch.setattr(
        rate_limiter,
        "MODEL_LIMITS",
        {"gemini-3.8-flash": rate_limiter.ModelLimits(rpm=2, tpm=1000, rpd=10)},
    )
    now = [1_800_000_000.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(rate_limiter.time, "time", lambda: now[0])
    monkeypatch.setattr(rate_limiter.time, "sleep", sleep)
    rate_limiter.reserve_request("gemini-3.8-flash", 10)
    rate_limiter.reserve_request("gemini-3.8-flash", 10)
    rate_limiter.reserve_request("gemini-3.8-flash", 10)

    assert sleeps == [60.0]
    assert rate_limiter.daily_attempt_count("gemini-3.8-flash") == 3


def test_daily_budget_resets_on_new_pacific_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm import rate_limiter

    monkeypatch.setenv("GAPFINDER_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv(
        "GAPFINDER_LLM_BUDGET_PATH", str(tmp_path / "budget.json")
    )
    monkeypatch.setattr(
        rate_limiter,
        "MODEL_LIMITS",
        {"gemini-3.8-flash": rate_limiter.ModelLimits(rpm=5, tpm=100, rpd=1)},
    )
    day = ["2026-09-28"]
    monkeypatch.setattr(rate_limiter, "_pacific_date", lambda: day[0])
    rate_limiter.reserve_request("gemini-3.8-flash", 1)
    day[0] = "2026-09-29"
    rate_limiter.reserve_request("gemini-3.8-flash", 1)

    state = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))
    assert state["pacific_date"] == "2026-09-29"
    assert state["models"]["gemini-3.8-flash"]["attempts_today"] == 1


def test_corrupt_daily_budget_state_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gapfinder.llm.rate_limiter import _pacific_date, _read_state

    path = tmp_path / "budget.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="budget file is invalid"):
        _read_state(path, _pacific_date())


def test_request_budget_blocks_attempt_after_hard_cap() -> None:
    from gapfinder.llm.client import RequestBudget

    budget = RequestBudget(2)
    budget.record_attempt()
    budget.record_attempt()
    with pytest.raises(RuntimeError, match="Run request cap reached"):
        budget.record_attempt()
    assert budget.attempts == 2
