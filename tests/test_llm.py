"""Offline tests for the Gemini client: message conversion and retry rules, no API call."""

import pytest

from agent.llm import GeminiClient, ToolCall, is_daily_limit, retry_delay_seconds


def test_tool_results_are_grouped_into_one_user_turn():
    messages = [
        {"role": "user", "text": "job ad"},
        {"role": "model", "text": "", "tool_calls": [ToolCall("read_cv", {}, "1"), ToolCall("compare_skills", {"required_skills": ["Python"]}, "2")]},
        {"role": "tool", "name": "read_cv", "id": "1", "result": {"skills": ["Python"]}},
        {"role": "tool", "name": "compare_skills", "id": "2", "result": {"matching": ["Python"]}},
    ]
    contents = GeminiClient._to_contents(messages)

    assert [c.role for c in contents] == ["user", "model", "user"]
    assert len(contents[1].parts) == 2
    responses = [p.function_response for p in contents[2].parts]
    assert [r.name for r in responses] == ["read_cv", "compare_skills"]
    assert responses[0].response == {"result": {"skills": ["Python"]}}


PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def make_429(retry_delay: str | None, quota_id: str = PER_MINUTE):
    """A 429 error shaped like the real ones from the Gemini API."""
    from google.genai.errors import ClientError

    details = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"quotaId": quota_id}]}]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    body = {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED", "details": details}}
    return ClientError(429, body)


def make_503():
    from google.genai.errors import ServerError

    return ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


class FlakyModels:
    """Fails with the given errors first, then answers 'ok'."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def make_client(errors):
    client = GeminiClient.__new__(GeminiClient)  # skip __init__: no API key, no network
    client.model = "test-model"
    client.client = type("C", (), {"models": FlakyModels(errors)})()
    client.waits = []
    client.sleep = client.waits.append
    return client


def test_retry_delay_is_read_from_error():
    assert retry_delay_seconds(make_429("37s")) == 37
    assert retry_delay_seconds(make_429("79590s")) == 79590
    assert retry_delay_seconds(RuntimeError("no delay here")) is None


def test_per_minute_limit_waits_and_retries():
    client = make_client([make_429("20s"), make_429("5s")])
    assert client._generate(contents="hi", config=None) == "ok"
    assert client.waits == [21, 6]


def test_per_minute_limit_without_retry_delay_waits_one_minute():
    client = make_client([make_429(None)])
    assert client._generate(contents="hi", config=None) == "ok"
    assert client.waits == [61]


def test_overloaded_model_is_retried():
    client = make_client([make_503()])
    assert client._generate(contents="hi", config=None) == "ok"
    assert client.waits == [21]


def test_daily_limit_is_raised_immediately():
    error = make_429("79590s", quota_id=PER_DAY)
    assert is_daily_limit(error)
    assert not is_daily_limit(make_429("20s"))
    client = make_client([error])
    with pytest.raises(Exception, match="429"):
        client._generate(contents="hi", config=None)
    assert client.waits == []


def test_retries_are_limited():
    client = make_client([make_429("1s")] * 10)
    with pytest.raises(Exception, match="429"):
        client._generate(contents="hi", config=None)
    assert client.client.models.calls == 4  # 1 try + 3 retries


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()
