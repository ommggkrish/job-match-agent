"""Offline tests for the Gemini client: message conversion only, no API call."""

import pytest

from agent.llm import GeminiClient, ToolCall


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


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()
