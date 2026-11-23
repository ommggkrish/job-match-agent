"""LLM client. This is the ONLY file that talks to the Gemini API.

The rest of the code uses two methods:
- next_turn(): send the conversation + tool list, get back text and/or tool calls
- structured(): send a prompt, get back a validated Pydantic object (JSON mode)

Tests replace this class with a fake, so they run without an API key.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.5-flash"

MAX_RATE_LIMIT_WAIT = 120  # seconds; longer waits mean the daily quota is used up
MAX_RATE_LIMIT_RETRIES = 3


def retry_delay_seconds(exc) -> float | None:
    """Read Google's 'retryDelay': '37s' from a 429 error. None if it is not there."""
    match = re.search(r"retryDelay'?\"?\s*:\s*'?\"?(\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class ModelTurn:
    """One answer from the model: some text, and zero or more tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider object, sent back unchanged in the next request


class LLM(Protocol):
    def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn: ...

    def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T: ...


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        from google import genai  # imported here so tests never need the package configured

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        self.client = genai.Client(api_key=api_key)
        self.sleep = time.sleep  # replaced in tests

    # ------------------------------------------------------- rate-limit retry

    def _generate(self, contents, config):
        """generate_content with a wait-and-retry for the per-minute rate limit.

        The free tier limits requests per minute AND per day. Both return HTTP 429.
        A short "retry in N seconds" means the per-minute limit: wait and try again.
        A long one (hours) means the daily limit: give up and raise the error.
        """
        from google.genai.errors import ClientError

        for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.models.generate_content(model=self.model, contents=contents, config=config)
            except ClientError as exc:
                wait = retry_delay_seconds(exc)
                if exc.code != 429 or wait is None or wait > MAX_RATE_LIMIT_WAIT or attempt == MAX_RATE_LIMIT_RETRIES:
                    raise
                self.sleep(wait + 1)

    # ------------------------------------------------------------------ tools

    def next_turn(self, system: str, messages: list[dict], tools: list[dict]) -> ModelTurn:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(
                name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]
            )
            for t in tools
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            temperature=0,
        )
        response = self._generate(contents=self._to_contents(messages), config=config)
        content = response.candidates[0].content
        calls = [
            ToolCall(name=fc.name, args=dict(fc.args or {}), id=fc.id) for fc in (response.function_calls or [])
        ]
        text = "".join(p.text for p in (content.parts or []) if getattr(p, "text", None) and not getattr(p, "thought", False))
        return ModelTurn(text=text, tool_calls=calls, raw=content)

    @staticmethod
    def _to_contents(messages: list[dict]) -> list:
        """Convert our simple message dicts into Gemini `Content` objects.

        Tool results that follow each other are grouped into one user turn,
        which is what Gemini expects after a turn with several function calls.
        """
        from google.genai import types

        contents: list = []
        pending_results: list = []

        def flush() -> None:
            if pending_results:
                contents.append(types.Content(role="user", parts=list(pending_results)))
                pending_results.clear()

        for msg in messages:
            if msg["role"] == "tool":
                pending_results.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=msg.get("id"), name=msg["name"], response={"result": msg["result"]}
                        )
                    )
                )
                continue
            flush()
            if msg["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=msg["text"])]))
            elif msg.get("raw") is not None:
                contents.append(msg["raw"])  # keeps Gemini's thought signatures intact
            else:
                parts = [types.Part(text=msg["text"])] if msg.get("text") else []
                parts += [
                    types.Part(function_call=types.FunctionCall(id=c.id, name=c.name, args=c.args))
                    for c in msg.get("tool_calls", [])
                ]
                contents.append(types.Content(role="model", parts=parts))
        flush()
        return contents

    # ------------------------------------------------------------- JSON mode

    def structured(self, prompt: str, schema: type[T], system: str | None = None) -> T:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
        )
        response = self._generate(contents=prompt, config=config)
        return schema.model_validate_json(response.text)  # raises if the JSON does not fit the model
