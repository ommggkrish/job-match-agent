"""A scripted stand-in for GeminiClient, so tests need no API key and no internet."""

from agent.llm import ModelTurn, ToolCall


class FakeLLM:
    def __init__(self, turns: list[ModelTurn], structured_answers: dict | None = None):
        self.turns = list(turns)
        self.structured_answers = structured_answers or {}  # schema name -> object to return
        self.next_turn_calls: list[list[dict]] = []
        self.structured_prompts: list[str] = []

    def next_turn(self, system, messages, tools):
        self.next_turn_calls.append(list(messages))
        if self.turns:
            return self.turns.pop(0)
        return ModelTurn(text="DONE")

    def structured(self, prompt, schema, system=None):
        self.structured_prompts.append(prompt)
        return self.structured_answers[schema.__name__]


def call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, id=f"id-{name}")
