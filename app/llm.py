"""SQL generation with the Anthropic API: the prompt, the structured Draft, and a protocol so tests can use a fake."""
from typing import Protocol

import anthropic
from pydantic import BaseModel

from app.config import settings
from app.memory import Item

MAX_TOKENS = 16000

# Deliberately identical in schema and memory modes: only the user message differs, so the comparison is fair.
SYSTEM_PROMPT = """You translate a business question into one SQLite query.

- Use the SQLite dialect.
- Use only the tables and columns that appear in the schema. Do not invent names.
- Business rules may be provided. Apply a rule when it is relevant to the question, and list the ids of the rules you applied in items_used. Do not list rules you did not use.
- Return exactly one SELECT statement (a WITH clause is allowed). No explanations inside the sql field.
- In assumptions, state briefly any interpretation you had to guess."""


class Draft(BaseModel):
    sql: str
    items_used: list[str]
    assumptions: str


class LLMUnavailable(RuntimeError):
    """No usable credentials: nothing else will work either, so a run should stop."""


class LLMTransient(RuntimeError):
    """The API could not be reached or timed out. Worth losing one question over, not the whole run."""


class LLMRefusal(RuntimeError):
    """The model declined to answer."""


class LLM(Protocol):
    def generate(self, question: str, schema_text: str, items: list[Item]) -> Draft: ...

    def repair(self, question: str, schema_text: str, items: list[Item], sql: str, error: str) -> Draft: ...


def _rules_text(items: list[Item]) -> str:
    if not items:
        return ""
    blocks = []
    for it in items:
        lines = [f"[{it.id}] ({it.kind}) {it.title}", it.body]
        if it.sql:
            lines.append(f"SQL: {it.sql}")
        blocks.append("\n".join(lines))
    return "Business rules:\n\n" + "\n\n".join(blocks) + "\n\n"


def build_user_text(question: str, schema_text: str, items: list[Item]) -> str:
    return f"Schema:\n{schema_text}\n\n{_rules_text(items)}Question: {question}"


def build_repair_text(question: str, schema_text: str, items: list[Item], sql: str, error: str) -> str:
    return (
        f"{build_user_text(question, schema_text, items)}\n\n"
        f"This query failed:\n{sql}\n\nError: {error}\n\nReturn a corrected query."
    )


class AnthropicLLM:
    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:  # lazy: importing or building the pipeline needs no credentials
            key = settings.anthropic_api_key
            # only pass a key we were given, so environment variables and `ant auth login` profiles still work
            self._client = anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()
        return self._client

    def _parse(self, user_text: str) -> Draft:
        try:
            response = self.client.messages.parse(
                model=settings.llm_model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_text}],
                output_config={"effort": settings.llm_effort},
                output_format=Draft,
            )
        except (anthropic.AuthenticationError, anthropic.CredentialsError) as e:
            raise LLMUnavailable(f"Anthropic API unavailable: {type(e).__name__}") from e
        except TypeError as e:
            # with no key and no profile the SDK raises a plain TypeError while building the request
            if "authentication method" not in str(e):
                raise
            raise LLMUnavailable("Anthropic API unavailable: no credentials found") from e
        except anthropic.APIConnectionError as e:  # includes APITimeoutError
            raise LLMTransient(f"Anthropic API not reachable: {type(e).__name__}") from e
        if response.stop_reason == "refusal":
            raise LLMRefusal("the model refused to answer this question")
        if response.parsed_output is None:
            raise LLMRefusal("the model returned no structured answer")
        return response.parsed_output

    def generate(self, question: str, schema_text: str, items: list[Item]) -> Draft:
        return self._parse(build_user_text(question, schema_text, items))

    def repair(self, question: str, schema_text: str, items: list[Item], sql: str, error: str) -> Draft:
        return self._parse(build_repair_text(question, schema_text, items, sql, error))
