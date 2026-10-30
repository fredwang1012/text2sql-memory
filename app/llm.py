"""SQL generation: the prompt, the structured Draft, a protocol so tests can use a fake, and two backends.

The Anthropic backend uses structured outputs. The OpenAI-compatible backend talks to any /chat/completions endpoint
in JSON mode (Ollama, Gemini, OpenRouter, Groq, vLLM and others) and validates the reply against the same model.
"""
import json
import re
import time
import urllib.error
import urllib.request
from typing import Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

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


T = TypeVar("T", bound=BaseModel)

DISTILL_PROMPT = """A user corrected the answer of a text-to-SQL system. Turn the correction into business rules that
will help with OTHER questions on the same database, not only this one.

- Write 0 to 3 rules. Write none when the correction holds nothing reusable.
- Each rule states one fact: what a column, code or term means (glossary), how a number is computed (metric, with the
  SQL expression in sql), which tables join on which keys (join_path), a default or rule of thumb (convention), or the
  exact stored spelling and meaning of values (value_hint).
- Use only table and column names that appear in the schema. Spell stored values exactly.
- terms are the words a future question would use for this fact (lowercase).
- Do not restate a rule that is already listed under existing rules.
- Do not describe this particular question or its answer; generalise."""

CORRECT_PROMPT = """You are a business analyst checking an answer from a text-to-SQL system. You know the correct query
and its result; the system does not. Tell the system, in plain business language and at most three sentences, what it
got wrong and what the right interpretation is: which data, definitions, filters, codes or values.

- Do not write SQL, do not quote the correct query, and do not give the final result values.
- Speak about meaning (for example "the region is stored in A3 of district", "only owners can issue cards"), so the
  advice also helps with other questions."""


class ProposedRule(BaseModel):
    kind: Literal["glossary", "metric", "join_path", "convention", "value_hint"]
    title: str
    terms: list[str]
    body: str
    sql: str | None = None


class Distilled(BaseModel):
    rules: list[ProposedRule]


class Correction(BaseModel):
    text: str


class LLMUnavailable(RuntimeError):
    """No usable credentials: nothing else will work either, so a run should stop."""


class LLMTransient(RuntimeError):
    """The API could not be reached or timed out. Worth losing one question over, not the whole run."""


class LLMRefusal(RuntimeError):
    """The model declined to answer."""


class LLM(Protocol):
    def generate(self, question: str, schema_text: str, items: list[Item]) -> Draft: ...

    def repair(self, question: str, schema_text: str, items: list[Item], sql: str, error: str) -> Draft: ...


class Learner(Protocol):
    """The extra calls of the learning loop: distilling a correction, and (in simulation only) writing one."""

    def distill(self, question: str, schema_text: str, sql: str, correction: str, items: list[Item]) -> list[ProposedRule]: ...

    def correct(self, question: str, schema_text: str, sql: str, rows: str, gold_sql: str, gold_rows: str) -> str: ...


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


def build_distill_text(question: str, schema_text: str, sql: str, correction: str, items: list[Item]) -> str:
    existing = "\n".join(f"- [{i.id}] ({i.kind}) {i.title}: {i.body}" for i in items) or "(none)"
    return (f"Schema:\n{schema_text}\n\nExisting rules:\n{existing}\n\nQuestion: {question}\n\n"
            f"The system's SQL:\n{sql or '(none)'}\n\nThe user's correction: {correction}")


def build_correct_text(question: str, schema_text: str, sql: str, rows: str, gold_sql: str, gold_rows: str) -> str:
    return (f"Schema:\n{schema_text}\n\nQuestion: {question}\n\nThe system's SQL:\n{sql or '(none)'}\n"
            f"Its result:\n{rows}\n\nThe correct SQL (do not reveal it):\n{gold_sql}\nThe correct result:\n{gold_rows}")


def build_repair_text(question: str, schema_text: str, items: list[Item], sql: str, error: str) -> str:
    return (
        f"{build_user_text(question, schema_text, items)}\n\n"
        f"This query failed:\n{sql}\n\nError: {error}\n\nReturn a corrected query."
    )


class _Backend:
    """generate and repair on top of one structured call; subclasses implement structured()."""

    def structured(self, system: str, user_text: str, output: type[T]) -> T:
        raise NotImplementedError

    def generate(self, question: str, schema_text: str, items: list[Item]) -> Draft:
        return self.structured(SYSTEM_PROMPT, build_user_text(question, schema_text, items), Draft)

    def repair(self, question: str, schema_text: str, items: list[Item], sql: str, error: str) -> Draft:
        return self.structured(SYSTEM_PROMPT, build_repair_text(question, schema_text, items, sql, error), Draft)

    def distill(self, question: str, schema_text: str, sql: str, correction: str, items: list[Item]) -> list[ProposedRule]:
        text = build_distill_text(question, schema_text, sql, correction, items)
        return self.structured(DISTILL_PROMPT, text, Distilled).rules

    def correct(self, question: str, schema_text: str, sql: str, rows: str, gold_sql: str, gold_rows: str) -> str:
        text = build_correct_text(question, schema_text, sql, rows, gold_sql, gold_rows)
        return self.structured(CORRECT_PROMPT, text, Correction).text


class AnthropicLLM(_Backend):
    def __init__(self):
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:  # lazy: importing or building the pipeline needs no credentials
            key = settings.anthropic_api_key
            # only pass a key we were given, so environment variables and `ant auth login` profiles still work
            self._client = anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()
        return self._client

    def structured(self, system: str, user_text: str, output: type[T]) -> T:
        try:
            response = self.client.messages.parse(
                model=settings.llm_model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user_text}],
                output_config={"effort": settings.llm_effort},
                output_format=output,
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


MAX_RETRIES = 3  # for rate limits and server errors, which free tiers hit often
MAX_WAIT_S = 60.0
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_JSON_OBJECT = re.compile(r"\{.*\}", re.S)


def json_instructions(output: type[BaseModel]) -> str:
    return ("\n\nAnswer with one JSON object and nothing else. It must match this JSON schema:\n"
            + json.dumps(output.model_json_schema()))


def _retry_after(headers) -> float | None:
    try:
        return float(headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


class OpenAICompatLLM(_Backend):
    """Any OpenAI-compatible /chat/completions endpoint, in JSON mode. A key is optional (Ollama needs none)."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 sleep=time.sleep):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.model = model or settings.llm_model
        key = settings.llm_api_key
        self.api_key = api_key if api_key is not None else (key.get_secret_value() if key else None)
        self.sleep = sleep

    def _send(self, payload: dict) -> dict:
        """POST the payload; returns the decoded reply. Raises urllib errors as they come."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, payload: dict) -> dict:
        for attempt in range(MAX_RETRIES + 1):
            try:
                return self._send(payload)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    raise LLMUnavailable(f"{self.base_url} refused the request: HTTP {e.code}") from e
                if e.code not in _RETRY_STATUS or attempt == MAX_RETRIES:
                    raise LLMTransient(f"{self.base_url} answered HTTP {e.code}") from e
                self.sleep(min(_retry_after(e.headers) or 2.0 * 2**attempt, MAX_WAIT_S))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                raise LLMTransient(f"{self.base_url} not reachable: {e}") from e
        raise AssertionError("unreachable")

    def structured(self, system: str, user_text: str, output: type[T]) -> T:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system + json_instructions(output)},
                         {"role": "user", "content": user_text}],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        reply = self._post(payload)
        try:
            choice = reply["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMRefusal("the model returned no answer") from e
        if choice.get("finish_reason") == "content_filter":
            raise LLMRefusal("the model refused to answer this question")
        match = _JSON_OBJECT.search(text)  # tolerate code fences or a stray sentence around the object
        try:
            return output.model_validate_json(match.group(0) if match else text)
        except ValidationError as e:
            raise LLMRefusal("the model returned no structured answer") from e


def make_llm() -> LLM:
    """The backend chosen by LLM_PROVIDER."""
    return OpenAICompatLLM() if settings.llm_provider == "openai" else AnthropicLLM()


def unavailable_hint() -> str:
    if settings.llm_provider == "openai":
        return (f"The model endpoint {settings.llm_base_url} refused the request. Check LLM_API_KEY, LLM_BASE_URL "
                "and LLM_MODEL.")
    return "The Anthropic API is not available. Set ANTHROPIC_API_KEY or run `ant auth login`."
