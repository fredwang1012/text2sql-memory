import anthropic
import httpx2 as httpx  # the HTTP library the SDK itself uses
import pytest

from app import llm as llm_mod
from app.config import Settings, settings
from app.llm import AnthropicLLM, LLMTransient, LLMUnavailable

SECRET = "sk-ant-test-secret-value"


def test_settings_read_the_key_from_a_dotenv_and_never_show_it(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text(f"ANTHROPIC_API_KEY={SECRET}\n", encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.anthropic_api_key.get_secret_value() == SECRET
    assert SECRET not in repr(s) and SECRET not in str(s)
    assert Settings(_env_file=None).anthropic_api_key is None


def _capture_client(monkeypatch):
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", Client)
    return calls


def test_key_is_passed_to_the_client_only_when_set(monkeypatch):
    calls = _capture_client(monkeypatch)
    monkeypatch.setattr(settings, "anthropic_api_key", Settings(anthropic_api_key=SECRET).anthropic_api_key)
    AnthropicLLM().client
    assert calls == [{"api_key": SECRET}]
    calls.clear()
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    AnthropicLLM().client
    assert calls == [{}]  # the SDK's own credential lookup stays in charge


class _Failing:
    def __init__(self, exc):
        self.messages = self
        self.exc = exc

    def parse(self, **kwargs):
        raise self.exc


def _llm_raising(exc):
    obj = AnthropicLLM()
    obj._client = _Failing(exc)
    return obj


def _request():
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def test_timeout_and_connection_errors_are_transient():
    for exc in (anthropic.APITimeoutError(request=_request()), anthropic.APIConnectionError(request=_request())):
        with pytest.raises(LLMTransient):
            _llm_raising(exc).generate("q", "schema", [])


def test_authentication_errors_are_fatal():
    response = httpx.Response(401, request=_request())
    exc = anthropic.AuthenticationError("bad key", response=response, body=None)
    with pytest.raises(LLMUnavailable):
        _llm_raising(exc).generate("q", "schema", [])
