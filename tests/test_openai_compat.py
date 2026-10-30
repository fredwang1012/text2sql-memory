import io
import json
import urllib.error

import pytest

from app.config import Settings, settings
from app.llm import (AnthropicLLM, Draft, LLMRefusal, LLMTransient, LLMUnavailable, OpenAICompatLLM, make_llm,
                     unavailable_hint)


def _reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


class Scripted(OpenAICompatLLM):
    """Answers from a script of replies or exceptions instead of the network."""

    def __init__(self, *script, **kw):
        super().__init__(base_url="http://model.test/v1/", model="qwen", api_key=kw.pop("api_key", None),
                         sleep=lambda s: self.waits.append(s))
        self.script, self.payloads, self.waits = list(script), [], []

    def _send(self, payload):
        self.payloads.append(payload)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _http(code, retry_after=None):
    headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return urllib.error.HTTPError("http://model.test", code, "x", headers, io.BytesIO(b""))


GOOD = json.dumps({"sql": "SELECT 1", "items_used": ["r-1"], "assumptions": ""})


def test_request_shape_and_parsed_draft():
    llm = Scripted(_reply(GOOD))
    out = llm.generate("the question", "CREATE TABLE t (x)", [])
    assert out == Draft(sql="SELECT 1", items_used=["r-1"], assumptions="")
    p = llm.payloads[0]
    assert p["model"] == "qwen" and p["response_format"] == {"type": "json_object"} and p["temperature"] == 0
    assert "JSON schema" in p["messages"][0]["content"] and '"items_used"' in p["messages"][0]["content"]
    assert "the question" in p["messages"][1]["content"]
    assert llm.base_url == "http://model.test/v1"


def test_json_wrapped_in_a_code_fence_is_accepted():
    assert Scripted(_reply(f"```json\n{GOOD}\n```")).generate("q", "s", []).sql == "SELECT 1"


@pytest.mark.parametrize("reply", [_reply("not json"), _reply('{"sql": 1}'), {"choices": []}, _reply(None),
                                   _reply(GOOD, finish="content_filter")])
def test_unusable_replies_are_refusals(reply):
    with pytest.raises(LLMRefusal):
        Scripted(reply).generate("q", "s", [])


def test_rate_limits_are_retried_with_backoff():
    llm = Scripted(_http(429, retry_after=7), _http(503), _reply(GOOD))
    assert llm.generate("q", "s", []).sql == "SELECT 1"
    assert llm.waits == [7.0, 4.0]


def test_retries_give_up_as_transient():
    llm = Scripted(*[_http(429, retry_after=999)] * 4)
    with pytest.raises(LLMTransient):
        llm.generate("q", "s", [])
    assert llm.waits == [60.0] * 3


def test_auth_errors_are_fatal_and_other_errors_transient():
    with pytest.raises(LLMUnavailable):
        Scripted(_http(401)).generate("q", "s", [])
    with pytest.raises(LLMTransient):
        Scripted(_http(400)).generate("q", "s", [])
    with pytest.raises(LLMTransient):
        Scripted(urllib.error.URLError("refused")).generate("q", "s", [])


def test_key_is_sent_as_a_bearer_token_only_when_set(monkeypatch):
    sent = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def fake_urlopen(req, timeout):
        sent.append(req)
        return Resp(json.dumps(_reply(GOOD)).encode())

    monkeypatch.setattr("app.llm.urllib.request.urlopen", fake_urlopen)
    OpenAICompatLLM(base_url="http://m/v1", model="m", api_key="k-123").generate("q", "s", [])
    OpenAICompatLLM(base_url="http://m/v1", model="m", api_key="").generate("q", "s", [])
    assert sent[0].full_url == "http://m/v1/chat/completions"
    assert sent[0].get_header("Authorization") == "Bearer k-123"
    assert sent[1].get_header("Authorization") is None


def test_provider_setting_picks_the_backend(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    assert isinstance(make_llm(), OpenAICompatLLM)
    assert "LLM_BASE_URL" in unavailable_hint()
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    assert isinstance(make_llm(), AnthropicLLM)
    assert "ANTHROPIC_API_KEY" in unavailable_hint()


def test_settings_read_provider_and_hide_the_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=openai\nLLM_BASE_URL=https://openrouter.ai/api/v1\nLLM_API_KEY=or-secret\n",
                   encoding="utf-8")
    s = Settings(_env_file=env)
    assert s.llm_provider == "openai" and s.llm_base_url == "https://openrouter.ai/api/v1"
    assert s.llm_api_key.get_secret_value() == "or-secret" and "or-secret" not in repr(s)


def test_distill_and_correct_use_their_own_prompts():
    from app.llm import CORRECT_PROMPT, DISTILL_PROMPT

    rules = {"rules": [{"kind": "glossary", "title": "Salary", "terms": ["salary"], "body": "A11 is the average salary"}]}
    llm = Scripted(_reply(json.dumps(rules)), _reply(json.dumps({"text": "Use A11 for salary."})))
    (rule,) = llm.distill("q", "schema", "SELECT 1", "salary is A11", [])
    assert rule.title == "Salary" and rule.sql is None
    assert llm.correct("q", "schema", "SELECT 1", "1", "SELECT A11 FROM district", "9000") == "Use A11 for salary."
    assert llm.payloads[0]["messages"][0]["content"].startswith(DISTILL_PROMPT)
    assert "Existing rules:\n(none)" in llm.payloads[0]["messages"][1]["content"]
    assert llm.payloads[1]["messages"][0]["content"].startswith(CORRECT_PROMPT)
    assert "do not reveal it" in llm.payloads[1]["messages"][1]["content"]
