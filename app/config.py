"""Settings read from the environment or the repo-root .env file; a relative DATA_DIR is resolved against the repo root."""
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]  # text2sql-memory/


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # optional: without it the SDK falls back to its own credential lookup (environment, `ant auth login` profile)
    anthropic_api_key: SecretStr | None = None
    data_dir: Path = ROOT / "data"
    # auto: multilingual-e5 embeddings when the "embed" extra is installed, else BM25 only. off: always BM25 only.
    embed: Literal["auto", "off"] = "auto"
    # anthropic: the Anthropic API. openai: any OpenAI-compatible endpoint at llm_base_url (Ollama, Gemini, OpenRouter)
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"  # Anthropic only
    llm_base_url: str = "http://localhost:11434/v1"  # openai provider only; this default is a local Ollama
    llm_api_key: SecretStr | None = None  # openai provider only; Ollama needs none
    db_id: str = "financial"
    memory_db: Path | None = None  # defaults to data_dir/memory.db

    @field_validator("data_dir", "memory_db")
    @classmethod
    def _anchor_to_root(cls, v: Path | None) -> Path | None:
        # the CLI may be started from any directory, so a relative path must not depend on the working directory
        return v if v is None or v.is_absolute() else ROOT / v

    @model_validator(mode="after")
    def _default_memory_db(self) -> "Settings":
        if self.memory_db is None:
            self.memory_db = self.data_dir / "memory.db"
        return self

    @property
    def bird_dir(self) -> Path:
        return self.data_dir / "bird"


settings = Settings()
