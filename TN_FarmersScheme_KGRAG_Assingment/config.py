"""Central configuration for the Tamil Nadu Agriculture Schemes RAG app."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
PLACEHOLDER_VALUES = {
    "",
    "replace_with_your_openai_api_key",
    "replace_with_your_langsmith_api_key",
    "your_openai_api_key_here",
    "your_langsmith_api_key_here",
    "replace_with_your_neo4j_uri",
    "replace_with_your_neo4j_password",
}


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _is_missing_secret(value: str | None) -> bool:
    return value is None or value.strip() in PLACEHOLDER_VALUES


@dataclass(frozen=True)
class AppConfig:
    """Typed application settings loaded from environment variables."""

    root_dir: Path
    openai_api_key: str
    openai_chat_model: str
    openai_embedding_model: str
    langsmith_tracing: bool
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str
    source_url: str
    schemes_landing_url: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    request_timeout: int
    request_retries: int
    request_delay_seconds: float
    data_directory: Path
    # Neo4j Knowledge Graph settings
    neo4j_uri: str = ""
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_index_name: str = "scheme_chunks"
    # Fields below have defaults so existing callers/tests keep working unchanged.
    openai_max_output_tokens: int = 700
    openai_timeout: int = 60
    max_input_chars: int = 1200
    max_history_messages: int = 50
    app_env: str = "development"
    admin_actions_enabled: bool = False
    admin_password: str = ""

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() in {"production", "prod"}

    @property
    def admin_password_configured(self) -> bool:
        return not _is_missing_secret(self.admin_password)

    @property
    def schemes_json_path(self) -> Path:
        return self.data_directory / "schemes.json"

    @property
    def schemes_csv_path(self) -> Path:
        return self.data_directory / "schemes.csv"

    @property
    def index_metadata_path(self) -> Path:
        return self.data_directory / "neo4j_metadata.json"

    @property
    def neo4j_configured(self) -> bool:
        return (
            not _is_missing_secret(self.neo4j_uri)
            and not _is_missing_secret(self.neo4j_password)
        )

    @property
    def tracing_status(self) -> str:
        if not self.langsmith_tracing:
            return "Disabled"
        if _is_missing_secret(self.langsmith_api_key):
            return "Enabled, API key missing"
        return "Enabled"


def load_config() -> AppConfig:
    """Load `.env`, create local directories, and return typed settings."""

    load_dotenv(ROOT_DIR / ".env")

    data_directory = ROOT_DIR / os.getenv("DATA_DIRECTORY", "data")
    data_directory.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        root_dir=ROOT_DIR,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        langsmith_tracing=_bool_env("LANGSMITH_TRACING", True),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "tn-agriculture-schemes-rag"),
        langsmith_endpoint=os.getenv(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ),
        source_url=os.getenv(
            "SOURCE_URL", "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
        ),
        schemes_landing_url=os.getenv(
            "SCHEMES_LANDING_URL", "https://www.tn.gov.in/schemes.php"
        ),
        chunk_size=_int_env("CHUNK_SIZE", 1000),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 150),
        retriever_k=_int_env("RETRIEVER_K", 4),
        request_timeout=_int_env("REQUEST_TIMEOUT", 30),
        request_retries=_int_env("REQUEST_RETRIES", 3),
        request_delay_seconds=_float_env("REQUEST_DELAY_SECONDS", 1.0),
        data_directory=data_directory,
        neo4j_uri=os.getenv("NEO4J_URI", ""),
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
        neo4j_index_name=os.getenv("NEO4J_INDEX_NAME", "scheme_chunks"),
        openai_max_output_tokens=_int_env("OPENAI_MAX_OUTPUT_TOKENS", 700),
        openai_timeout=_int_env("OPENAI_TIMEOUT", 60),
        max_input_chars=_int_env("MAX_INPUT_CHARS", 1200),
        max_history_messages=_int_env("MAX_HISTORY_MESSAGES", 50),
        app_env=os.getenv("APP_ENV", "development"),
        admin_actions_enabled=_bool_env("ADMIN_ACTIONS_ENABLED", True),
        admin_password=os.getenv("ADMIN_PASSWORD", ""),
    )

    _apply_langsmith_environment(config)
    return config


def validate_config(config: AppConfig, require_openai: bool = False) -> list[str]:
    """Return user-facing validation errors without exposing secret values."""

    errors: list[str] = []

    if require_openai and _is_missing_secret(config.openai_api_key):
        errors.append("OPENAI_API_KEY is missing. Add it to `.env` before asking questions.")

    if config.langsmith_tracing and _is_missing_secret(config.langsmith_api_key):
        errors.append(
            "LANGSMITH_TRACING is true, but LANGSMITH_API_KEY is missing. "
            "Add the key or set LANGSMITH_TRACING=false."
        )

    parsed_source = urlparse(config.source_url)
    if parsed_source.scheme not in {"http", "https"} or not parsed_source.netloc:
        errors.append("SOURCE_URL must be a valid http or https URL.")

    if not config.data_directory.exists() or not os.access(config.data_directory, os.W_OK):
        errors.append("DATA_DIRECTORY could not be created or is not writable.")

    if config.chunk_overlap >= config.chunk_size:
        errors.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    if config.chunk_size <= 0 or config.retriever_k <= 0:
        errors.append("CHUNK_SIZE and RETRIEVER_K must be positive integers.")

    if config.request_timeout <= 0 or config.request_retries <= 0:
        errors.append("REQUEST_TIMEOUT and REQUEST_RETRIES must be positive integers.")

    if config.max_input_chars <= 0 or config.openai_max_output_tokens <= 0:
        errors.append("MAX_INPUT_CHARS and OPENAI_MAX_OUTPUT_TOKENS must be positive.")

    if (
        config.is_production
        and config.admin_actions_enabled
        and not config.admin_password_configured
    ):
        errors.append(
            "Admin actions are enabled in production but ADMIN_PASSWORD is not set. "
            "Set ADMIN_PASSWORD or set ADMIN_ACTIONS_ENABLED=false."
        )

    return errors


def _apply_langsmith_environment(config: AppConfig) -> None:
    """Set tracing variables consumed by LangChain and LangSmith."""

    os.environ["LANGSMITH_TRACING"] = "true" if config.langsmith_tracing else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if config.langsmith_tracing else "false"
    os.environ["LANGSMITH_PROJECT"] = config.langsmith_project
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = config.langsmith_endpoint
    if not _is_missing_secret(config.langsmith_api_key):
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key
    if not _is_missing_secret(config.openai_api_key):
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
