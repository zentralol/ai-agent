"""Application settings loaded from the environment.

Settings are validated at load time and exposed through a cached accessor so the
rest of the service can depend on a single immutable configuration instance.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Immutable, environment-driven configuration for the agent service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8010, alias="PORT")

    # LLM provider configuration. Not used by the minimal contract layer yet, but
    # validated here so later phases can rely on it being present and typed.
    llm_provider: str = Field(default="deepseek", alias="LLM_PROVIDER")
    llm_model: str = Field(default="deepseek-v4-flash", alias="LLM_MODEL")
    llm_api_key: str | None = Field(default=None, alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, alias="LLM_BASE_URL")

    # Shared secret for internal service-to-service auth. The Express gateway
    # sends it as the ``X-Internal-Service-Token`` header on every call. When
    # unset, inbound auth is skipped (development only) and a warning is logged.
    agent_internal_token: str | None = Field(default=None, alias="AGENT_INTERNAL_TOKEN")

    # Supabase access stays server-side. It is used for controlled preference
    # lookup and later for agent-owned run/trace tables.
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_service_role_key: str | None = Field(
        default=None, alias="SUPABASE_SERVICE_ROLE_KEY"
    )
    supabase_user_preferences_table: str = Field(
        default="onboarding_preferences", alias="SUPABASE_USER_PREFERENCES_TABLE"
    )
    supabase_timeout_seconds: float = Field(
        default=3.0, alias="SUPABASE_TIMEOUT_SECONDS"
    )

    # Conversation persistence. The agent owns reading history for context and
    # writing user/assistant turns. Table names are fixed constants in
    # app.conversations.repository since they must match the product's Supabase
    # migrations; only the history window is tunable.
    conversation_history_limit: int = Field(
        default=20, alias="CONVERSATION_HISTORY_LIMIT"
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, immutable settings instance."""

    return Settings()
