from __future__ import annotations
import os
from functools import lru_cache
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    llm_max_agent_steps: int = int(os.getenv("LLM_MAX_AGENT_STEPS", "6"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "50"))
    max_files_per_session: int = int(os.getenv("MAX_FILES_PER_SESSION", "10"))
    max_rows_preview: int = int(os.getenv("MAX_ROWS_PREVIEW", "20"))
    session_ttl_minutes: int = int(os.getenv("SESSION_TTL_MINUTES", "120"))
    data_dir: str = os.getenv("DATA_DIR", "/tmp/ai_data_analyst")
    cors_origins: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024
@lru_cache
def get_settings() -> Settings:
    return Settings()
