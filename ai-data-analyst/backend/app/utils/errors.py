"""Application-level exceptions with clear, user-facing messages.

Keeping these distinct from generic exceptions lets the API layer map them
to correct HTTP status codes and lets tools/agent code fail predictably.
"""
from __future__ import annotations
class AppError(Exception):
    """Base class for all handled application errors."""
    status_code = 400
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
class ValidationError(AppError):
    """Raised when uploaded data or a request fails validation."""
    status_code = 422
class SessionNotFoundError(AppError):
    status_code = 404
class DatasetNotFoundError(AppError):
    status_code = 404
class QueryExecutionError(AppError):
    """SQL or pandas execution failed. Message should be safe to show to the
    LLM so it can self-correct, and to the user for transparency."""
    status_code = 400
class LLMError(AppError):
    status_code = 502
