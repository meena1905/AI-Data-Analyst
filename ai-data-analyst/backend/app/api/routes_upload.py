from __future__ import annotations
import logging
from fastapi import APIRouter, File, UploadFile
from app.config import get_settings
from app.data.loader import load_csv_into_duckdb
from app.data.profiler import profile_dataframe
from app.models import UploadResponse
from app.session import session_store
from app.utils.errors import ValidationError
from app.utils.logging import Timer
logger = logging.getLogger(__name__)
router = APIRouter(tags=["upload"])
@router.post("/api/upload", response_model=UploadResponse)
async def upload_csvs(
    files: list[UploadFile] = File(...),
    session_id: str | None = None,
) -> UploadResponse:
    settings = get_settings()
    if len(files) == 0:
        raise ValidationError("No files were provided.")
    if len(files) > settings.max_files_per_session:
        raise ValidationError(f"Too many files (max {settings.max_files_per_session}).")

    session = session_store.get_or_create(session_id)

    with Timer(logger, f"upload {len(files)} file(s)"):
        for f in files:
            raw = await f.read()
            from app.data.loader import validate_csv_bytes
            validate_csv_bytes(raw, f.filename or "upload.csv", settings.max_upload_bytes)

            table_name, df = load_csv_into_duckdb(
                session.conn, raw, f.filename or "upload.csv", set(session.tables.keys())
            )
            profile = profile_dataframe(df, table_name, f.filename or "upload.csv")
            session.tables[table_name] = profile
            logger.info("Loaded table '%s' (%d rows, %d cols) from %s", table_name, profile.n_rows, profile.n_columns, f.filename)
    return UploadResponse(session_id=session.id, tables=list(session.tables.values()))
