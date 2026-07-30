import pytest
from app.data.loader import (
    load_csv_into_duckdb,
    sanitize_column_names,
    sanitize_table_name,
    validate_csv_bytes,
)
from app.utils.errors import ValidationError
def test_sanitize_table_name_basic():
    assert sanitize_table_name("Sales Data.csv", set()) == "sales_data"
def test_sanitize_table_name_dedup():
    existing = {"sales_data"}
    assert sanitize_table_name("Sales Data.csv", existing) == "sales_data_2"
def test_sanitize_table_name_leading_digit():
    assert sanitize_table_name("2024_report.csv", set()).startswith("t_")
def test_sanitize_column_names_dedup_and_clean():
    cols = sanitize_column_names(["Revenue ($)", "Revenue ($)", "Region!"])
    assert cols == ["revenue", "revenue_1", "region"]
def test_validate_csv_bytes_rejects_non_csv():
    with pytest.raises(ValidationError):
        validate_csv_bytes(b"a,b\n1,2", "data.txt", max_bytes=1000)
def test_validate_csv_bytes_rejects_empty():
    with pytest.raises(ValidationError):
        validate_csv_bytes(b"", "data.csv", max_bytes=1000)
def test_validate_csv_bytes_rejects_too_large():
    with pytest.raises(ValidationError):
        validate_csv_bytes(b"x" * 100, "data.csv", max_bytes=10)
def test_load_csv_into_duckdb_registers_table(conn):
    raw = b"Order ID,Region,Revenue\n1,East,100\n2,West,200\n"
    table_name, df = load_csv_into_duckdb(conn, raw, "orders.csv", set())
    assert table_name == "orders"
    assert list(df.columns) == ["order_id", "region", "revenue"]
    result = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
    assert result[0] == 2
def test_load_csv_into_duckdb_rejects_no_rows(conn):
    raw = b"a,b,c\n"
    with pytest.raises(ValidationError):
        load_csv_into_duckdb(conn, raw, "empty.csv", set())
def test_load_csv_parses_date_columns(conn):
    raw = b"order_date,revenue\n2024-01-01,100\n2024-01-08,200\n2024-01-15,150\n"
    table_name, df = load_csv_into_duckdb(conn, raw, "orders.csv", set())
    import pandas as pd
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"])
