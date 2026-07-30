import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import duckdb
import pandas as pd
import pytest
@pytest.fixture
def conn():
    c = duckdb.connect(":memory:")
    yield c
    c.close()
@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "order_id": range(1, 11),
        "region": ["East", "West", "East", "North", "West", "East", "South", "North", "West", "East"],
        "revenue": [100, 200, 150, 300, 250, 120, 90, 310, 275, 130],
        "order_date": pd.date_range("2024-01-01", periods=10, freq="7D"),
    })
