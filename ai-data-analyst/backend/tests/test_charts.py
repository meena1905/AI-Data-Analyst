import pandas as pd
import pytest
from app.analysis.charts import build_chart_spec
def test_build_chart_spec_bar():
    df = pd.DataFrame({"region": ["East", "West"], "revenue": [100, 200]})
    spec = build_chart_spec(df, "bar", "Revenue by region", x="region", y="revenue")
    assert spec.type == "bar"
    assert spec.x == "region"
    assert spec.y == "revenue"
    assert len(spec.data) == 2
def test_build_chart_spec_infers_axes_when_missing():
    df = pd.DataFrame({"region": ["East", "West"], "revenue": [100, 200]})
    spec = build_chart_spec(df, "bar", "Revenue")
    assert spec.x == "region"
    assert spec.y == "revenue"
def test_build_chart_spec_rejects_empty_df():
    with pytest.raises(ValueError):
        build_chart_spec(pd.DataFrame(), "bar", "Empty")
def test_build_chart_spec_handles_datetime_column():
    df = pd.DataFrame({
        "order_date": pd.date_range("2024-01-01", periods=3, freq="D"),
        "revenue": [10, 20, 30],
    })
    spec = build_chart_spec(df, "line", "Trend", x="order_date", y="revenue")
    assert isinstance(spec.data[0]["order_date"], str)
