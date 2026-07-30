import numpy as np
import pandas as pd
from app.analysis.anomaly import detect_anomalies, detect_iqr, detect_isolation_forest, detect_zscore
def _df_with_outlier():
    rng = np.random.default_rng(42)
    values = rng.normal(100, 5, size=200).tolist()
    values.append(500.0)  # obvious outlier
    return pd.DataFrame({"revenue": values})
def test_detect_zscore_finds_injected_outlier():
    df = _df_with_outlier()
    anomalies = detect_zscore(df, threshold=3.0)
    flagged_values = {a["value"] for a in anomalies}
    assert 500.0 in flagged_values
def test_detect_iqr_finds_injected_outlier():
    df = _df_with_outlier()
    anomalies = detect_iqr(df)
    flagged_values = {a["value"] for a in anomalies}
    assert 500.0 in flagged_values
def test_detect_zscore_no_false_positives_on_uniform_column():
    df = pd.DataFrame({"constant": [5] * 50})
    assert detect_zscore(df) == []
def test_detect_isolation_forest_multivariate():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": rng.normal(0, 1, 100).tolist() + [10],
        "y": rng.normal(0, 1, 100).tolist() + [-10],
    })
    anomalies = detect_isolation_forest(df, contamination=0.02)
    assert any(a["row_index"] == 100 for a in anomalies)
def test_detect_anomalies_dispatch():
    df = _df_with_outlier()
    assert detect_anomalies(df, "zscore")
    assert detect_anomalies(df, "iqr")
def test_anomaly_explanation_is_nonempty_string():
    df = _df_with_outlier()
    anomalies = detect_zscore(df)
    for a in anomalies:
        assert isinstance(a["explanation"], str) and len(a["explanation"]) > 10
