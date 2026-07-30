from __future__ import annotations
import numpy as np
import pandas as pd
def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
def detect_zscore(df: pd.DataFrame, threshold: float = 3.0) -> list[dict]:
    anomalies = []
    for col in _numeric_columns(df):
        series = df[col].dropna()
        if series.std(ddof=0) == 0 or len(series) < 5:
            continue
        mean, std = series.mean(), series.std(ddof=0)
        z = (series - mean) / std
        flagged = z[z.abs() > threshold]
        for idx, z_val in flagged.items():
            anomalies.append({
                "row_index": int(idx),
                "column": col,
                "value": float(df.loc[idx, col]),
                "method": "zscore",
                "score": round(float(z_val), 2),
                "explanation": (
                    f"'{col}' value {df.loc[idx, col]:.2f} is {abs(z_val):.1f} standard "
                    f"deviations {'above' if z_val > 0 else 'below'} the column mean "
                    f"({mean:.2f} ± {std:.2f})."
                ),
            })
    return anomalies
def detect_iqr(df: pd.DataFrame, k: float = 1.5) -> list[dict]:
    anomalies = []
    for col in _numeric_columns(df):
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - k * iqr, q3 + k * iqr
        flagged = series[(series < lower) | (series > upper)]
        for idx, val in flagged.items():
            side = "below" if val < lower else "above"
            bound = lower if val < lower else upper
            anomalies.append({
                "row_index": int(idx),
                "column": col,
                "value": float(val),
                "method": "iqr",
                "score": round(float(abs(val - bound) / (iqr or 1)), 2),
                "explanation": (
                    f"'{col}' value {val:.2f} falls outside the typical range "
                    f"[{lower:.2f}, {upper:.2f}] (IQR-based), {side} the expected bound."
                ),
            })
    return anomalies
def detect_isolation_forest(df: pd.DataFrame, contamination: float = 0.03) -> list[dict]:
    from sklearn.ensemble import IsolationForest

    cols = _numeric_columns(df)
    sub = df[cols].dropna()
    if len(sub) < 10 or len(cols) < 1:
        return []

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    labels = model.fit_predict(sub)
    scores = model.decision_function(sub)  # lower = more anomalous

    anomalies = []
    for pos, (idx, is_outlier) in enumerate(zip(sub.index, labels)):
        if is_outlier != -1:
            continue
        row = df.loc[idx]
        z_contribs = {
            c: abs((row[c] - df[c].mean()) / (df[c].std(ddof=0) or 1)) for c in cols
        }
        top_col = max(z_contribs, key=z_contribs.get)
        anomalies.append({
            "row_index": int(idx),
            "column": top_col,
            "value": float(row[top_col]),
            "method": "isolation_forest",
            "score": round(float(-scores[pos]), 3),  # higher = more anomalous
            "explanation": (
                f"Row flagged as a multivariate outlier by Isolation Forest "
                f"(anomaly score {-scores[pos]:.3f}); the most unusual field is "
                f"'{top_col}' = {row[top_col]:.2f}, which is far from typical for this dataset "
                f"when considered alongside the row's other values."
            ),
        })
    anomalies.sort(key=lambda a: a["score"], reverse=True)
    return anomalies
def detect_anomalies(df: pd.DataFrame, method: str = "zscore") -> list[dict]:
    if method == "zscore":
        return detect_zscore(df)
    if method == "iqr":
        return detect_iqr(df)
    if method == "isolation_forest":
        return detect_isolation_forest(df)
    raise ValueError(f"Unknown anomaly detection method: {method}")
