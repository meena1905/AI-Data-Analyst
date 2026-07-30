import numpy as np
import pandas as pd
from app.data.profiler import profile_dataframe, profile_to_prompt_text
def test_profile_flags_high_missing_column():
    df = pd.DataFrame({
        "id": range(10),
        "mostly_null": [None] * 8 + [1, 2],
    })
    profile = profile_dataframe(df, "t", "f.csv")
    assert any("mostly_null" in w for w in profile.quality_warnings)
def test_profile_flags_constant_column():
    df = pd.DataFrame({"id": range(10), "flag": [True] * 10})
    profile = profile_dataframe(df, "t", "f.csv")
    assert any("constant" in w for w in profile.quality_warnings)
def test_profile_flags_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    profile = profile_dataframe(df, "t", "f.csv")
    assert any("duplicate" in w.lower() for w in profile.quality_warnings)
def test_profile_to_prompt_text_contains_table_name():
    df = pd.DataFrame({"a": [1, 2, 3]})
    profile = profile_dataframe(df, "my_table", "f.csv")
    text = profile_to_prompt_text(profile)
    assert "my_table" in text
    assert "a (" in text
