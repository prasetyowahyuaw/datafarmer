import pytest
import pandas as pd
from datafarmer.analysis import get_features_info, get_null_proportion

def test_get_features_info():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd'],
        'C': [1.1, 2.2, 3.3, 4.4],
        'D': [True, False, True, False]
    })

    df_features = get_features_info(df)

    assert df_features.shape[0] > 0

def test_get_null_proportion():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', None, 'd'],
        'C': [None, 2.2, None, 4.4],
        'D': [True, False, True, False]
    })

    df_null_proportion = get_null_proportion(df)
    assert df_null_proportion.shape[0] > 0