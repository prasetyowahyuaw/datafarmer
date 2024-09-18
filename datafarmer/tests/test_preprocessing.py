import pytest
import pandas as pd
from datafarmer.analysis import get_features_info

def get_features_info():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': ['a', 'b', 'c', 'd'],
        'C': [1.1, 2.2, 3.3, 4.4],
        'D': [True, False, True, False]
    })

    df_features = get_features_info(df)

    assert df_features.shape[0] > 0