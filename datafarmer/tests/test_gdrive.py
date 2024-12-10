import pytest
from datafarmer.io import write_gdrive_file

import pandas as pd
import io


def test_write_gdrive_file():
    
    # sample dataframe with multi numerical and text columns
    data = pd.DataFrame({
        "column1": [1, 2, 3],
        "column2": [4, 5, 6],
        "column3": ["a", "b", "c"],
    })

    
    id = write_gdrive_file(data, "test.csv", "test_project_id")
    assert id is not None

    

