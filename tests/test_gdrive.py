from datafarmer.io import write_gdrive_file

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

def test_write_gdrive_file():

    # sample dataframe with multi numerical and text columns
    data = pd.DataFrame(
        {
            "column1": [1, 2, 3],
            "column2": [4, 5, 6],
            "column3": ["a", "b", "c"],
        }
    )

    uploaded_metadata = write_gdrive_file(data, "test_1.csv", FOLDER_ID, PROJECT_ID)
    print(uploaded_metadata)
    assert uploaded_metadata is not None
    assert isinstance(uploaded_metadata, dict)
