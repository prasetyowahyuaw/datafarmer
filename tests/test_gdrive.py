from datafarmer.io import write_gdrive_file

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
FOLDER_NAME = os.getenv("GDRIVE_FOLDER_NAME")
FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID")

def test_write_gdrive_file_from_cred_owner_drive():

    # sample dataframe with multi numerical and text columns
    data = pd.DataFrame(
        {
            "column1": [1, 2, 3],
            "column2": [4, 5, 6],
            "column3": ["a", "b", "c"],
        }
    )

    uploaded_metadata = write_gdrive_file(data, "test_1_owner.csv", FOLDER_NAME, PROJECT_ID)
    print(uploaded_metadata)
    assert uploaded_metadata is not None
    assert isinstance(uploaded_metadata, dict)


def test_write_gdrive_file_from_cred_shared_drive():

    # sample dataframe with multi numerical and text columns
    data = pd.DataFrame(
        {
            "column1": [1, 2, 3],
            "column2": [4, 5, 6],
            "column3": ["a", "b", "c"],
        }
    )

    uploaded_metadata = write_gdrive_file(data, "test_1_shared.csv", FOLDER_ID, PROJECT_ID, is_shared_drive=True)
    print(uploaded_metadata)
    assert uploaded_metadata is not None
    assert isinstance(uploaded_metadata, dict)