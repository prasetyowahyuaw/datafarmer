from datafarmer.io import read_sheet

import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
SHEET_ID = os.getenv("SHEET_ID")
SHEET_NAME = os.getenv("SHEET_NAME", "Sheet1")

def test_read_sheet():

    data = read_sheet(SHEET_ID, SHEET_NAME)

    assert isinstance(data, pd.DataFrame)
    assert not data.empty