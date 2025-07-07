import pandas as pd
import gspread
import google.auth

def read_sheet(sheet_id: str, sheet_name: str) -> pd.DataFrame:
    """Read a Google Sheet into a pandas DataFrame.

    Args:
        sheet_id (str): The ID of the Google Sheet.
        sheet_name (str): The name of the sheet within the Google Sheet.

    Returns:
        pd.DataFrame: The data from the specified sheet.
    """
    creds, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly"
        ]
    )

    client = gspread.authorize(creds)
    sheet = client.open_by_key(sheet_id)
    worksheet = sheet.worksheet(sheet_name)
    data = worksheet.get_all_records()
    
    return pd.DataFrame(data)