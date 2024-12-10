import google.auth
import pandas as pd
import io
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload

def write_gdrive_file(data: pd.DataFrame, file_name: str, project_id: str) -> dict:
    """
    writes a pandas dataframe to a Google Drive file. and return id
    """
    creds, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/drive.file'],
        quota_project_id=project_id
    )

    print(f"creds: {creds}")

    data_buffer = io.StringIO()
    data.to_csv(data_buffer, index=False)
    data_content = data_buffer.getvalue().encode("utf-8")

    service = build("drive", "v3", credentials=creds)

    file_metadata = {
        "name": file_name,
        "mimeType": "text/csv",
    }

    media = MediaInMemoryUpload(
        data_content, 
        mimetype="text/csv", 
        resumable=True
    )

    file = service.files().create(
        body=file_metadata, 
        media_body=media, 
        fields='id, name, webViewLink'
    ).execute()

    return file