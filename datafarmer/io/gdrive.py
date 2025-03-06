import google.auth
import pandas as pd
import io
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaInMemoryUpload


def write_gdrive_file(
    data: pd.DataFrame, file_name: str, folder_id: str, project_id: str
) -> dict:
    """Write a dataframe to a google drive file

    Args:
        data (pd.DataFrame): input dataframe
        file_name (str): file name
        folder_id (str): folder name
        project_id (str): project id of the google cloud project

    Returns:
        dict: metadata of the uploaded file, contains id, name and webViewLink
    """
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/drive.file"],
        quota_project_id=project_id,
    )

    service = build("drive", "v3", credentials=creds)

    query = f"name='{folder_id}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    folders = results.get("files", [])

    if folders:
        folder_id = folders[0]["id"]
    else:
        file_metadata = {
            "name": folder_id,
            "mimeType": "application/vnd.google-apps.folder",
        }
        file = service.files().create(body=file_metadata, fields="id").execute()
        folder_id = file.get("id")

    data_buffer = io.StringIO()
    data.to_csv(data_buffer, index=False)
    data_content = data_buffer.getvalue().encode("utf-8")
    file_metadata = {"name": file_name, "mimeType": "text/csv", "parents": [folder_id]}

    media = MediaInMemoryUpload(data_content, mimetype="text/csv", resumable=True)

    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
        .execute()
    )

    return file
