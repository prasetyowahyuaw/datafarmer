from google.cloud import bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import pandas as pd
import os

def is_oauth_set() -> bool:
    """Checks if the Google Cloud credentials are set. Returns True if it is set, otherwise False."""
    
    try:
        credentials, project = default()
        return True
    except DefaultCredentialsError:
        return False

def read_bigquery(query: str, project_id: str) -> pd.DataFrame:
    """Reads the content of a BigQuery by given query. 
    then returns it as a DataFrame.

    Args:
        query (str): raw query string
        project_id (str): project id of the bigquery billing

    Returns:
        pd.DataFrame: dataframe of the query result
    """

    assert is_oauth_set(), "Google Cloud credentials are not set. please run 'gcloud auth application-default login' to set the credentials."

    client = bigquery.Client(project=project_id)
    return client.query(query).to_dataframe()