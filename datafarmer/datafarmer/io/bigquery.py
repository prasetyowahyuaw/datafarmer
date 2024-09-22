from google.cloud import bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import pandas as pd
import os
from datafarmer.utils import logger

import logging

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

def write_bigquery(
    df: pd.DataFrame,
    project_id: str,
    table_id: str,
    dataset_id: str,
    mode: str = "WRITE_TRUNCATE",
    table_schema: list = None,
    partition_field: str = None,
) -> None:
    """ingest a dataframe to bigquery table

    Args:
        df (pd.DataFrame): dataframe to be ingested
        project_id (str): biquery project id
        table_id (str): table name
        dataset_id (str): dataset name
        mode (str, optional): WRITE_TRUNCATE, WRITE_APPEND, and WRITE EMPTY. Defaults to "WRITE_TRUNCATE". Defaults to "WRITE_TRUNCATE".
        table_schema (list, optional): table schema including its column name and types. Defaults to None.
        partition_field (str, optional): table partition. Defaults to None.
    """

    assert is_oauth_set(), "Google Cloud credentials are not set. please run 'gcloud auth application-default login' to set the credentials."
    assert isinstance(df, pd.DataFrame), "data should be a pandas dataframe"

    client = bigquery.Client(project=project_id)

    job_config = bigquery.LoadJobConfig(
        schema=table_schema,
        create_disposition="CREATE_IF_NEEDED",
        write_disposition=mode,
        time_partitioning=bigquery.TimePartitioning(
            field=partition_field
        ) if partition_field else None
    )

    job = client.load_table_from_dataframe(
        df, 
        f"{project_id}.{dataset_id}.{table_id}", 
        job_config=job_config
    )

    job.result()  # Waits for the job to complete.