from google.cloud import bigquery
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
import pandas as pd
import polars as pl
import os
from datafarmer.utils import logger
from typing import Union


def is_oauth_set() -> bool:
    """Checks if the Google Cloud credentials are set. Returns True if it is set, otherwise False."""

    try:
        credentials, project = default()
        return True
    except DefaultCredentialsError:
        return False


def get_oauth_path() -> str:
    """Get the path of the Google Cloud credentials file."""

    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    else:
        return os.path.expanduser(
            "~/.config/gcloud/application_default_credentials.json"
        )


def get_bigquery_schema(dataset_id: str, project_id: str) -> list[dict]:
    """
    Retrieve a BigQuery schema from a given dataset id and project id

    Args:
        dataset_id (str): dataset id / name
        project_id (str): project id

    Returns:
        Dict: return a list of dictionary containing table name and its schema
    """

    assert is_oauth_set(), (
        "Google Cloud credentials are not set. please run 'gcloud auth application-default login' to set the credentials."
    )

    client = bigquery.Client(project=project_id)
    schemas = list()
    tables = client.list_tables(dataset_id)

    for table in tables:
        table_id = f"{table.project}.{table.dataset_id}.{table.table_id}"
        table = client.get_table(table_id)
        schema = [field.to_api_repr() for field in table.schema]
        schemas.append(dict(table_name=table_id, schema=schema))

    return schemas


def read_bigquery(
    query: str, project_id: str, return_type: str = "pandas"
) -> Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]:
    """Reads the content of a BigQuery by given query.
    then returns it as a DataFrame.

    Args:
        query (str): raw query string
        project_id (str): project id of the bigquery billing
        return_type (str, optional): return type of the query result, there are "pandas" and "polars". Defaults to "pandas".

    Returns:
        DataFrame: dataframe of the query result
    """

    assert is_oauth_set(), (
        "Google Cloud credentials are not set. please run 'gcloud auth application-default login' to set the credentials."
    )

    # check if the return type is valid
    assert return_type in ["pandas", "polars"], "return type is not valid."

    client = bigquery.Client(project=project_id)

    if return_type == "pandas":
        return client.query(query).to_dataframe()
    elif return_type == "polars":
        return pl.from_pandas(client.query(query).to_dataframe())


def preview_bigquery(
    query: str,
    project_id: str,
) -> str:
    assert is_oauth_set(), (
        "Google Cloud credentials are not set. please run 'gcloud auth application-default login' to set the credentials."
    )

    client = bigquery.Client(project=project_id)

    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    query_job = client.query(query, job_config=job_config)

    bytes_processed = query_job.total_bytes_processed
    mb = bytes_processed / (1024**2)
    gb = bytes_processed / (1024**3)

    if gb >= 1:
        return f"{gb:.1f} GB"
    else:
        return f"{mb:.0f} MB"


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

    assert is_oauth_set(), (
        "Google Cloud credentials are not set. please run 'gcloud auth application-default login' to set the credentials."
    )
    assert isinstance(df, pd.DataFrame), "data should be a pandas dataframe"

    client = bigquery.Client(project=project_id)

    job_config = bigquery.LoadJobConfig(
        schema=table_schema,
        create_disposition="CREATE_IF_NEEDED",
        write_disposition=mode,
        time_partitioning=(
            bigquery.TimePartitioning(field=partition_field)
            if partition_field
            else None
        ),
    )

    job = client.load_table_from_dataframe(
        df, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config
    )

    job.result()
