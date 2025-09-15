import pytest
from datafarmer.io import read_bigquery, is_oauth_set, write_bigquery, get_bigquery_schema, preview_bigquery, get_bigquery_info
import pandas as pd
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
import os
import json

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")

def test_read_bigquery():
    query = """
    SELECT edition FROM `bigquery-public-data.america_health_rankings.ahr`
    """
    start = datetime.now()
    df_pandas = read_bigquery(query, project_id=PROJECT_ID, return_type="pandas")
    end_pandas = datetime.now() - start
    print(f"Pandas processing time: {end_pandas}, rows: {df_pandas.shape[0]}")
    start = datetime.now()
    df_polars = read_bigquery(query, project_id=PROJECT_ID, return_type="polars")
    end_polars = datetime.now() - start
    print(f"Polars processing time: {end_polars}, rows: {df_polars.shape[0]}")
    assert isinstance(df_pandas, pd.DataFrame)
    assert isinstance(df_polars, pl.DataFrame)
    assert df_pandas.shape[0] > 0
    assert df_polars.shape[0] > 0

def test_is_oauth_set():
    assert is_oauth_set() is True

def test_write_bigquery():
    data = pd.DataFrame({"prompt": ["how to make a cake", "what is the education system in india"]})
    write_bigquery(df=data, project_id=PROJECT_ID, table_id="test_table", dataset_id=DATASET_ID, mode="WRITE_TRUNCATE")

    assert True

def test_get_bigquery_schema():
    bigquery_schemas = get_bigquery_schema(
        dataset_id=DATASET_ID,
        project_id=PROJECT_ID,
    )

    print(f"Bigquery schemas: {bigquery_schemas}")
    assert isinstance(bigquery_schemas, list)
    assert len(bigquery_schemas) > 0

def test_preview_bigquery():
    query = """
    SELECT * FROM `bigquery-public-data.america_health_rankings.ahr`
    """
    preview = preview_bigquery(query, project_id=PROJECT_ID)
    print(f"Preview bigquery cost: {preview}")
    assert isinstance(preview, str)

def test_get_bigquery_info():
    table = get_bigquery_info(project_id=PROJECT_ID, dataset_id=DATASET_ID, table_id=TABLE_ID)

    print(f"Table metadata: {table}")

    assert table['num_rows'] > 0