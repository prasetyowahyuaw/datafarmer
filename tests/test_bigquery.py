import pytest
from datafarmer.io import read_bigquery, is_oauth_set, write_bigquery
import pandas as pd
import polars as pl
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
DATASET_ID = os.getenv("DATASET_ID")

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