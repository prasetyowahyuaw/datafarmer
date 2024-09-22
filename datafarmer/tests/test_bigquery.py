import pytest
from datafarmer.io import read_bigquery, is_oauth_set, write_bigquery
import pandas as pd

def test_read_bigquery():
    query = """
    SELECT * FROM `sample` LIMIT 2
    """
    df = read_bigquery(query, project_id="project_id")
    assert df.shape[0] > 0

def test_is_oauth_set():
    assert is_oauth_set() is True

def test_write_bigquery():
    project_id = "project_id"
    data = pd.DataFrame({"prompt": ["how to make a cake", "what is the education system in india"]})
    write_bigquery(df=data, project_id=project_id, table_id="test_table", dataset_id="dev_prasetyo_wianto", mode="WRITE_TRUNCATE")

    assert True