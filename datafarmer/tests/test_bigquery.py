import pytest
from datafarmer.io import read_bigquery, is_oauth_set

def test_read_bigquery():
    query = """
    SELECT * FROM `tvlk-cxp-gcp-prod.prod_cx_L2.salesforce_user`
    """
    df = read_bigquery(query, project_id="tvlk-data-customerexp-dev")
    assert df.shape[0] > 0

def test_is_oauth_set():
    assert is_oauth_set() is True