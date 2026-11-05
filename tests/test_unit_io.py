"""Offline unit tests for the I/O layer."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from datafarmer.io import bigquery as bq
from datafarmer.io import write_gdrive_file


def test_bigquery_requires_credentials():
    with patch.object(bq, "is_oauth_set", return_value=False):
        for call in (
            lambda: bq.read_bigquery("select 1", "p"),
            lambda: bq.preview_bigquery("select 1", "p"),
            lambda: bq.get_bigquery_info("p", "d", "t"),
            lambda: bq.get_bigquery_schema("d", "p"),
            lambda: bq.write_bigquery(pd.DataFrame({"a": [1]}), "p", "t", "d"),
        ):
            with pytest.raises(RuntimeError, match="credentials"):
                call()


def test_read_bigquery_invalid_return_type():
    with patch.object(bq, "is_oauth_set", return_value=True):
        with pytest.raises(ValueError):
            bq.read_bigquery("select 1", "p", return_type="csv")


@pytest.mark.parametrize("return_type", ["pandas", "polars"])
def test_read_bigquery_return_types(return_type):
    import polars as pl

    with patch.object(bq, "is_oauth_set", return_value=True), patch.object(
        bq.bigquery, "Client"
    ) as client:
        client.return_value.query.return_value.to_dataframe.return_value = pd.DataFrame({"a": [1]})
        out = bq.read_bigquery("select 1", "p", return_type=return_type)
    assert isinstance(out, pd.DataFrame if return_type == "pandas" else pl.DataFrame)


def test_preview_bigquery_formats():
    with patch.object(bq, "is_oauth_set", return_value=True), patch.object(
        bq.bigquery, "Client"
    ) as client:
        job = client.return_value.query.return_value
        job.total_bytes_processed = 5 * 1024**2
        assert bq.preview_bigquery("q", "p") == "5 MB"
        job.total_bytes_processed = int(1.5 * 1024**3)
        assert bq.preview_bigquery("q", "p") == "1.5 GB"


def test_write_bigquery_rejects_non_dataframe():
    with patch.object(bq, "is_oauth_set", return_value=True):
        with pytest.raises(TypeError):
            bq.write_bigquery([1, 2], "p", "t", "d")


def test_gdrive_folder_name_is_escaped():
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {"files": [{"id": "F"}]}
    service.files.return_value.create.return_value.execute.return_value = {"id": "x"}
    with patch("datafarmer.io.gdrive.google.auth.default", return_value=(MagicMock(), "p")), patch(
        "datafarmer.io.gdrive.build", return_value=service
    ):
        write_gdrive_file(pd.DataFrame({"a": [1]}), "f.csv", "Bob's \\ folder", "p")
    q = service.files.return_value.list.call_args.kwargs["q"]
    assert "name='Bob\\'s \\\\ folder'" in q
