import os
import pytest
from datafarmer.io import read_text, read_yaml


@pytest.fixture
def sample_text_file(tmpdir):
    file_path = tmpdir.join("sample.txt")
    file_path.write("Hello, World!")
    return file_path


@pytest.fixture
def sample_yaml_file(tmpdir):
    file_path = tmpdir.join("sample.yaml")
    file_path.write("key: value\nlist:\n  - item1\n  - item2")
    return file_path


def test_read_text(sample_text_file):
    text_content = read_text(sample_text_file)
    assert text_content == "Hello, World!"


def test_read_yaml(sample_yaml_file):
    yaml_content = read_yaml(sample_yaml_file)
    assert yaml_content == {"key": "value", "list": ["item1", "item2"]}
