"""Skip tests that call real cloud services when their credentials/config are absent.

Offline tests live in ``tests/test_unit_*.py`` and always run.
"""

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

# test module -> environment variables it needs to talk to the real service
LIVE_REQUIREMENTS = {
    "test_bigquery.py": ["PROJECT_ID", "DATASET_ID", "TABLE_ID"],
    "test_gdrive.py": ["PROJECT_ID"],
    "test_sheet.py": ["PROJECT_ID", "SHEET_ID"],
    "test_gemini.py": ["PROJECT_ID"],
    "test_vertex_rag.py": ["PROJECT_ID", "VERTEX_CORPUS_ID"],
    "test_anthropic.py": ["ANTHROPIC_API_KEY"],
    "test_github_copilot.py": ["GITHUB_TOKEN"],
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        required = LIVE_REQUIREMENTS.get(item.path.name)
        if not required:
            continue
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            item.add_marker(
                pytest.mark.skip(reason=f"live test, missing env: {', '.join(missing)}")
            )
