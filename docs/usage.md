## I/O

This module helps you read and write data from several sources.

### Read BigQuery

Returns a DataFrame by parsing a query and a project ID.

```python
from datafarmer.io import read_bigquery

query = "SELECT * FROM `project.dataset.table`"
data = read_bigquery(
    query=query,
    project_id="project_id",
    return_type="pandas",  # or "polars"
)
```

### Write BigQuery

Ingests a pandas DataFrame into a BigQuery table.

```python
from datafarmer.io import write_bigquery
import pandas as pd

data = pd.DataFrame({
    "id": [1, 2, 3],
    "value": ["Belalang Tempur", "Soto Lamongan", "Kapal Selam"],
})

write_bigquery(
    df=data,
    project_id="project_id",
    dataset_id="dataset_id",
    table_id="table_id",
    mode="WRITE_TRUNCATE",  # WRITE_TRUNCATE | WRITE_APPEND | WRITE_EMPTY
)
```

You can also pass `table_schema` (a list of BigQuery schema field dicts) and `partition_field` to enable time partitioning.

### Preview BigQuery

Estimates how much data a query will scan without actually running it. Useful for cost checking before execution.

```python
from datafarmer.io import preview_bigquery

estimate = preview_bigquery(
    query="SELECT * FROM `project.dataset.table`",
    project_id="project_id",
)
print(estimate)  # e.g. "320 MB" or "1.4 GB"
```

### Get BigQuery Schema

Returns the schema of every table in a given dataset.

```python
from datafarmer.io import get_bigquery_schema

schemas = get_bigquery_schema(
    dataset_id="dataset_id",
    project_id="project_id",
)
# [{"table_name": "project.dataset.table", "schema": [...]}, ...]
```

### Get BigQuery Table Info

Returns metadata about a specific table (row count, byte size, schema, timestamps, labels, etc.).

```python
from datafarmer.io import get_bigquery_info

info = get_bigquery_info(
    project_id="project_id",
    dataset_id="dataset_id",
    table_id="table_id",
)
print(info["num_rows"])
print(info["last_modified"])
```

### Read Text

Returns the content of a text file as a string.

```python
from datafarmer.io import read_text

prompt = read_text("folder/prompt.txt")
```

### Read YAML

Returns the content of a YAML file as a dictionary.

```python
from datafarmer.io import read_yaml

setup = read_yaml("folder/setup.yml")
```

### Read Google Sheet

Reads a Google Sheet into a pandas DataFrame.

```python
from datafarmer.io import read_sheet

df = read_sheet(
    sheet_id="your_google_sheet_id",
    sheet_name="Sheet1",
)
```

### Write to Google Drive

Uploads a DataFrame as a CSV file to a Google Drive folder.

```python
from datafarmer.io import write_gdrive_file
import pandas as pd

data = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})

# Using a folder name (personal drive — folder is created if it doesn't exist)
result = write_gdrive_file(
    data=data,
    file_name="output.csv",
    folder_id="My Folder Name",
    project_id="project_id",
    is_shared_drive=False,
)

# Using a folder ID (shared drive)
result = write_gdrive_file(
    data=data,
    file_name="output.csv",
    folder_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs",
    project_id="project_id",
    is_shared_drive=True,
)

print(result["webViewLink"])  # direct link to the uploaded file
```

---

## LLM

This module provides LLM wrappers with built-in async batching, retry logic, and a unified interface across providers.

All LLM classes share the same core methods:

| Method | Description |
|---|---|
| `generate_from_dataframe(data)` | Synchronous generation over a DataFrame |
| `await generate_async_from_dataframe(data)` | Async generation (use inside `async` functions) |

The input DataFrame must have a `prompt` column. An `id` column is optional — if missing, the row index is used automatically. Any extra columns are passed as `**kwargs` to the underlying provider.

### Gemini

Wraps Google Gemini via Vertex AI SDK (default) or the newer `google-genai` SDK.

```python
from datafarmer.llm import Gemini
import pandas as pd

gemini = Gemini(project_id="project_id", gemini_version="gemini-2.5-flash-lite")

data = pd.DataFrame({
    "id": ["A", "B", "C"],
    "prompt": [
        "how to make a cake",
        "what is the education system in india",
        "explain the concept of gravity",
    ],
})

result = gemini.generate_from_dataframe(data)
```

#### With system instruction

```python
gemini = Gemini(
    project_id="project_id",
    gemini_version="gemini-2.5-flash-lite",
    system_instruction="You are a helpful assistant. Answer concisely.",
)
```

#### With structured JSON output (`google-genai` SDK)

```python
from datafarmer.llm import Gemini
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel
import pandas as pd

class PersonResponse(BaseModel):
    name: str
    age: int
    address: str

gemini = Gemini(
    project_id="project_id",
    google_sdk_version="genai",
    gemini_version="gemini-2.5-flash-lite",
)

data = pd.DataFrame({
    "id": ["A", "B"],
    "prompt": [
        "Extract JSON from: John is a 25 year old engineer in New York.",
        "Extract JSON from: Alice is a 30 year old doctor in Los Angeles.",
    ],
})

result = gemini.generate_from_dataframe(
    data,
    generation_config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PersonResponse,
    ),
)
```

#### With audio or image files

Add `audio_file_path` or `image_file_path` columns to your DataFrame and they are automatically attached to the request.

```python
data = pd.DataFrame({
    "id": ["A"],
    "prompt": ["Transcribe this audio."],
    "audio_file_path": ["path/to/audio.mp3"],
})

result = gemini.generate_from_dataframe(data)
```

#### Async usage

```python
result = await gemini.generate_async_from_dataframe(data, batch_size=50)
```

---

### Anthropic

Wraps the Anthropic (Claude) API. Requires an `ANTHROPIC_API_KEY` environment variable or an explicit `api_key` parameter.

```python
from datafarmer.llm import Anthropic
import pandas as pd

anthropic = Anthropic(
    model="claude-sonnet-4-6",
    system_instruction="You are a helpful data analyst.",
)

data = pd.DataFrame({
    "id": ["A", "B"],
    "prompt": [
        "Summarise the key trends in e-commerce for 2024.",
        "What are the main causes of customer churn?",
    ],
})

result = anthropic.generate_from_dataframe(data)
```

| Parameter | Default | Description |
|---|---|---|
| `model` | `"claude-sonnet-4-6"` | Anthropic model ID |
| `api_key` | `None` | Falls back to `ANTHROPIC_API_KEY` env var |
| `system_instruction` | `None` | System prompt |
| `max_tokens` | `8192` | Max tokens in the response |
| `max_attempts` | `3` | Retry attempts on rate limits / server errors |

---

### GithubCopilot

Uses the GitHub Models API (OpenAI-compatible endpoint). Authentication is via a GitHub personal token — fetched automatically from `gh auth token` if not provided.

```python
from datafarmer.llm import GithubCopilot
import pandas as pd

copilot = GithubCopilot(
    model="gpt-4o",
    system_instruction="You are a Python coding assistant.",
)

data = pd.DataFrame({
    "id": ["A", "B"],
    "prompt": [
        "Write a Python function to flatten a nested list.",
        "Explain the difference between a list and a generator.",
    ],
})

result = copilot.generate_from_dataframe(data)
```

!!! tip
    Make sure you are logged in to the GitHub CLI first:
    ```sh
    gh auth login
    ```

| Parameter | Default | Description |
|---|---|---|
| `model` | `"gpt-4o"` | Model name on GitHub Models |
| `github_token` | `None` | Falls back to `gh auth token` |
| `system_instruction` | `None` | System prompt |
| `max_attempts` | `3` | Retry attempts on rate limits / server errors |

---

### VertexRag

Wraps Vertex AI RAG (Retrieval-Augmented Generation) for building corpora, importing documents, and querying them.

```python
from datafarmer.llm import VertexRag

rag = VertexRag(project_id="project_id")
```

#### Create a corpus

```python
corpus = rag.set_corpus(
    display_name="my-knowledge-base",
    embedding_model="text-embedding-004",
)
corpus_name = corpus.name
```

#### Import files

Supports local paths, GCS (`gs://`), and Google Drive URLs.

```python
rag.import_files_to_rag(
    corpus_name=corpus_name,
    paths=[
        "https://drive.google.com/drive/folders/your_folder_id",
        "gs://your-bucket/docs/",
    ],
    chunk_size=512,
    chunk_overlap=100,
)
```

#### Query the corpus directly

```python
response = rag.get_retrieval_query(
    corpus_name=corpus_name,
    query="What is the refund policy?",
    similarity_top_k=10,
    vector_distance_threshold=0.5,
)
```

#### Use RAG as a Gemini tool

```python
from datafarmer.llm import Gemini

rag_tool = rag.get_rag_tool(
    corpus_name=corpus_name,
    similarity_top_k=10,
    vector_distance_threshold=0.5,
)

gemini = Gemini(
    project_id="project_id",
    gemini_version="gemini-2.5-flash-lite",
    tools=[rag_tool],
)
```

---

## Analysis

Helper functions for exploratory data analysis.

### Get Features Info

Returns a summary of each column: its dtype, number of unique values, and the unique values themselves.

```python
from datafarmer.analysis import get_features_info
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", "Bob", "Alice"],
    "score": [90, 85, 90],
    "grade": ["A", "B", "A"],
})

info = get_features_info(df)
print(info)
```

```
  Feature   Dtypes  Unique Values        Values
0    name   object              2  [Alice, Bob]
1   score    int64              2      [90, 85]
2   grade   object              2        [A, B]
```

### Get Null Proportion

Returns only the columns that have at least one null value, along with their null count and proportion.

```python
from datafarmer.analysis import get_null_proportion
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice", None, "Charlie"],
    "score": [90, 85, None],
    "grade": ["A", "B", "A"],
})

null_info = get_null_proportion(df)
print(null_info)
```

```
       Null Samples  Null Proportion
name              1         0.333333
score             1         0.333333
```

---

## Utils

### Logger

A pre-configured logger instance ready to use across your scripts.

```python
from datafarmer.utils import logger

logger.info("Starting pipeline...")
logger.warning("Missing values detected.")
logger.error("Failed to connect to BigQuery.")
```

Output format: `2025-01-01 12:00:00,000 | INFO: Starting pipeline...`
