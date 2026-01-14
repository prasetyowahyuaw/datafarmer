## I/O

this modules can help you to read/write a data with several sources. the current capabilities are:

### Read BigQuery

return a dataframe by parsing query and the project id only.

```python
from datafarmer.io import read_bigquery

query = "select * from `project.dataset.table`"
data = read_bigquery(
    query=query
    project_id="project_id",
    return_type="pandas"
)
```

`read_bigquery` function also can return polars dataframe, just change the `return_type` value to `polars`.

### Write BigQuery

if we have a dataframe and want to store it in the BigQuery table

```python
from datafarmer.io import write_bigquery
import pandas as pd

data = pd.DataFrame({
    "id": [1,2,3],
    "value": ["Belalang Tempur", "Soto Lamongan", "Kapal Selam"]
})
write_bigquery(
    df=data,
    project_id="project_id",
    table_id="table_id",
    dataset_id="dataset_id",
    mode="WRITE_TRUNCATE"
)
```

### Read Text

return string value from file reading result

```python
from datafarmer.io import read_file

prompt = read_file("folder/prompt.txt")
```

### Read Yaml

return dictionary value from yaml file

```python
from datafarmer.io import read_yaml

setup = read_yaml("folder/setup.yml)
```

## LLM

this module contains LLM wrapper, and still for google gemini only, for others will be coming soon. mostly the usage from this modules is to do asynchronous generation given by a dataframe.

### Generating from Dataframe

```python
from datafarmer.llm import Gemini

gemini = Gemini(project_id="project_id", gemini_version="gemini-2.5-flash-lite")
data = DataFrame({
    "prompt": [
        "how to make a cake",
        "what is the education system in india",
        "explain the concept of gravity",
    ],
    "id": ["A", "B", "C"],
})

result = gemini.generate_from_dataframe(data)
```

by default the `Gemini` class uses vertex-ai sdk. alternatively you can also use `google-genai` new sdk by modified the parameter. New `google-genai` sdk have multiples capabilities, can be checked in [here](https://googleapis.github.io/python-genai/#generate-content).

```python
from datafarmer.llm import Gemini
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

class SampleResponse(BaseModel):
    name: str
    age: int
    address: str

gemini = Gemini(
    project_id=PROJECT_ID,
    google_sdk_version="genai",
    gemini_version="gemini-2.5-flash-lite",
)

data = DataFrame(
    {
        "prompt": [
            "please generate the json response with name, age, and address from the following context. Context: John is a 25 year old software engineer living in New York.",
            "please generate the json response with name, age, and address from the following context. Context: Alice is a 30 year old doctor living in Los Angeles.",
            "please generate the json response with name, age, and address from the following context. Context: Bob is a 28 year old artist living in San Francisco.",
        ],
        "id": ["A", "B", "C"],
    }
)

result = gemini.generate_from_dataframe(
    data,
    generation_config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SampleResponse
    )
)

```

## Utils

😅 coming soon ...

## Analysis

😅 coming soon ...
