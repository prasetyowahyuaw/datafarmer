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

gemini = Gemini(project_id="project_id", gemini_version="gemini-2.0-flash")
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

## Utils
😅 coming soon ...
## Analysis
😅 coming soon ...