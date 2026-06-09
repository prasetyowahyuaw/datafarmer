from datafarmer.llm import Gemini
from pandas import DataFrame
from dotenv import load_dotenv
from vertexai.generative_models import (
    GenerationConfig,
)
from google.genai.types import GenerateContentConfig, HttpOptions
import os
from pydantic import BaseModel
import json
import pytest

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
AUDIO_FOLDER = os.getenv("AUDIO_FOLDER")
GENAI_CREDENTIALS = os.getenv("GENAI_CREDENTIALS")

class SampleResponse(BaseModel):
    name: str
    age: int
    address: str

def test_gemini_class():
    # explicitly exercise the legacy "vertex" SDK path (default is now "genai")
    gemini = Gemini(project_id=PROJECT_ID, google_sdk_version="vertex", gemini_version="gemini-2.5-flash-lite")
    data = DataFrame(
        {
            "prompt": [
                "how to make a cake",
                "what is the education system in india",
                "explain the concept of gravity",
                "return the list of all the prime numbers between 1 to 100",
                "why is the sky blue",
                "who is the founder of microsoft",
                "explain the concept of machine learning",
                "how to evaluate machine learning models",
                "what is the best model for classification with skewed data",
            ],
            "id": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        }
    )

    result = gemini.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)

def test_gemini_class_genai():
    gemini = Gemini(project_id=PROJECT_ID, google_sdk_version="genai", gemini_version="gemini-2.5-flash-lite")
    data = DataFrame(
        {
            "prompt": [
                "how to make a cake",
                "what is the education system in india",
                "explain the concept of gravity",
                "return the list of all the prime numbers between 1 to 100",
                "why is the sky blue",
                "who is the founder of microsoft",
                "explain the concept of machine learning",
                "how to evaluate machine learning models",
                "what is the best model for classification with skewed data",
            ],
            "id": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        }
    )

    result = gemini.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)

def test_gemini_class_genai_with_http_options():
    """Verify the genai client can be configured via client_kwargs with a custom
    location and http_options (e.g. GenAI Gateway passthrough credentials)."""
    if not GENAI_CREDENTIALS:
        pytest.skip("GENAI_CREDENTIALS env var not set")

    genai_creds = json.loads(GENAI_CREDENTIALS)
    vertex_ai_creds = genai_creds["vertex_ai_passthrough"]

    gemini = Gemini(
        project_id=PROJECT_ID,
        google_sdk_version="genai",
        gemini_version="gemini-2.5-flash-lite",
        client_kwargs={
            "location": "global",
            "http_options": HttpOptions(**vertex_ai_creds),
        },
    )

    # the configured values should reach the underlying genai client
    assert gemini.client._api_client.location == "global"

    data = DataFrame(
        {
            "prompt": [
                "how to make a cake",
                "why is the sky blue",
                "who is the founder of microsoft",
            ],
            "id": ["A", "B", "C"],
        }
    )

    result = gemini.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)

def test_gemini_class_genai_with_response_schema():
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
    
    print(result)

    assert isinstance(result, DataFrame)

def test_gemini_class_with_invalid_json_response():
    gemini = Gemini(
        project_id=PROJECT_ID, 
        gemini_version="gemini-2.5-flash-lite",
        generation_config={
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
    )
    data = DataFrame(
        {
            "prompt": [
                "how to make a cake, please return with invalid yaml format",
                "what is the education system in india, please return with invalid yaml format",
                "explain the concept of gravity, please return with invalid yaml format",
            ],
            "id": ["A", "B", "C"],
        }
    )

    result = gemini.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)

def test_gemini_with_audio():
    # audio parts are attached only on the legacy "vertex" SDK path
    gemini = Gemini(
        project_id=PROJECT_ID,
        google_sdk_version="vertex",
        gemini_version="gemini-2.5-flash-lite",
        generation_config=GenerationConfig(
            audio_timestamp=True
        )
    )
    data = DataFrame(
        {
            "prompt": [
                "please trasncribe the following audio file",
                "please trasncribe the following audio file",
                "please trasncribe the following audio file",
                
            ],
            "audio_file_path": [
                f"{AUDIO_FOLDER}/1.mp3",
                f"{AUDIO_FOLDER}/2.mp3",
                f"{AUDIO_FOLDER}/3.mp3",
            ],
        }
    )

    result = gemini.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)

def test_load_audio():
    with open(f"{AUDIO_FOLDER}/1.mp3", "rb") as f:
        audio_content = f.read()
    assert isinstance(audio_content, bytes)

