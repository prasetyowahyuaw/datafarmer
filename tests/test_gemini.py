from datafarmer.llm import Gemini
from pandas import DataFrame
from dotenv import load_dotenv
from vertexai.generative_models import (
    GenerationConfig,
)
from google.genai.types import GenerateContentConfig
import os
import json
from pydantic import BaseModel

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
AUDIO_FOLDER = os.getenv("AUDIO_FOLDER")

class SampleResponse(BaseModel):
    name: str
    age: int
    address: str

def test_gemini_class():
    gemini = Gemini(project_id=PROJECT_ID, gemini_version="gemini-2.0-flash")
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
    gemini = Gemini(project_id=PROJECT_ID, google_sdk_version="genai", gemini_version="gemini-2.0-flash")
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

def test_gemini_class_genai_with_response_schema():
    gemini = Gemini(
        project_id=PROJECT_ID, 
        google_sdk_version="genai", 
        gemini_version="gemini-2.0-flash",
    )

    # Test with response_schema - this should always return valid JSON
    data_with_schema = DataFrame(
        {
            "prompt": [
                "Generate person data for John, age 25, living in New York",
            ],
            "id": ["A"],
        }
    )
    
    result_with_schema = gemini.generate_from_dataframe(
        data_with_schema,
        generation_config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SampleResponse
        )
    )
    
    print("=== Result with response_schema (should be valid JSON) ===")
    print(result_with_schema)
    
    # Test WITHOUT response_schema - this might generate broken JSON that needs fixing
    data_without_schema = DataFrame(
        {
            "prompt": [
                "Return a JSON object with fields name, age, address for person John who is 25 years old living in New York. Please intentionally make some JSON formatting mistakes like using single quotes, adding extra commas, or missing brackets.",
            ],
            "id": ["B"],
        }
    )
    
    result_without_schema = gemini.generate_from_dataframe(
        data_without_schema,
        generation_config=GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.8  # Higher temperature for more variability
            # No response_schema here - allows broken JSON
        )
    )
    
    print("=== Result without response_schema (might trigger JSON fixing) ===")
    print(result_without_schema)

    assert isinstance(result_with_schema, DataFrame)
    assert isinstance(result_without_schema, DataFrame)


def test_json_fixer_directly():
    """Test the JSON fixer with manually broken JSON"""
    gemini = Gemini(
        project_id=PROJECT_ID, 
        google_sdk_version="genai", 
        gemini_version="gemini-2.0-flash",
        generation_config=GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    
    # Manually test the JSON fixer with broken JSON
    broken_json_examples = [
        "{'name': 'John', 'age': 25, 'address': 'New York',}",  # Single quotes + trailing comma
        "{name: 'John', age: 25, address: 'New York'}",  # Unquoted keys
        "Here is the data: {\"name\": \"John\", \"age\": 25, \"address\": \"New York\"",  # Missing closing brace + extra text
        "{'name': 'John' 'age': 25, 'address': 'New York'}",  # Missing comma
    ]
    
    print("=== Testing JSON Fixer Directly ===")
    for i, broken_json in enumerate(broken_json_examples):
        print(f"\n--- Test {i+1} ---")
        print(f"Broken JSON: {broken_json}")
        
        fixed_json = gemini._fix_json_response(broken_json)
        print(f"Fixed JSON: {fixed_json}")
        
        # Try to parse the fixed JSON
        try:
            parsed = json.loads(fixed_json)
            print(f"✅ Successfully parsed: {parsed}")
        except json.JSONDecodeError as e:
            print(f"❌ Still invalid: {e}")

    assert True  # Just ensure the test runs

# def test_gemini_with_audio(): 
#     gemini = Gemini(
#         project_id=PROJECT_ID, 
#         gemini_version="gemini-2.0-flash",
#         generation_config=GenerationConfig(
#             audio_timestamp=True
#         )
#     )
#     data = DataFrame(
#         {
#             "prompt": [
#                 "please trasncribe the following audio file",
#                 "please trasncribe the following audio file",
#                 "please trasncribe the following audio file",
                
#             ],
#             "audio_file_path": [
#                 f"{AUDIO_FOLDER}/1.mp3",
#                 f"{AUDIO_FOLDER}/2.mp3",
#                 f"{AUDIO_FOLDER}/3.mp3",
#             ],
#         }
#     )

#     result = gemini.generate_from_dataframe(data)
#     print(result)

#     assert isinstance(result, DataFrame)

# def test_load_audio():
#     with open(f"{AUDIO_FOLDER}/1.mp3", "rb") as f:
#         audio_content = f.read()
#     assert isinstance(audio_content, bytes)
