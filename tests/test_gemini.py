from datafarmer.llm import Gemini
from pandas import DataFrame
from dotenv import load_dotenv
from vertexai.generative_models import (
    GenerationConfig,
)
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
AUDIO_FOLDER = os.getenv("AUDIO_FOLDER")

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

def test_gemini_with_audio():
    gemini = Gemini(
        project_id=PROJECT_ID, 
        gemini_version="gemini-2.0-flash",
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