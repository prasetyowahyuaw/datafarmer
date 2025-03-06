from datafarmer.llm import Gemini
from pandas import DataFrame
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")

def test_gemini_class():
    gemini = Gemini(project_id=PROJECT_ID)
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
