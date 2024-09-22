from datafarmer.llm import Gemini
import pandas as pd


def test_gemini_class():
    gemini = Gemini(project_id="project_id")
    data = pd.DataFrame(
        {"prompt":
            [
                "how to make a cake",
                "what is the education system in india",
                "explain the concept of gravity",
                "return the list of all the prime numbers between 1 to 100",
                "why is the sky blue",
                "who is the founder of microsoft",
                "explain the concept of machine learning",
                "how to evaluate machine learning models",
                "what is the best model for classification with skewed data",
            ]
        }
    )

    gemini = Gemini(project_id="your_project_id")
    result = gemini.generate_from_dataframe(data)

    assert isinstance(result, pd.DataFrame)
