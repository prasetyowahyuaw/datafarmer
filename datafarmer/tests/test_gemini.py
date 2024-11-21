from datafarmer.llm import Gemini
import pandas as pd


def test_gemini_class():
    gemini = Gemini(project_id="xxxx-xxx-xx")
    data = pd.DataFrame(
        {
            "prompt":
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
                ],
            "id":
                [
                    "A","B","C","D","E","F","G","H","I"
                ]
        
        }
    )

    result = gemini.generate_from_dataframe(data)
    result_2 = gemini.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, pd.DataFrame)
