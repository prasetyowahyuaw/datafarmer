from datafarmer.llm import Anthropic
from pandas import DataFrame
from dotenv import load_dotenv

load_dotenv()


def test_anthropic_basic():
    anthropic = Anthropic(model="claude-haiku-4-5-20251001")
    data = DataFrame(
        {
            "prompt": [
                "how to make a cake",
                "what is the education system in india",
                "explain the concept of gravity",
                "return the list of all the prime numbers between 1 to 100",
                "why is the sky blue",
            ],
            "id": ["A", "B", "C", "D", "E"],
        }
    )

    result = anthropic.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)


def test_anthropic_with_system_instruction():
    anthropic = Anthropic(
        model="claude-haiku-4-5-20251001",
        system_instruction="You are a concise assistant. Always answer in one sentence.",
    )
    data = DataFrame(
        {
            "prompt": [
                "who is the founder of microsoft",
                "what is machine learning",
            ],
            "id": ["A", "B"],
        }
    )

    result = anthropic.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)
