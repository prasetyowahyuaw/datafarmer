from datafarmer.llm import GithubCopilot
from pandas import DataFrame
from dotenv import load_dotenv
import os

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")


def test_github_copilot_basic():
    copilot = GithubCopilot(model="gpt-4o", github_token=GITHUB_TOKEN)
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

    result = copilot.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)


def test_github_copilot_with_system_instruction():
    copilot = GithubCopilot(
        model="gpt-4o",
        github_token=GITHUB_TOKEN,
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

    result = copilot.generate_from_dataframe(data)
    print(result)

    assert isinstance(result, DataFrame)
