from datafarmer.llm import VertexRag, Gemini
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
CORPUS_ID = os.getenv("VERTEX_CORPUS_ID")
rag = VertexRag(project_id=PROJECT_ID)

def test_vertexrag_set_corpus():
    rag_corpus = rag.set_corpus(display_name="test_corpus")
    rag_corpus_name = rag_corpus.name
    print(f" created corpus name : {rag_corpus_name}")

    assert rag_corpus_name is not None


def test_vertexrag_get_documents_from_corpus():
    documents = rag.get_documents_from_corpus(corpus_name=CORPUS_ID)
    print(f"documents : {documents}")
    assert documents is not None


def test_vertexrag_get_retrieval_query():
    retrieval_response = rag.get_retrieval_query(
        corpus_name=CORPUS_ID,
        query="how to submit refund",
    )
    print(f"retrieval_response : {retrieval_response}")
    assert retrieval_response is not None


def test_vertex_rag_tool():
    rag_tool = rag.get_rag_tool(
        corpus_name=CORPUS_ID,
        similarity_top_k=10,
        vector_distance_threshold=0.6
    )

    gemini = Gemini(project_id=PROJECT_ID,tools=[rag_tool])
    data = pd.DataFrame(
        {
            "prompt":["how to submit refund", "what if I want to cancel my order"],
        }
    )

    result = gemini.generate_from_dataframe(data)
    print(f"result generation: {result}")

    assert isinstance(result, pd.DataFrame)