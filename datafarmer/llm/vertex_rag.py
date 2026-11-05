from typing import Any, Optional

import vertexai
from google.genai import types
from vertexai import rag
from vertexai.rag import RagCorpus

from datafarmer.utils import logger


class VertexRag:
    def __init__(self, project_id: str, location: str = "us-central1") -> None:
        self.project_id = project_id
        self.location = location

        vertexai.init(project=self.project_id, location=self.location)

    def get_documents_from_corpus(
        self,
        corpus_name: str,
        page_size: int = 100,
    ) -> Any:
        """
        return the documents from the rag corpus

        Args:
            corpus_name (str): corpus name format "projects/{project_id}/locations/{location}/corpora/{corpus_id}"
            page_size (int, optional): number of files fetched per page. Defaults to 100.
        """

        return list(rag.list_files(corpus_name=corpus_name, page_size=page_size))

    def set_corpus(
        self,
        display_name: str,
        embedding_model: str = "gemini-embedding-001",
    ) -> RagCorpus:
        """
        create a rag corpus and return RagCorpus object

        Args:
            display_name (str): name the rag corpus
            embedding_model (str, optional): name of the embedding model. Defaults to "gemini-embedding-001".
        """

        backend_config = rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=f"publishers/google/models/{embedding_model}"
                )
            )
        )

        return rag.create_corpus(
            display_name=display_name, backend_config=backend_config
        )

    def import_files_to_rag(
        self,
        corpus_name: str,
        paths: list[str],
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        max_embedding_requests_per_min: int = 900,
    ) -> None:
        """
        import files to the rag corpus

        Args:
            corpus_name (str): corpus name format "projects/{project_id}/locations/{location}/corpora/{corpus_id}"
            paths (list[str]): list of file paths can be local files or google(gcs or gdrive). https://drive.google.com/drive/folders/{folder_id} or https://drive.google.com/file/d/{file_id}.
            chunk_size (int, optional): size of text chunks. Defaults to 512.
            chunk_overlap (int, optional): overlap between chunks. Defaults to 100.
            max_embedding_requests_per_min (int, optional): rate limit for embedding requests. Defaults to 900.
        """

        response = rag.import_files(
            corpus_name=corpus_name,
            paths=paths,
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            ),
            max_embedding_requests_per_min=max_embedding_requests_per_min,
        )

        logger.info(f"rag imported : {response.imported_rag_files_count} files")

    def get_retrieval_query(
        self,
        corpus_name: str,
        query: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = 0.5,
        file_ids: Optional[list[str]] = None,
    ) -> Any:
        """
        perform direct context retrieval

        Args:
            corpus_name (str): corpus name format "projects/{project_id}/locations/{location}/corpora/{corpus_id}"
            query (str): query text
            similarity_top_k (int, optional): number of top similar chunks to retrieve. Defaults to 10.
            vector_distance_threshold (float, optional): similarity treshold. Defaults to 0.5.
            file_ids (Optional[list[str]], optional): optional list of specific file ids to query. Defaults to None.

        Returns:
            Any: retrieved context response
        """

        rag_resource = rag.RagResource(rag_corpus=corpus_name, rag_file_ids=file_ids)

        return rag.retrieval_query(
            rag_resources=[rag_resource],
            text=query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=similarity_top_k,
                filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),
            ),
        )

    def get_rag_tool(
        self,
        corpus_name: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = 0.5,
    ) -> types.Tool:
        """
        get the rag tool, to be passed to ``Gemini(tools=[...])``

        Args:
            corpus_name (str): corpus name, format "projects/{project_id}/locations/{location}/corpora/{corpus_id}"
            similarity_top_k (int, optional): number of top similar chunks to retrieve. Defaults to 10.
            vector_distance_threshold (float, optional): similarity treshold. Defaults to 0.5.
        Returns:
            types.Tool: google-genai retrieval tool backed by Vertex RAG
        """

        return types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(rag_corpus=corpus_name)
                    ],
                    similarity_top_k=similarity_top_k,
                    vector_distance_threshold=vector_distance_threshold,
                )
            )
        )
