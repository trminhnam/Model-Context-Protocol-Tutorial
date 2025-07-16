import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore")


class Retriever:
    def __init__(
        self,
        embedding_model: SentenceTransformer,
        qdrant_path: str,
        collection_name: str = "documents",
        embedding_size: Optional[int] = None,
    ):
        self.embedding_model = embedding_model
        self.qdrant_path = qdrant_path
        self.collection_name = collection_name

        # Connect to Qdrant local instance (using the local path)
        self.client = QdrantClient(
            path=self.qdrant_path,
        )

        # If embedding_size is not provided, attempt to infer it
        if embedding_size is None:
            embedding_size = self.embedding_model.get_sentence_embedding_dimension()
        else:
            if embedding_size <= 0:
                raise ValueError("Vector size must be a positive integer.")
        self.embedding_size = embedding_size

        # Ensure the collection exists
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
            )

    def add_documents(self, documents: list[Dict[str, Any]]) -> Dict[str, str]:
        """
        Embeds the documents and adds them to the Qdrant collection.
        Each document is assigned a unique UUID.
        """
        try:
            embeddings = self.embedding_model.encode([doc["text"] for doc in documents])

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),  # unique id for each document
                    vector=embedding,
                    payload=doc,
                )
                for doc, embedding in zip(documents, embeddings)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            return {"status": "success", "message": "Documents added successfully."}
        except Exception as e:
            logging.error(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}

    def add_document(self, document: Dict[str, Any]) -> Dict[str, str]:
        """
        Embeds a single document and adds it to the Qdrant collection.
        The document is assigned a unique UUID.
        """
        try:
            embedding = self.embedding_model.encode([document["text"]])[0]
            point = PointStruct(
                id=str(uuid.uuid4()),  # unique id for the document
                vector=embedding,
                payload=document,
            )
            self.client.upsert(collection_name=self.collection_name, points=[point])
            return {"status": "success", "message": "Document added successfully."}
        except Exception as e:
            logging.error(f"Error adding document: {e}")
            return {"status": "error", "message": str(e)}

    def retrieve(self, query: str, limit: int = 5) -> List[dict]:
        """
        Retrieves documents similar to the query using cosine similarity.
        Returns a list of dictionaries containing the document text and its score.
        """
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
        )

        results = [
            {"text": point.payload["text"], "score": point.score} for point in search_result.points
        ]
        return results


if __name__ == "__main__":
    # Example usage
    embedding_model_id = "Alibaba-NLP/gte-multilingual-base"
    embedding_model = SentenceTransformer(
        embedding_model_id, trust_remote_code=True, cache_folder="./cache"
    )
    retriever = Retriever(
        embedding_model=embedding_model,
        qdrant_path="./qdrant_database",
        collection_name="mcp_database",
    )

    results = retriever.retrieve("What is organic chemistry?", limit=5)
    for i, result in enumerate(results):
        print(f"Result {i + 1} (Score: {result['score']:.4f}): {result['text'][:250]}...")
