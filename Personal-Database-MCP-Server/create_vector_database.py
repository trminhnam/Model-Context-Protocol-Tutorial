import os
import json

import os
import warnings

from tqdm import tqdm
from typing import Union, List, Literal
from pydantic import BaseModel, Field

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


warnings.filterwarnings("ignore")

DOCUMENT_DIR = "./documents"


def discover_and_get_all_files(root_dir, allowed_extensions=None, recursive=False):
    if allowed_extensions is None:
        allowed_extensions = [".json", ".txt", ".md"]

    all_files = []
    for item_name in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item_name)
        if os.path.isfile(item_path) and any(item_path.endswith(ext) for ext in allowed_extensions):
            all_files.append(item_path)
        elif os.path.isdir(item_path) and recursive:
            all_files.extend(discover_and_get_all_files(item_path, allowed_extensions, recursive))
    return all_files


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_md_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":

    all_document_paths = discover_and_get_all_files(
        DOCUMENT_DIR, allowed_extensions=[".json", ".txt", ".md"], recursive=True
    )
    print(f"Found {len(all_document_paths)} documents in {DOCUMENT_DIR}.")

    documents = []
    with tqdm(total=len(all_document_paths), desc="Loading documents") as pbar:
        for doc_path in all_document_paths:
            if doc_path.endswith(".json"):
                doc = load_json_file(doc_path)
            elif doc_path.endswith(".txt"):
                doc = {"text": load_txt_file(doc_path)}
            elif doc_path.endswith(".md"):
                doc = {"text": load_md_file(doc_path)}
            else:
                continue

            # Add metadata
            doc["path"] = doc_path
            documents.append(doc)
            pbar.update(1)

    # embedding_model_id = "jinaai/jina-embeddings-v3"
    embedding_model_id = "Alibaba-NLP/gte-multilingual-base"
    embedding_model = SentenceTransformer(
        embedding_model_id, trust_remote_code=True, cache_folder="./cache"
    )

    document_embeddings = embedding_model.encode(
        [doc["text"] for doc in documents],
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=16,
    )

    client = QdrantClient(path="./qdrant_database")
    client.create_collection(
        collection_name="mcp_database",
        vectors_config=models.VectorParams(
            size=embedding_model.get_sentence_embedding_dimension(), distance=models.Distance.COSINE
        ),
    )

    client.upload_points(
        collection_name="mcp_database",
        points=[
            models.PointStruct(
                id=idx,
                vector=document_embeddings[idx],
                payload={
                    "text": doc["text"],
                    "path": doc["path"],
                    "source": doc.get("source", ""),
                    "split": doc.get("split", ""),
                    "id": doc.get("id", ""),
                },
            )
            for idx, doc in enumerate(documents)
        ],
    )

    hits = client.query_points(
        collection_name="mcp_database",
        query=embedding_model.encode("What is organic chemistry?"),
        limit=5,
    )
    for hit in hits.points:
        print(f"ID: {hit.id}, Score: {hit.score}, Text: {hit.payload['text'][:250]}...")
    print("Example query results printed above.")
