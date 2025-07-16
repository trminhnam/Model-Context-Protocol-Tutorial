import logging
import os
from typing import Any, Dict, List, Optional

import click
from langchain_community.tools import DuckDuckGoSearchResults
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from retriever import Retriever
from create_vector_database import (
    discover_and_get_all_files,
    load_json_file,
    load_txt_file,
    load_md_file,
)
import uuid
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AddDocumentResponse(BaseModel):

    status: str = Field(..., description="The status of the operation (success or error).")
    message: str = Field(
        ..., description="A message providing additional information about the operation."
    )

    def __str__(self):
        return f"AddDocumentResponse(status={self.status}, message={self.message})"


class QueryInput(BaseModel):

    query: str = Field(..., description="The query to retrieve relevant documents.")
    num_documents: int = Field(
        default=5,
        description="The number of documents to retrieve for the query.",
    )

    def __str__(self):
        return f'QueryInput(query="{self.query}", num_documents={self.num_documents})'


class RetrievedDocument(BaseModel):

    text: str = Field(..., description="The content of the retrieved document.")
    score: Optional[float] = Field(
        None, description="The score of the retrieved document based on similarity."
    )

    def __str__(self):
        return f"RetrievedDocument(text={self.text[:50]}..., score={self.score})"  # Display first 50 characters


class RetrievalResult(BaseModel):

    results: List[RetrievedDocument] = Field(
        ..., description="List of retrieved documents with their scores."
    )

    def __str__(self):
        return f"RetrievalResult(count={len(self.results)})"  # Display count of results


class SearchResult(BaseModel):

    results: List[RetrievedDocument] = Field(
        ..., description="List of retrieved documents from the search."
    )

    def __str__(self):
        return f"SearchResult(count={len(self.results)})"  # Display count of results


class LocalDocument(BaseModel):

    text: str = Field(..., description="The content of the local document.")
    path: str = Field(..., description="The path of the local document.")
    source: Optional[str] = Field(None, description="The source of the local document.")
    split: Optional[str] = Field(None, description="The split of the local document.")
    id: Optional[str] = Field(None, description="The id of the local document.")

    def __str__(self):
        return f"LocalDocument(path={self.path}, source={self.source}, split={self.split}, id={self.id}, text={self.text[:50]}...)"


class LocalDocumentList(BaseModel):

    documents: List[LocalDocument] = Field(..., description="The list of local documents.")

    def __str__(self):
        return f"LocalDocumentList(count={len(self.documents)})"


def create_mcp_server(
    retriever: Retriever,
    documents_path_by_topics: Dict[str, List[str]],
    all_document_paths: List[str],
):
    """
    Starts the MCP server with the retriever.
    """

    mcp = FastMCP(
        name="Retriever MCP Server",
        host="127.0.0.1",
        port=2545,
        description="A server that provides MCP-powered agentic RAG capabilities.",
    )

    ##### Tools #####
    @mcp.tool(
        name="retrieve_documents_from_database",
        title="Retrieve Documents from Database",
        description="Retrieve documents similar to the query from the database.",
    )
    def retrieve(input: QueryInput) -> RetrievalResult:

        retrieval_results = retriever.retrieve(input.query, input.num_documents)
        return RetrievalResult(
            results=[
                RetrievedDocument(text=result["text"], score=result["score"])
                for result in retrieval_results
            ]
        )

    @mcp.tool(
        name="search_query_on_internet",
        title="Search the Query on the Internet",
        description="Search for documents on the internet related to the query. This tool will be used when the database does not have relevant documents for the query.",
    )
    def search_query_on_internet(input: QueryInput) -> SearchResult:

        search_engine = DuckDuckGoSearchResults(
            output_format="list", num_results=input.num_documents, backend="text"
        )
        search_results = search_engine.invoke(input.query)
        return SearchResult(
            results=[
                RetrievedDocument(
                    text=f"Title: {result['title']}\nText: {result['snippet']}",
                    score=None,
                )
                for result in search_results
            ]
        )

    @mcp.tool(
        name="add_document_to_database",
        title="Add a Document to Database",
        description="Add a document to the database for future retrieval.",
    )
    def add_document_to_database(
        document: str, topic_name: Optional[str] = None, document_name: Optional[str] = None
    ) -> AddDocumentResponse:

        doc_id = str(uuid.uuid4())
        if topic_name is None:
            topic_name = "default"
        if not os.path.exists(f"./documents/{topic_name}"):
            os.makedirs(f"./documents/{topic_name}")
        if document_name is None:
            document_name = doc_id

        # save to the documents folder
        with open(f"./documents/{topic_name}/{document_name}.json", "w") as f:
            doc_data = {
                "text": document,
                "path": f"./documents/{topic_name}/{document_name}.json",
                "source": "user-added",
                "split": topic_name,
                "id": doc_id,
            }
            json.dump(doc_data, f, indent=4)

        # add to the database
        retriever.add_document(doc_data)

        return AddDocumentResponse(status="success", message="Document added to database.")

    ##### Resources #####
    @mcp.resource(
        uri="document://topics",
        name="get_all_topics",
        title="Get All Topics in the Database",
        description="A resource to get all topics in the database.",
    )
    def get_all_topics() -> List[str]:
        """
        MCP resource to get all topics in the database.
        """

        return sorted(list(documents_path_by_topics.keys()))

    @mcp.resource(
        uri="document://topics/{topic_name}",
        name="get_all_documents_by_topic",
        title="Get All Documents by Topic",
        description="A resource to get all documents by topic.",
    )
    def get_all_documents_by_topic(topic_name: str) -> LocalDocumentList:

        doc_paths = documents_path_by_topics[topic_name]
        documents = []
        for doc_path in doc_paths:
            if doc_path.endswith(".json"):
                doc = load_json_file(doc_path)
                doc = LocalDocument(
                    text=doc["text"],
                    path=doc_path,
                    source=doc.get("source", ""),
                    split=doc.get("split", ""),
                    id=doc.get("id", ""),
                )
            elif doc_path.endswith(".txt"):
                doc = LocalDocument(
                    text=load_txt_file(doc_path),
                    path=doc_path,
                    source=None,
                    split=None,
                    id=None,
                )
            elif doc_path.endswith(".md"):
                doc = LocalDocument(
                    text=load_md_file(doc_path),
                    path=doc_path,
                    source=None,
                    split=None,
                    id=None,
                )
            else:
                continue
            documents.append(doc)
        return LocalDocumentList(documents=documents)

    @mcp.resource(
        uri="document://topics/{topic_name}/pages/{page_number}",
        name="get_documents_by_topic",
        title="Get Documents by Topic",
        description="A resource to get documents by topic using pagination. Each page contains 10 documents.",
    )
    def get_documents_by_topic(topic_name: str, page_number: int) -> LocalDocumentList:

        doc_paths = documents_path_by_topics[topic_name]
        start_index, end_index = (page_number - 1) * 10, page_number * 10
        documents = []
        for doc_path in doc_paths[start_index:end_index]:
            if doc_path.endswith(".json"):
                doc = load_json_file(doc_path)
            elif doc_path.endswith(".txt"):
                doc = LocalDocument(
                    text=load_txt_file(doc_path),
                    path=doc_path,
                    source=None,
                    split=None,
                    id=None,
                )
            elif doc_path.endswith(".md"):
                doc = LocalDocument(
                    text=load_md_file(doc_path),
                    path=doc_path,
                    source=None,
                    split=None,
                    id=None,
                )
            else:
                continue
            documents.append(doc)
        return LocalDocumentList(documents=documents)

    ##### Prompts #####
    @mcp.prompt(
        name="retrieve_documents_from_database_prompt",
        title="Retrieve Documents from Database Prompt",
        description="Prompt to retrieve documents similar to the query from the database.",
    )
    def get_retrieve_document_from_database_prompt(
        query: str, num_documents: int = 5
    ) -> base.UserMessage:

        return base.UserMessage(
            content=f"""Answer the following question. Remember to use the `retrieve_documents_from_database` tool to find {num_documents} relevant documents in the database to support your answer.

Question: {query}
"""
        )

    @mcp.prompt(
        name="retrieve_document_and_search_internet_prompt",
        title="Retrieve Document and Search Internet Prompt",
        description="Prompt to retrieve documents similar to the query from the database and search the internet if necessary.",
    )
    def get_retrieve_document_and_search_internet_prompt(
        query: str, num_documents: int = 5
    ) -> base.UserMessage:

        return base.UserMessage(
            content=f"""Answer the following question. Remember to use the `retrieve_documents_from_database` tool to find {num_documents} relevant documents in the database to support your answer. If the database does not have relevant documents, use the `search_documents_on_internet` tool to search for documents on the internet.

Question: {query}
"""
        )

    @mcp.prompt(
        name="search_query_on_internet_prompt",
        title="Search Internet Prompt",
        description="Prompt to search the internet for documents related to the query.",
    )
    def get_search_internet_prompt(query: str, num_documents: int = 5) -> base.UserMessage:

        return base.UserMessage(
            content=f"""Answer the following question. Remember to use the `search_documents_on_internet` tool to find {num_documents} relevant documents on the internet to support your answer.

Question: {query}
"""
        )

    @mcp.prompt(
        name="add_single_document_to_database_prompt",
        title="Add Single Document to Database Prompt",
        description="Prompt to add a single document to the database.",
    )
    def get_add_single_document_to_database_prompt(document: str) -> base.UserMessage:

        return base.UserMessage(
            content=f"""Add the following document to the database for future retrieval using the `add_single_document_to_database` tool.
            
Document: {document}
"""
        )

    return mcp


@click.command()
def run_mcp_server():
    """
    Command-line entry point to run the MCP server.
    """
    embedding_model = SentenceTransformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True,
        cache_folder="./cache",
    )
    qdrant_path = "./qdrant_database"

    retriever = Retriever(
        embedding_model=embedding_model,
        qdrant_path=qdrant_path,
        collection_name="mcp_database",
    )

    DOCUMENT_DIR = "./documents"
    DOCUMENTS_PATH_BY_TOPICS = {}
    for folder in os.listdir(DOCUMENT_DIR):
        if os.path.isdir(os.path.join(DOCUMENT_DIR, folder)):
            DOCUMENTS_PATH_BY_TOPICS[folder] = discover_and_get_all_files(
                os.path.join(DOCUMENT_DIR, folder),
                allowed_extensions=[".json", ".txt", ".md"],
                recursive=True,
            )

    ALL_DOCUMENT_PATHS = []
    for folder in DOCUMENTS_PATH_BY_TOPICS:
        ALL_DOCUMENT_PATHS.extend(DOCUMENTS_PATH_BY_TOPICS[folder])

    mcp_server = create_mcp_server(retriever, DOCUMENTS_PATH_BY_TOPICS, ALL_DOCUMENT_PATHS)
    # mcp_server.run(transport="streamable-http")
    print("Starting MCP server...")
    mcp_server.run(transport="stdio")


if __name__ == "__main__":
    run_mcp_server()
