import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Dict, Sequence

import click
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)
from pydantic import BaseModel, Field
from pydantic.networks import AnyUrl
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send


class Book(BaseModel):
    title: str = Field(..., description="The title of the book")
    author: str = Field(..., description="The author of the book")
    isbn: str = Field(..., description="The ISBN of the book")
    tags: list[str] = Field(default_factory=list, description="Tags associated with the book")


class BookISBNInput(BaseModel):
    isbn: str = Field(..., description="The ISBN of the book to be removed or retrieved")


class AddBookInput(Book):
    pass


class BookIndexInput(BaseModel):
    index: int = Field(..., description="The index of the book in the library")


class SuggestByAbstractInput(BaseModel):
    abstract: str = Field(
        ..., description="The abstract of the book for which a title is suggested"
    )


class AnalyzeBookInput(BaseModel):
    book: Book = Field(..., description="The book to analyze")
    query: str = Field(..., description="The query for analysis of the book")


class LibraryManagement:
    def __init__(self, books_path: Path):
        self.books_path = books_path
        if not books_path.exists():
            books_path.write_text("[]", encoding="utf-8")
        self.books = json.loads(books_path.read_text(encoding="utf-8"))

    def save_books(self):
        self.books_path.write_text(json.dumps(self.books, indent=4), encoding="utf-8")

    def add_book(self, book: Book) -> str:
        if any(b["isbn"] == book.isbn.strip() for b in self.books):
            return f"Book with ISBN '{book.isbn}' already exists."

        if not all([book.title.strip(), book.author.strip(), book.isbn.strip()]):
            return "Title, author, and ISBN cannot be empty."

        clean_tags = [t.strip() for t in book.tags if isinstance(t, str) and t.strip()]
        self.books.append(
            {
                "title": book.title.strip(),
                "author": book.author.strip(),
                "isbn": book.isbn.strip(),
                "tags": clean_tags,
            }
        )
        self.save_books()
        return f"Book '{book.title}' by {book.author} added to the library."

    def remove_book(self, isbn: str) -> str:
        updated = [b for b in self.books if b["isbn"] != isbn.strip()]
        if len(updated) == len(self.books):
            return f"No book found with ISBN '{isbn}'."
        self.books = updated
        self.save_books()
        return f"Book with ISBN '{isbn}' removed from the library."

    def get_num_books(self) -> int:
        return len(self.books)

    def get_all_books(self) -> list:
        return self.books

    def get_book_by_index(self, index: int) -> dict:
        if 0 <= index < len(self.books):
            return self.books[index]
        return {"error": "Book not found."}

    def get_book_by_isbn(self, isbn: str) -> dict:
        for b in self.books:
            if b["isbn"] == isbn.strip():
                return b
        return {"error": "Book not found."}

    def get_suggesting_random_book_prompt(self) -> str:
        return "Suggest a random book from the library. The suggestion should include the title, author, and a brief description."

    def get_suggesting_book_title_by_abstract_prompt(self, abstract: str) -> str:
        return f"Suggest a memorable, descriptive title for a book based on the following abstract: {abstract}"

    def get_analyzing_book_messages(self, book: dict, query: str) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": "This is the book I want to analyze: " + json.dumps(book),
            },
            {
                "role": "assistant",
                "content": "Sure! Let's analyze this book together. What would you like to know?",
            },
            {"role": "user", "content": query},
        ]


@click.command()
@click.option(
    "--log-level",
    default="INFO",
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--transport",
    default="http",
    type=click.Choice(["http", "stdio", "sse"]),
    help="Transport type to use for the MCP server (default: streamable_http)",
)
@click.option(
    "--port",
    default=8000,
    type=int,
    help="Port to run the HTTP server on (default: 8000)",
)
def serve(log_level: str, transport: str, port: int) -> None:
    """
    Start the MCP server for library management.
    This server allows adding, removing, and querying books in a library.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    books_path = Path("books.json")
    library = LibraryManagement(books_path)

    server = Server("mcp-library")

    #### TOOLS ####
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available tools for the library management system."""
        return [
            Tool(
                name="add_book",
                description="Add a book to the library",
                inputSchema={
                    "type": "object",
                    "required": ["title", "author", "isbn"],
                    "properties": {
                        "title": {
                            "description": "The title of the book",
                            "type": "string",
                        },
                        "author": {
                            "description": "The author of the book",
                            "type": "string",
                        },
                        "isbn": {
                            "description": "The ISBN of the book",
                            "type": "string",
                        },
                        "tags": {
                            "description": "Tags associated with the book",
                            "items": {"type": "string"},
                            "type": "array",
                        },
                    },
                },
            ),
            Tool(
                name="remove_book",
                description="Remove a book by its ISBN",
                inputSchema=BookISBNInput.model_json_schema(),
            ),
            Tool(
                name="get_num_books",
                description="Get the total number of books",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict
    ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
        """Call a specific tool by name with the provided arguments."""
        try:
            match name:
                case "add_book":
                    book = AddBookInput(**arguments)
                    result = library.add_book(book)
                    return [TextContent(type="text", text=result)]

                case "remove_book":
                    data = BookISBNInput(**arguments)
                    result = library.remove_book(data.isbn)
                    return [TextContent(type="text", text=result)]

                case "get_num_books":
                    result = library.get_num_books()
                    return [TextContent(type="text", text=str(result))]

                case _:
                    raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            raise ValueError(f"LibraryServer Error: {str(e)}")

    #### RESOURCES ####
    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List all available resources."""

        return [
            Resource(
                name="all_books",
                title="All Books",
                uri=AnyUrl("books://all"),
                description="Get all books in the library",
            ),
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list[ResourceTemplate]:
        """List all available resource templates."""
        return [
            ResourceTemplate(
                name="book_by_index",
                title="Book by Index",
                uriTemplate="books://index/{index}",
                description="Get a book by its index in the library",
            ),
            ResourceTemplate(
                name="book_by_isbn",
                title="Book by ISBN",
                uriTemplate="books://isbn/{isbn}",
                description="Get a book by its ISBN",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> str:
        """Read a resource by its URI."""
        uri_str = str(uri)
        if uri_str == "books://all":
            books = library.get_all_books()
            return json.dumps(books, indent=4)
        elif uri_str.startswith("books://index/"):
            index_str = uri_str.split("/")[-1]
            try:
                index = int(index_str)
                book = library.get_book_by_index(index)
                if "error" in book:
                    raise ValueError(book["error"])
                return json.dumps(book, indent=4)
            except ValueError:
                raise ValueError(f"Invalid index: {index_str}")
        elif uri_str.startswith("books://isbn/"):
            isbn = uri_str.split("/")[-1]
            book = library.get_book_by_isbn(isbn)
            if "error" in book:
                raise ValueError(book["error"])
            return json.dumps(book, indent=4)
        else:
            raise ValueError(f"Resource '{uri}' not found.")

    #### PROMPTS ####
    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """List all available prompts."""

        return [
            Prompt(
                name="suggest_random_book",
                description="Suggest a random book from the library. The suggestion should include the title, author, and a brief description.",
            ),
            Prompt(
                name="suggest_book_title_by_abstract",
                description="Suggest a memorable, descriptive title for a book based on the following abstract.",
                arguments=[
                    PromptArgument(
                        name="abstract",
                        description="The abstract of the book.",
                        required=True,
                    )
                ],
            ),
            Prompt(
                name="analyze_book",
                description="Analyze a book based on its content and user query.",
                arguments=[
                    PromptArgument(name="book", description="The book to analyze.", required=True),
                    PromptArgument(
                        name="query",
                        description="The query for analysis.",
                        required=True,
                    ),
                ],
            ),
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: Dict[str, str] | None) -> GetPromptResult:
        """Get a specific prompt by name."""
        prompts = await list_prompts()
        for prompt in prompts:
            if prompt.name == name:
                if arguments is None:
                    arguments = {}
                if prompt.arguments:
                    for arg in prompt.arguments:
                        if arg.name not in arguments and arg.required:
                            raise ValueError(f"Missing required argument: {arg.name}")
                break
        else:
            raise ValueError(f"Prompt '{name}' not found.")

        if name == "suggest_random_book":
            prompt_result = library.get_suggesting_random_book_prompt()
            return GetPromptResult(
                description=prompt.description,
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=prompt_result),
                    )
                ],
            )
        elif name == "suggest_book_title_by_abstract":
            prompt_result = library.get_suggesting_book_title_by_abstract_prompt(**arguments)
            return GetPromptResult(
                description=prompt.description,
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=prompt_result),
                    )
                ],
            )
        elif name == "analyze_book":
            book, query = arguments["book"], arguments["query"]
            messages = library.get_analyzing_book_messages(book, query)
            return GetPromptResult(
                description=prompt.description,
                messages=[
                    PromptMessage(
                        role=m["role"],
                        content=TextContent(type="text", text=m["content"]),
                    )
                    for m in messages
                ],
            )
        else:
            raise ValueError(f"Prompt '{name}' is not implemented.")

    #### Start the MCP server based on the specified transport ####
    logger.info("🚀 Launching Library MCP Server ...")

    if transport == "stdio":

        async def arun_stdio_server():
            """Run the MCP server using stdio transport."""

            logger.info("Starting MCP server with stdio transport...")
            options = server.create_initialization_options()
            async with stdio_server() as (read_stream, write_stream):
                await server.run(read_stream, write_stream, options)

        asyncio.run(arun_stdio_server())

    elif transport in ["http", "sse"]:

        session_manager = StreamableHTTPSessionManager(
            app=server,
            event_store=None,
            json_response=(
                True if transport == "http" else False
            ),  # Use JSON response for SSE, otherwise standard HTTP
            stateless=True,
        )

        # Handle HTTP requests with the session manager
        async def handle_streamable_http(scope: Scope, receive: Receive, send: Send) -> None:
            await session_manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            """Context manager for session manager."""
            async with session_manager.run():
                logger.info("Application started with StreamableHTTP session manager!")
                try:
                    yield
                finally:
                    logger.info("Application shutting down...")

        # Create an ASGI application using the transport
        starlette_app = Starlette(
            routes=[
                Mount("/mcp", app=handle_streamable_http),
            ],
            lifespan=lifespan,
        )

        uvicorn.run(starlette_app, host="127.0.0.1", port=port, log_level=log_level.lower())

    else:
        raise ValueError(f"Unsupported transport type: {transport}")


if __name__ == "__main__":
    serve()
