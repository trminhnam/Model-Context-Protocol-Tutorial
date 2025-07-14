import json
import random

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.metadata_utils import get_display_name
from pydantic.networks import AnyUrl

# Create server parameters for stdio connection
server_params = StdioServerParameters(command="python", args=["./server.py"])


# Ref: https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#client-display-utilities
async def display_tools(session: ClientSession):
    """Display available tools with human-readable names"""
    tools_response = await session.list_tools()

    for tool in tools_response.tools:
        # get_display_name() returns the title if available, otherwise the name
        display_name = get_display_name(tool)
        print(f"Tool: {display_name}")
        if tool.description:
            print(f"   {tool.description}")


async def display_resources(session: ClientSession):
    """Display available resources with human-readable names"""
    resources_response = await session.list_resources()

    for resource in resources_response.resources:
        display_name = get_display_name(resource)
        print(f"Resource: {display_name} ({resource.uri})")


async def display_resource_templates(session: ClientSession):
    """Display available resource templates with human-readable names"""
    resource_templates_response = await session.list_resource_templates()

    for template in resource_templates_response.resourceTemplates:
        display_name = get_display_name(template)
        print(f"Resource Template: {display_name} ({template.uriTemplate})")
        if template.description:
            print(f"   {template.description}")


async def display_prompts(session: ClientSession):
    """Display available prompts with human-readable names"""
    prompts_response = await session.list_prompts()

    for prompt in prompts_response.prompts:
        display_name = get_display_name(prompt)
        print(f"Prompt: {display_name} ({prompt.name})")
        if prompt.description:
            print(f"   {prompt.description}")


async def test_book_library_management_mcp_server(session: ClientSession):
    # List available prompts
    print("*** Available Tools ***")
    await display_tools(session)

    # List available resources
    print("\n*** Available Resources ***")
    await display_resources(session)

    # List available resource templates
    print("\n*** Available Resource Templates ***")
    await display_resource_templates(session)

    # List available prompts
    print("\n*** Available Prompts ***")

    await display_prompts(session)
    print("\n" + "#" * 50 + "\n\n")

    ### Interact with MCP Server
    print("*** Calling Tool: get_num_books ***")
    num_book_response = await session.call_tool(
        "get_num_books",
        arguments={},
    )
    print(f"Results={num_book_response.content[0].text}\n")

    print("*** Get all books ***")
    all_books_response = await session.read_resource(AnyUrl("books://all"))
    print(f"All Books:\n{all_books_response.contents[0].text}\n")

    print("*** Adding a new book ***")
    new_book = {
        "title": "Random Book Name " + str(random.randint(1, 1000)),
        "author": "Random Author " + str(random.randint(1, 1000)),
        "isbn": f"{random.randint(9780000000000, 9789999999999)}",
        "tags": random.choices(
            [
                "fiction",
                "non-fiction",
                "science",
                "history",
                "fantasy",
                "kids",
                "mystery",
                "romance",
                "thriller",
                "biography",
                "self-help",
                "health",
                "travel",
                "cookbook",
                "poetry",
                "graphic novel",
            ],
            k=3,
        ),
    }
    add_nook_response = await session.call_tool("add_book", arguments=new_book)
    print(f"Response from add_book: {add_nook_response.content[0].text}\n")

    print("*** Number of Books in Library after addition ***")
    num_books = await session.call_tool("get_num_books")
    print(f"Number of Books in Library: {num_books.content[0].text}\n")

    print("*** Retrieve the added book by index ***")
    read_book_response = await session.read_resource(
        AnyUrl(f"books://index/{int(num_books.content[0].text) - 1}")
    )
    print(f"Retrieved Book:\n{read_book_response.contents[0].text}\n")

    print("*** Retrieve the added book by ISBN ***")
    read_book_response = await session.read_resource(
        AnyUrl(f"books://isbn/{new_book['isbn']}")
    )
    print(f"Retrieved Book:\n{read_book_response.contents[0].text}\n")

    print("*** Removing the added book ***")
    remove_book_response = await session.call_tool(
        "remove_book", arguments={"isbn": new_book["isbn"]}
    )
    print(f"Response from remove_book: {remove_book_response.content[0].text}\n")

    print("*** Number of Books in Library after removal ***")
    num_books = await session.call_tool("get_num_books")
    print(f"Number of Books in Library: {num_books.content[0].text}\n")

    print("\n" + "#" * 50 + "\n")

    print("*** Get suggest_random_book prompt ***")
    prompt_response = await session.get_prompt("suggest_random_book", arguments={})
    print(f"Prompt Response: {prompt_response.messages}\n")

    print("*** Get suggest_book_title_by_abstract prompt ***")
    prompt_response = await session.get_prompt(
        "suggest_book_title_by_abstract",
        arguments={"abstract": "A book about the wonders of the universe."},
    )
    print(f"Prompt Response: {prompt_response.messages}\n")

    print("*** Get analyze_book prompt ***")

    prompt_response = await session.get_prompt(
        "analyze_book",
        arguments={
            "book": json.dumps(new_book),
            "query": "What is the main theme of this book?",
        },
    )
    messages = []
    for message in prompt_response.messages:
        messages.append(
            {
                "role": message.role,
                "content": message.content.text,
            }
        )
    print(f"Prompt Response: {json.dumps(messages, indent=2)}\n")


async def test_mcp_server_with_stdio_transport():
    async with streamablehttp_client("http://localhost:8000/mcp") as transport:
        async with ClientSession(transport[0], transport[1]) as session:
            # Initialize the connection
            await session.initialize()

            # Test the book library management MCP server
            await test_book_library_management_mcp_server(session)

            print("\n" + "#" * 50 + "\n")
            print("Test completed successfully!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_mcp_server_with_stdio_transport())
