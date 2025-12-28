"""
MCP Server for Swiss Legal Research Tools

This server exposes Swiss legal research capabilities via MCP protocol:
- Primary law search (Fedlex, admin.ch)
- Case law search (BGer, Federal Court decisions)
- General web search

Run with: python mcp_server.py
"""

import os
import json
import asyncio
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import Tavily
from tavily import TavilyClient

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
if tavily_api_key.startswith("tvly-tvly-"):
    # Fix double prefix if present
    tavily_api_key = tavily_api_key.replace("tvly-tvly-", "tvly-")

tavily = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


def search_swiss_primary_law(query: str, max_results: int = 5) -> dict:
    """Search Swiss primary law sources (Fedlex, admin.ch)"""
    if not tavily:
        return {"error": "Tavily API key not configured"}
    
    search_query = f"{query} site:fedlex.admin.ch OR site:admin.ch Bundesrecht Gesetz"
    
    try:
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            include_domains=["fedlex.admin.ch", "admin.ch"],
            search_depth="advanced"
        )
        return {
            "query": query,
            "source": "Swiss Primary Law (Fedlex/admin.ch)",
            "results": results.get("results", [])
        }
    except Exception as e:
        return {"error": str(e)}


def search_swiss_case_law(query: str, max_results: int = 5) -> dict:
    """Search Swiss case law (Federal Court BGE decisions)"""
    if not tavily:
        return {"error": "Tavily API key not configured"}
    
    search_query = f"{query} site:bger.ch BGE Bundesgericht Urteil"
    
    try:
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            include_domains=["bger.ch"],
            search_depth="advanced"
        )
        return {
            "query": query,
            "source": "Swiss Case Law (BGer)",
            "results": results.get("results", [])
        }
    except Exception as e:
        return {"error": str(e)}


def search_swiss_legal_commentary(query: str, max_results: int = 5) -> dict:
    """Search Swiss legal commentary and doctrine"""
    if not tavily:
        return {"error": "Tavily API key not configured"}
    
    search_query = f"{query} Schweizer Recht Kommentar Doktrin"
    
    try:
        results = tavily.search(
            query=search_query,
            max_results=max_results,
            search_depth="advanced"
        )
        return {
            "query": query,
            "source": "Swiss Legal Commentary",
            "results": results.get("results", [])
        }
    except Exception as e:
        return {"error": str(e)}


# Create MCP Server
server = Server("swiss-legal-research")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Swiss legal research tools"""
    return [
        Tool(
            name="search_swiss_primary_law",
            description="""Search Swiss primary law sources including Fedlex and admin.ch.
            Use this to find: Federal laws, ordinances, constitutional provisions.
            Returns: Article numbers (Art.), paragraphs (Abs.), SR numbers.
            Example queries: "Arbeitsrecht Kündigungsfrist", "OR Art. 335".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Legal search query in German, French, or Italian"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_swiss_case_law",
            description="""Search Swiss Federal Court (Bundesgericht) case law.
            Use this to find: BGE decisions, court precedents, judicial interpretations.
            Returns: BGE references (volume/page), case numbers, key holdings.
            Example queries: "Fristlose Kündigung BGE", "Mietrecht Kündigung".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Case law search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_swiss_legal_commentary",
            description="""Search Swiss legal commentary and doctrine.
            Use this to find: Academic analysis, legal interpretations, scholarly opinions.
            Returns: Commentary excerpts, doctrinal positions, academic references.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Commentary search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a Swiss legal research tool"""
    
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)
    
    if name == "search_swiss_primary_law":
        result = search_swiss_primary_law(query, max_results)
    elif name == "search_swiss_case_law":
        result = search_swiss_case_law(query, max_results)
    elif name == "search_swiss_legal_commentary":
        result = search_swiss_legal_commentary(query, max_results)
    else:
        result = {"error": f"Unknown tool: {name}"}
    
    return [TextContent(
        type="text",
        text=json.dumps(result, indent=2, ensure_ascii=False)
    )]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
