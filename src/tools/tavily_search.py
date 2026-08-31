
import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from ..utils.redaction import redact_sensitive_text


load_dotenv()



class SearchResult(BaseModel):
    """A single search result from Tavily."""
    title: str = Field(description="Title of the page")
    url: str = Field(description="URL of the page")
    content: str = Field(description="Extracted relevant content")
    score: float = Field(description="Relevance score (0-1)")


class SearchResponse(BaseModel):
    """Complete response from a Tavily search."""
    query: str = Field(description="The search query used")
    results: list[SearchResult] = Field(default_factory=list)
    answer: Optional[str] = Field(
        default=None,
        description="AI-generated answer summarizing results (if available)"
    )



class TavilySearchTool:
    """
    Web search tool using Tavily API.
    
    Tavily is designed specifically for AI agents:
    - Returns clean, extracted content
    - Filters out ads and irrelevant info
    - Provides relevance scoring
    - Optional AI-generated summaries
    
    Usage:
        tool = TavilySearchTool()
        results = tool.search("Python ModuleNotFoundError fix")
        for result in results.results:
            print(result.title, result.content[:100])
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Tavily client."""
        configured_key = str(api_key or os.getenv("TAVILY_API_KEY") or "").strip()
        if not configured_key:
            raise ValueError(
                "Tavily web research is not configured. Set TAVILY_API_KEY or continue without web research."
            )

        self.client = TavilyClient(api_key=configured_key)
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        include_answer: bool = True
    ) -> SearchResponse:
        """
        Perform a web search using Tavily.
        
        Args:
            query: The search query
            max_results: Maximum number of results to return (1-10)
            search_depth: "basic" (faster) or "advanced" (more thorough)
            include_answer: Whether to include AI-generated summary
            
        Returns:
            SearchResponse with results and optional answer
        """
        query = redact_sensitive_text(query, limit=500)
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=include_answer,
                timeout=30,
            )
            
            results = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", "No title"),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0.0)
                ))
            
            search_response = SearchResponse(
                query=query,
                results=results,
                answer=response.get("answer")
            )
            
            return search_response

        except Exception:
            return SearchResponse(query=query, results=[], answer=None)
    
    def search_multiple(
        self,
        queries: list[str],
        max_results_per_query: int = 3
    ) -> list[SearchResponse]:
        """
        Perform multiple searches and return all results.
        Useful when we have multiple research queries from triage.
        
        Args:
            queries: List of search queries
            max_results_per_query: Results per query
            
        Returns:
            List of SearchResponse objects
        """
        all_responses = []
        for query in queries:
            response = self.search(
                query=query,
                max_results=max_results_per_query,
                search_depth="basic"
            )
            all_responses.append(response)
        
        return all_responses
