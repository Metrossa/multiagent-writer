# tools/web_search.py
from langchain.tools import BaseTool
from langchain_community.utilities import SerpAPIWrapper
from typing import Optional
from pydantic import Field

class WebSearchTool(BaseTool):
    name: str = "WebSearch"
    description: str = "Searches the web for academic and philosophy-related content."
    search_wrapper: SerpAPIWrapper = Field(default=None)
    
    def __init__(self, serpapi_api_key: str):
        super().__init__()
        self.search_wrapper = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    
    def _run(self, query: str) -> str:
        """Performs a synchronous web search based on the given query."""
        return self.search_wrapper.run(query)
    
    async def _arun(self, query: str) -> str:
        """Asynchronous version (not implemented in this example)."""
        raise NotImplementedError("Async method not implemented")
