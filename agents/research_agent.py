from langchain.agents import AgentType, initialize_agent
from langchain_openai import ChatOpenAI
from tools.web_search import WebSearchTool
from tools.doc_parser import DocumentParserTool
import os

class ResearchAgent:
    def __init__(self, serpapi_api_key: str, openai_api_key: str = None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview",
            api_key=openai_api_key
        )
        
        self.tools = [
            WebSearchTool(serpapi_api_key=serpapi_api_key),
            DocumentParserTool(openai_api_key=openai_api_key)
        ]
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def process_documents(self, document_paths: list[str]) -> str:
        """Process and summarize multiple documents."""
        try:
            summaries = []
            for path in document_paths:
                # Check if file exists
                if not os.path.exists(path):
                    summaries.append(f"Warning: Document not found: {path}")
                    continue
                    
                doc_summary = self.tools[1]._run(path)
                if "Error parsing document" in doc_summary:
                    summaries.append(f"Warning: Failed to parse {path}: {doc_summary}")
                else:
                    summaries.append(f"Summary of {path}:\n{doc_summary}")
            
            return "\n\n".join(summaries)
        except Exception as e:
            return f"Error processing documents: {str(e)}"
    
    def run(self, query: str, document_paths: list[str] = None) -> str:
        """Executes a research query, incorporating both web search and document analysis."""
        try:
            # First, process any provided documents
            document_context = ""
            if document_paths:
                document_context = self.process_documents(document_paths)
                print(f"Document Analysis Results:\n{document_context}\n")
            
            # Combine document insights with web search
            if document_context:
                query = f"""
                Based on the following document analysis:
                {document_context}
                
                And considering additional web research, please {query}
                
                Important: Use the document analysis as your primary source, supplemented by web research.
                """
            
            result = self.agent.run(query)
            return result
        except Exception as e:
            return f"Error in research process: {str(e)}"
