from agents.manager_agent import ManagerAgent
from agents.research_agent import ResearchAgent
from agents.drafting_agent import DraftingAgent
from agents.writer_agent import WriterAgent
import os

def main():
    # Add your API keys here
    serpapi_api_key = "serp-api-key"
    openai_api_key = "openai-api-key"
    
    # Initialize agents with API keys
    research_agent = ResearchAgent(
        serpapi_api_key=serpapi_api_key,
        openai_api_key=openai_api_key
    )
    drafting_agent = DraftingAgent(openai_api_key=openai_api_key)
    writer_agent = WriterAgent(openai_api_key=openai_api_key)
    manager_agent = ManagerAgent(research_agent, drafting_agent, writer_agent)
    
    # List your document paths
    document_paths = [
        "docs/Sinner.pdf",
        "docs/book1.pdf"
    ]
    
    # Your detailed philosophy prompt
    detailed_prompt = (
        "insert multiline prompt here"
    )
    
    final_document = manager_agent.handle_request(
        detailed_prompt,
        document_paths=document_paths  # Add your documents here
    )
    print("\n===== FINAL PHILOSOPHY PAPER DRAFT =====\n")
    print(final_document)

if __name__ == "__main__":
    main()
