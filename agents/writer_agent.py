from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import os

# Updated to use ChatPromptTemplate
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a writer specialized in crafting philosophy papers."),
    ("human", """Using the outline below:
{outline}

And the following research summary:
{research_summary}

Write a comprehensive draft of a philosophy paper. Ensure the writing is scholarly, 
uses proper citations, and adheres to the style guidelines provided in the reference documents.
Address both parts of the prompt: (a)  and (b) it, including relevant citations.""")
])

class WriterAgent:
    def __init__(self, openai_api_key: str = None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview",
            api_key=openai_api_key
        )
        self.chain = LLMChain(llm=self.llm, prompt=writer_prompt)
    
    def write_document(self, outline: str, research_summary: str) -> str:
        """Generates the full draft of the paper."""
        try:
            response = self.chain.invoke({
                "outline": outline,
                "research_summary": research_summary
            })
            
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict):
                return response.get('text', str(response))
            else:
                return str(response)
            
        except Exception as e:
            return f"Error generating document: {str(e)}"
