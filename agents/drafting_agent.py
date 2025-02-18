from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

# A prompt tailored for drafting philosophy paper outlines.
draft_prompt = PromptTemplate(
    input_variables=["topic", "research_summary"],
    template=(
        "You are an expert drafting assistant for philosophy papers. Given the topic '{topic}' "
        "and the following research summary:\n\n"
        "{research_summary}\n\n"
        "Create a detailed and structured outline. The outline should include sections such as "
        "Introduction, Background, Main Argument, Counterarguments, and Conclusion. "
        "Include notes on key philosophical arguments and citations where relevant."
    )
)

class DraftingAgent:
    def __init__(self, openai_api_key: str = None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        self.chain = LLMChain(llm=self.llm, prompt=draft_prompt)
    
    def create_outline(self, topic: str, research_summary: str) -> str:
        """
        Generates an outline based on the topic and collected research.
        """
        result = self.chain.invoke({"topic": topic, "research_summary": research_summary})
        outline = result if isinstance(result, str) else result.get(self.chain.output_key, result)
        return outline
