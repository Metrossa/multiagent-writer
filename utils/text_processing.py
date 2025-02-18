from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at summarizing philosophical texts."),
    ("human", """Summarize the following text chunk about Augustine's philosophy, focusing on key concepts about evil, free will, and God:

{chunk}

Provide a concise summary:""")
])

def chunk_text(text: str, chunk_size: int = 3000, chunk_overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks of specified size."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

def summarize_chunks(chunks: List[str], llm) -> str:
    """Summarize a list of text chunks using the provided LLM."""
    summary_chain = LLMChain(llm=llm, prompt=SUMMARY_PROMPT)
    summaries = []
    
    for chunk in chunks:
        try:
            response = summary_chain.invoke({"chunk": chunk})
            # Handle different response types
            if hasattr(response, 'content'):
                summary = response.content
            elif isinstance(response, dict):
                summary = response.get('text', response.get('output', str(response)))
            else:
                summary = str(response)
            
            if summary and not isinstance(summary, str):
                summary = str(summary)
            
            summaries.append(summary)
            
        except Exception as e:
            print(f"Error summarizing chunk: {str(e)}")
            continue
    
    if not summaries:
        return "Error: Could not generate any summaries from the document"
    
    combined_summary = "\n\n".join(summaries)
    
    # If combined summary is still too long, summarize it again
    if len(combined_summary) > 6000:
        try:
            response = summary_chain.invoke({"chunk": combined_summary})
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict):
                return response.get('text', response.get('output', str(response)))
            else:
                return str(response)
        except Exception as e:
            print(f"Error in final summarization: {str(e)}")
            return combined_summary[:6000] + "..."
    
    return combined_summary 