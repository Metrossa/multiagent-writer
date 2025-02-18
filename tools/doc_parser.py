# tools/doc_parser.py
from langchain.tools import BaseTool
from typing import Optional
import PyPDF2
import docx  # pip install python-docx
import os
from utils.text_processing import chunk_text, summarize_chunks
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel
import traceback

class DocumentParserTool(BaseTool):
    name: str = "DocumentParser"
    description: str = "Parses text from user-provided documents (PDFs, TXT files, DOCX files, etc.)."
    llm: Optional[ChatOpenAI] = Field(default=None)
    
    def __init__(self, openai_api_key: str = None):
        super().__init__()
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo-preview",
            api_key=openai_api_key
        )
    
    def _run(self, file_path: str) -> str:
        """Extracts and summarizes text from various document types."""
        try:
            # Extract full text
            raw_text = self._extract_text(file_path)
            
            if "Error" in raw_text:
                return raw_text
            
            # Chunk and summarize
            print(f"Extracting text from {file_path}...")
            chunks = chunk_text(raw_text)
            print(f"Created {len(chunks)} chunks from document")
            
            summary = summarize_chunks(chunks, self.llm)
            if not summary:
                return f"Error: Failed to generate summary for {file_path}"
            
            return summary
                
        except Exception as e:
            return f"Error parsing document: {str(e)}\nStack trace: {traceback.format_exc()}"
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from document based on file type."""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._parse_pdf(file_path)
        elif file_extension == '.txt':
            return self._parse_txt(file_path)
        elif file_extension == '.docx':
            return self._parse_docx(file_path)
        else:
            return f"Unsupported file type: {file_extension}"
    
    def _parse_pdf(self, file_path: str) -> str:
        try:
            text = ""
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                if len(reader.pages) == 0:
                    return "Error: PDF file appears to be empty"
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if not text.strip():
                return "Error: No text could be extracted from PDF"
            return text.strip()
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"
    
    def _parse_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            if not text:
                return "Error: Text file appears to be empty"
            return text
        except UnicodeDecodeError:
            # Try different encodings if utf-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read().strip()
            except Exception as e:
                return f"Error parsing text file: {str(e)}"
        except Exception as e:
            return f"Error parsing text file: {str(e)}"
    
    def _parse_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    async def _arun(self, file_path: str) -> str:
        """Asynchronous version (not implemented in this example)."""
        raise NotImplementedError("Async method not implemented")
