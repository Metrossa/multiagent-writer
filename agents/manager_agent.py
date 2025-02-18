class ManagerAgent:
    def __init__(self, research_agent, drafting_agent, writer_agent):
        self.research_agent = research_agent
        self.drafting_agent = drafting_agent
        self.writer_agent = writer_agent

    def handle_request(self, detailed_prompt: str, document_paths: list[str] = None) -> str:
        """
        Coordinates the complete process for the philosophy paper.
        """
        try:
            # Extract topic from the prompt text
            topic = detailed_prompt.split(":")[0].strip()
            
            # Step 1: Research with document processing
            print("\n=== Starting Research Phase ===")
            research_query = f"Analyze the following topic: {topic}"
            research_results = self.research_agent.run(
                query=research_query,
                document_paths=document_paths
            )
            
            if not research_results or "error" in research_results.lower():
                raise Exception(f"Research phase failed: {research_results}")
            
            # Step 2: Drafting      
            print("\n=== Starting Drafting Phase ===")
            prompt_details = (
                "insert multiline prompt here"
            )
            
            outline = self.drafting_agent.create_outline(
                topic=topic, 
                research_summary=research_results + "\n" + prompt_details
            )
            
            if not outline or "error" in outline.lower():
                raise Exception(f"Drafting phase failed: {outline}")
            
            # Step 3: Writing
            print("\n=== Starting Writing Phase ===")
            document = self.writer_agent.write_document(
                outline=outline,
                research_summary=research_results + "\n" + prompt_details
            )
            
            if not document or "error" in document.lower():
                raise Exception(f"Writing phase failed: {document}")
            
            return document
            
        except Exception as e:
            error_msg = f"Error in paper generation process: {str(e)}"
            print(f"\nERROR: {error_msg}")
            return error_msg
