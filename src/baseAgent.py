import logging
from datetime import datetime
from typing import Optional, List, Any, Sequence, Annotated, TypedDict, Union, Tuple
from src.config import Configs
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from src.utils import analyze_doc, extract_metadata, get_similar_reports
from src.filehandlers import extract_text_from_doc
from langgraph.prebuilt import create_react_agent
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

class BaseScopingAgent():
    """Base class for the scoping agent"""
    def __init__(self, 
                
                 llm = None, 
                 embedding_model=None, 
                 mistral_client = None,
                 qdrant_client = None,
                 vectorstore = None,
                 state = AgentState(messages=[]),
                 protocol_doc: str = None,
                 similar_reports: List[str] = None):
        
        self.llm = llm
        self.embedding_model = embedding_model
        self.mistral_client = mistral_client
        self.qdrant_client = qdrant_client
        self.vectorstore = vectorstore
        self.state = state

        #Protocol documentation
        self.protocol_doc = protocol_doc if protocol_doc is not None else ""
        self.similar_reports = similar_reports

    async def process_protocol(self, documentation: Annotated[Union[str, bytes], "The document content to process"]) -> Tuple[Any, str, str]:
        """
        Processes the protocol document.
        `doc_file` is a file-like object (opened in binary mode).
        Uses the write_to_markdown tool to extract and explain the document.
        """
        # If a string is passed and it's a valid file path, read from the file.
        if isinstance(documentation, str) and os.path.exists(documentation):
            with open(documentation, "rb") as f:
                doc_bytes = f.read()
        elif isinstance(documentation, (str, bytes)):
            doc_bytes = documentation if isinstance(documentation, bytes) else documentation.encode()
        else:
            # Assume it's a file-like object
            doc_bytes = documentation.read()

        # Call write_to_markdown with a dictionary input
        doc_text = extract_text_from_doc(doc_bytes)

        if not self.mistral_client:
            raise ValueError("Mistral client is not initialized.")
        client, model = self.mistral_client
        
        markdown_response = await analyze_doc(client, model, doc_text)
        metadata = await extract_metadata(client, model, doc_text)
        similar_reports = get_similar_reports(self.qdrant_client, metadata)
        with open("files/report.md", "w", encoding="utf-8") as output:
            output.write(markdown_response)
        # Store the processed project doc
    
        return similar_reports, doc_text, markdown_response
    
    def close(self):
        try:
            if self.qdrant_client:
                self.qdrant_client.close()
                print("Qdrant client closed successfully.")
        except Exception as e:
            print(f"Error closing Qdrant client: {e}")
    