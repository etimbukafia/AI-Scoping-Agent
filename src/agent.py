import os
import logging
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from tools import chat_with_vectorstore, write_to_markdown
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from server import configs
from typing import Any, Optional, Dict, List
from pydantic import Field
from langchain_core.messages import SystemMessage, HumanMessage
import sys
from functools import partial

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_debug.log"),
        logging.StreamHandler()  # This sends output to console/stderr
    ]
)

# Qdrant connection setup
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)


class Agent:
    def __init__(self, memory=None, llm=None, embeddings=None, projectDoc=[]):
        # Initialize dependencies at runtime.
        self.memory = memory if memory is not None else MemorySaver()
        self.llm = llm if llm is not None else configs.get_llm()
        self.embeddings = embeddings if embeddings is not None else configs.get_embeddings()
        self.toolList = [chat_with_vectorstore]
        self.tools = ToolNode(self.toolList)
        self.projectDoc = projectDoc
        self.system_message_template = (
            "# Security Expert Instructions\n\n"
            "You are a security expert with deep knowledge of web3 security and smart contract auditing. "
            "You assist in auditing and finding vulnerabilities in smart contracts. "
            "Always provide concise, helpful responses.\n\n"
            "You will use the protocol's documentation as context when answering questions: \n{project_doc}\n\n"
            "# Retrieved Context\n\n"
            "{docs_content}"
        )
    
    def __del__(self):
        """Clean up resources when agent is destroyed"""
        try:
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.close()
        except Exception as e:
            logging.warning(f"Error closing Qdrant client: {e}")

class State(MessagesState):
    project_doc: Optional[str] = Field(default=None)


def query_or_respond(state: State):
    """Generate tool call for retrieval or respond."""
    llm = configs.get_llm()
    llm_with_tools = llm.bind_tools([chat_with_vectorstore])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


def process_protocol(doc_file):
    """
    Processes the protocol document.
    `doc_file` is a file-like object (e.g., from a file opened in binary mode).
    Uses the write_to_markdown tool to extract and explain the document.
    """
    doc_bytes = doc_file.read()  # Read the file content into bytes
    # The write_to_markdown tool both writes to file and returns the markdown explanation.
    project_doc, markdown_response = write_to_markdown({"documentation": doc_bytes})
    logging.info(f"Project doc processed. Length: {len(project_doc) if isinstance(project_doc, list) else len(str(project_doc))}")
    return project_doc, markdown_response

def generate(state: State):
    """Generate answer."""
    try:
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]
        print(f"Found {len(tool_messages)} tool messages", flush=True)
        sys.stdout.flush()

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages) if tool_messages else "No relevant documents found."
        print(f"Docs content length: {len(docs_content)}", flush=True)
        sys.stdout.flush()

        project_doc = state.project_doc or "No project documentation provided."

        system_message_content = (
            "# Security Expert Instructions\n\n"
            "You are a security expert with deep knowledge of web3 security and smart contract auditing. "
            "You assist in auditing and finding vulnerabilities in smart contracts. "
            "Always provide concise, helpful responses.\n\n"
            "# Protocol Documentation\n\n"
            f"{project_doc}\n\n"
            "# Retrieved Context\n\n"
            f"{docs_content}"
        )

        print(f"System message created successfully. Length: {len(system_message_content)}", flush=True)
        sys.stdout.flush()
        print(f"System message preview: {system_message_content[:200]}...", flush=True)
        sys.stdout.flush()
    
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages

        # Run
        llm = configs.get_llm()
        response = llm.invoke(prompt)
        return {"messages": [response]}
    except Exception as e:
        logging.error(f"Error in generate: {e}")
        # Return a graceful error message
        return {"messages": [SystemMessage(content=f"An error occurred while generating a response: {str(e)}")]}
    

def response_from_ai(agent: Agent, user_input: str, project_doc: str) -> Dict[str, str]:   
    try:

        # Build the graph with the custom state
        graph_builder = StateGraph(State)

        """
        def bound_query_or_respond(state):
            return query_or_respond(state, agent)

        def bound_generate(state):
            return generate(state, agent)
        """

        graph_builder.add_node(query_or_respond)
        graph_builder.add_node(agent.tools)
        graph_builder.add_node(generate)

        graph_builder.set_entry_point("query_or_respond")
        graph_builder.add_conditional_edges(
            "query_or_respond",
            tools_condition,
            {END: END, "tools": "tools"},
        )
        graph_builder.add_edge("tools", "generate")
        graph_builder.add_edge("generate", END)

        graph = graph_builder.compile(checkpointer=agent.memory)
        config = {"configurable": {"thread_id": "001"}}

        # Initialize the state with both messages and project_doc
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "project_doc": project_doc
        }

        # Invoke the graph with the initial state
        resp = graph.invoke(initial_state, config=config)
        
        # Safe extraction of response content
        last_message = resp.get("messages", [])[-1] if resp.get("messages") else None
        response_content = last_message.content if last_message and hasattr(last_message, "content") else "No response generated."
        
        return {"ai_response": response_content}
    except Exception as e:
        logging.error(f"Error in response_from_ai: {e}")
        return {"ai_response": f"An error occurred: {str(e)}. Please try again."}

async def initialize_agent():
    """
    Asynchronously initialize Agent instance.
    """
    try:
        return Agent()
    except Exception as e:
        logging.error(f"Error initializing agent: {e}")
        raise
