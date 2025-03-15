import os
import sys
import logging
import asyncio
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Union, Tuple, Literal, Sequence
from typing_extensions import TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from server import configs, initialize_configs 
from langchain_core.tools import tool
from utils import analyze_doc, extract_metadata, get_similar_reports
from filehandlers import extract_text_from_doc
from dotenv import load_dotenv
import logging
from pydantic import BaseModel, Field
from typing import Union, Annotated
from functools import partial
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import format_security_report
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_debug.log"),
        logging.StreamHandler()  # output to console/stderr
    ]
)

# Qdrant connection setup
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

class ScopingAgent:

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self,
                 memory: Optional[Any] = None,
                 llm: Optional[Any] = None,
                 embeddings: Optional[Any] = None,
                 projectDoc: Optional[List[Any]] = None,
                 config_module=None):
        """
        Initialize the complete agent with all required components.
        
        Args:
            memory: Memory saver for conversation state
            llm: Language model for generating responses
            embeddings: Embeddings model for vector operations
            projectDoc: Project documentation context
            config_module: Module containing configuration functions
        """
        # Import configuration module dynamically if provided
        self.configs = configs

        # Initialize dependencies at runtime
        self.memory = memory if memory is not None else MemorySaver()
        self.llm = llm 
        self.embeddings = embeddings
        self.projectDoc = projectDoc if projectDoc is not None else ""
        #Prompts setup
        self.system_message_template = PromptTemplate(
            template = """
            You are a security expert with deep knowledge of web3 security and smart contract auditing. 
            You assist in auditing and identifying vulnerabilities in smart contracts. 
            Always provide concise, helpful, and evidence-based responses.
            Protocol Documentation: {project_doc}
            User Question: {question}
            If the question is asking about specific vulnerabilities, past security incidents, concrete examples of attacks, or security trends that would require retrieving historical data, choose 'tools'.
            If the question is about general security practices, concepts that could be answered with the project documentation, theoretical aspects of security, or is asking for explanations that don't require specific examples from past reports, choose 'direct'.
            
            Make your decision:""",
            
            input_variables=["question", "project_doc"],
        )

        self.chat_with_vectorstore_prompt = (
            "You are a seasoned security expert specializing in web3 security and smart contract auditing."
            "The context a comprehensive vector store of security reports, which include detailed risk classifications, vulnerabilities, and technical assessments."
            "When you receive a question, provide a precise, evidence-based answer that directly references the retrieved context."
            "Your answer should be structured clearly (using bullet points, headers, or numbered lists as needed) and avoid generic or vague responses."
            "Question: {question}"
            "Context: {context}"
            "Answer:"
        )

        # Setup Qdrant client and vector store
        #self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        #self.vector_store = None
        self.retriever = None

        # Setting up tools
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            "retrieve_security_reports",
            "Searches the vector store and return information from security reports pertaining to the user's question"
            
        )

        self.tools = [self.retriever_tool]
        
        # Initialize the graph  
        self._build_graph()

    def _get_llm(self):
        """Get language model from configs or provide fallback."""
        if hasattr(self, 'configs') and self.configs and hasattr(self.configs, 'get_llm'):
            return self.configs.get_llm()
        else:
            return init_chat_model("mistral-large-latest", model_provider="mistralai")

    def _get_embeddings(self):
        """Get embeddings model from configs or provide fallback."""
        if hasattr(self, 'configs') and self.configs and hasattr(self.configs, 'get_embeddings'):
            return self.configs.get_embeddings()
        else:
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _build_graph(self):
        """Build the LangGraph state graph."""
        self.graph_builder = StateGraph(self.AgentState)
        
        # Add nodes with bound methods
        self.graph_builder.add_node("query_or_respond", partial(self.query_or_respond))
        self.retrieve = ToolNode(self.tools)
        self.graph_builder.add_node("retrieve", self.retrieve)
        self.graph_builder.add_node("rewrite", partial(self.rewrite))
        self.graph_builder.add_node("generate", partial(self.generate))
        self.graph_builder.add_node("direct_answer", partial(self.direct_answer))

        self.graph_builder.add_edge(START, "query_or_respond")
        
        self.graph_builder.add_conditional_edges(
            "query_or_respond",
            lambda state: state.get("next", "direct"),
            {
                "tools": "retrieve",
                "direct": "direct_answer"
            },
        )

        self.graph_builder.add_conditional_edges(
            "retrieve",
            lambda state: state.get("next", "generate"),
            {
                "generate": "generate",
                "rewrite": "rewrite"
            }
        )
        
        self.graph_builder.add_edge("generate", END)
        self.graph_builder.add_edge("direct_answer", END)
        self.graph_builder.add_edge("rewrite", "query_or_respond")
        
        # Compile the graph
        self.graph = self.graph_builder.compile(checkpointer=self.memory)

    def should_use_tools(self, state) -> Dict[str, str]:
        """
        Determines whether to use tools or answer directly using protocol docs.
        """
        print("----CHECKING IF QUESTION NEEDS RETRIEVAL------")
        
        class decision(BaseModel):
            choice: str = Field(description="Choice between 'tools' or 'direct'")
            reasoning: str = Field(description="Reasoning for the decision")

        llm_with_tool = self.llm.with_structured_output(decision)

        # Get the user question
        messages = state["messages"]
        if len(messages) > 0:
            question = messages[0].content if hasattr(messages[0], 'content') else ""
        else:
             return {"next": "direct"}  # Default if no messages
            
        # Create prompt to decide if we need to retrieve documents
        prompt = self.system_message_template

        # Get project doc or use empty string if not available
        project_doc = self.projectDoc if self.projectDoc else "No project documentation available."
        
        # Invoke the decision chain
        result = prompt.format(question=question, project_doc=project_doc) | llm_with_tool
        decision_result = result.invoke({})
        
        print(f"Decision: {decision_result.choice} - {decision_result.reasoning}")
        return {"next": decision_result.choice}

    def grade_documents(self, state) ->  Dict[str, str]:
        """
        For determining if the retrieved documents are relevant to the question.

        Args:
            state (messages): The current state

        Returns:
            str: A decision for whether the documents are relevant or not
        """

        print("----CHECKING DOCUMENT RELEVANCY------")

        class grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes or 'no")

        llm_with_tool = self.llm.with_structured_output(grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        chain = prompt | llm_with_tool

        messages = state["messages"]

        question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                question = msg.content
                break
    
        if not question:
            # Fallback to first message if no human message found
            question = messages[0].content if messages else ""

        last_message = messages[-1]

        docs = last_message.content if hasattr(last_message, 'content') else ""

        scored_result = chain.invoke({"question": question, "context": docs})
        score = scored_result.binary_score

        if score == 'yes':
            print("---DECISION: DOCS RELEVANT -----")
            return {"next": "generate"}
        
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(score)
            return {"next": "rewrite"}


    def query_or_respond(self, state) -> Dict[str, Any]:
        """Generate tool call for retrieval or respond."""
        print("-----CALL AGENT------")
        try:
            messages = state["messages"]
            model = self.llm.bind_tools(self.tools)
            response = model.invoke(messages)         
            return {"messages": [response]}
        except Exception as e:
            logging.error(f"Error in query_or_respond: {e}")
            return {"messages": [SystemMessage(content=f"An error occurred: {str(e)}")]}
        
    def rewrite(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # Grader
        model = self.llm
        response = model.invoke(msg)
        return {"messages": [response]}
    

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt_template = PromptTemplate(
            template=self.chat_with_vectorstore_prompt,
            input_variables=["question", "context"]
        )

        # Chain
        rag_chain = prompt_template | self.llm | StrOutputParser()

        # Run   
        response = rag_chain.invoke({"context": docs, "question": question})
        return {"messages": [response]}
    
    def direct_answer(self, state):
        """
        Answer the question directly using the protocol documentation without retrieval.
        
        Args:
            state (messages): The current state
            
        Returns:
            dict: The updated state with the answer
        """
        print("---DIRECT ANSWER WITHOUT RETRIEVAL---")
        
        messages = state["messages"]
        # Find the original human question
        question = ""
        for msg in messages:
            if hasattr(msg, 'type') and msg.type == 'human':
                question = msg.content
                break
        
        if not question:
            # Fallback to first message if no human message found
            question = messages[0].content if messages else ""
        
        # Get project documentation
        project_doc = self.projectDoc if self.projectDoc else "No project documentation available."
        
        # Format system message with project doc but no retrieved context
        system_message = self.system_message_template.format(
            project_doc=project_doc,
            question=question
        )
        
        # Create prompt with the system message and user question
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=question)
        ]
        
        # Generate response using the LLM
        response = self.llm.invoke(messages)
        
        # Return the response
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}

    async def process_protocol(self, documentation: Annotated[Union[str, bytes], "The document content to process"]) -> Tuple[Any, str]:
        """
        Processes the protocol document.
        `doc_file` is a file-like object (opened in binary mode).
        Uses the write_to_markdown tool to extract and explain the document.
        """
        if isinstance(documentation, dict):
            doc_content = documentation.get("documentation")
        else:
            doc_content = documentation

        doc_bytes = doc_content.read()  # Read file content as bytes
        # Call write_to_markdown with a dictionary input
        doc_text = extract_text_from_doc(doc_bytes)
        client, model = self.configs.get_client()
        markdown_response = await analyze_doc(client, model, doc_text)
        metadata = await extract_metadata(client, model, doc_text)
        similar_reports = get_similar_reports(self.qdrant_client, metadata)
        with open("files/report.md", "w", encoding="utf-8") as output:
            output.write(markdown_response)
        # Store the processed project doc
    
        return similar_reports, doc_text, markdown_response

    def response_from_ai(self, initial_state: Dict) -> Dict[str, str]:
        """
        Get a response from the AI using the configured graph.
        
        Args:
            initial_state: Dictionary containing the initial state with messages
            
        Returns:
            Dictionary with AI response
        """
        try:
            config = {"configurable": {"thread_id": "001"}}

            # Invoke the graph with the initial state
            # Ensure initial_state is a dictionary containing a key "messages" with the starting messages
            if "messages" not in initial_state:
                raise ValueError("Initial state must contain a 'messages' key")
            
            # Ensure all messages are BaseMessage instances
            for i, msg in enumerate(initial_state["messages"]):
                if isinstance(msg, dict):
                    # Convert dict to appropriate message type
                    if msg.get("type") == "human":
                        initial_state["messages"][i] = HumanMessage(content=msg.get("content", ""))
                    elif msg.get("type") == "ai":
                        initial_state["messages"][i] = AIMessage(content=msg.get("content", ""))
                    elif msg.get("type") == "system":
                        initial_state["messages"][i] = SystemMessage(content=msg.get("content", ""))

            print(f"Starting graph with {len(initial_state['messages'])} messages")

             # Add a default next field if not present
            if "next" not in initial_state:
                initial_state["next"] = "direct"  # Default routing path

            resp = self.graph.invoke(initial_state, config=config)

            # Debug output
            print(f"Graph response type: {type(resp)}")
            print(f"Graph response keys: {resp.keys() if isinstance(resp, dict) else 'Not a dict'}")
                
            # Safely extract response content
            if isinstance(resp, dict) and "messages" in resp:
                messages = resp["messages"]
                # Find the last AI message
                for msg in reversed(messages):
                    if hasattr(msg, 'type') and msg.type == 'ai':
                        return {"ai_response": msg.content}
                
                # If no AI message found, take the last message content
                last_message = messages[-1] if messages else None
                response_content = last_message.content if last_message and hasattr(last_message, "content") else "No response generated."
                
                return {"ai_response": response_content}
            else:
                return {"ai_response": "Error: Invalid response format from graph"}
        except Exception as e:
            logging.error(f"Error in response_from_ai: {e}")
            return {"ai_response": f"An error occurred: {str(e)}. Please try again."}

    async def initialize(self):
        """
        Asynchronously initialize any resources that require async initialization.
        """
        try:
            await initialize_configs()
            # Refresh LLM and embeddings after config initialization
            self.llm = self.configs.get_llm()
            self.embeddings = self.configs.get_embeddings()
            # Rebuild vector store with new embeddings if needed
            self.qdrant_client, self.vector_store = self.configs.get_vector_store()
            self.retriever = self.vector_store.as_retriever()
            # Rebuild the graph
            self._build_graph()
        except Exception as e:
            logging.error(f"Error in async initialization: {e}")
            raise

    def __del__(self):
        """Clean up resources when agent is destroyed."""
        try:
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.close()
        except Exception as e:
            logging.warning(f"Error closing Qdrant client: {e}")

    @classmethod
    async def create(cls, config_module=None):
        """
        Factory method to create and initialize an agent asynchronously.
        
        Returns:
            Initialized CompleteAgent instance
        """
        agent = cls(config_module=config_module)
        await agent.initialize()
        return agent