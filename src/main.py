import logging
from baseAgent import BaseScopingAgent
from baseAgent import AgentState
from vectorstoreAgent import VectorstoreAgent
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from config import initialize_configs, configs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentOrchestrator:
    """
    Wraps an agent instance and builds a workflow graph as an instance attribute.
    """
    def __init__(self, agent: VectorstoreAgent):
        self.agent = agent
        self.workflow_graph = self._build_graph()

    @property
    def similar_reports(self):
        return self.agent.similar_reports

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        
        # Define nodes using the agent's instance methods and attributes
        workflow.add_node("agent", self.agent.agent)
        retrieve = ToolNode([self.agent.retriever_tool])
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite", self.agent.rewrite)
        workflow.add_node("generate", self.agent.generate)
        workflow.add_edge(START, "agent")

        workflow.add_conditional_edges(
            "agent",
            # Use the provided condition to determine routing
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            # Use the agent's grading method to decide next step
            self.agent.grade_documents,
        )

        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        return workflow.compile()

async def initialize(path):
    """
    Initialize agent components and create an AgentOrchestrator instance.
    """
    try:
        await initialize_configs()
        llm = configs.get_llm()
        embedding_model = configs.get_embeddings()
        mistral_client = configs.get_client()
        qdrant_client, vectorstore = configs.get_vector_store()

        # Create an instance of the vectorstore agent.
        vectorstore_agent = VectorstoreAgent(
            llm=llm,
            embedding_model=embedding_model,
            mistral_client=mistral_client,
            qdrant_client=qdrant_client,
            vectorstore=vectorstore
        )
        
        vectorstore_agent.vectorstore = vectorstore

        similar_reports, protocol_doc, markdown_response = await vectorstore_agent.process_protocol(path)
        
        # Save the outputs as instance attributes
        vectorstore_agent.similar_reports = similar_reports
        vectorstore_agent.protocol_doc = protocol_doc
        # Build the orchestrator with the agent instance so the graph becomes an instance attribute
        orchestrator = AgentOrchestrator(vectorstore_agent)
        
        logging.info("Configurations initialized successfully.")
        return orchestrator
    except Exception as e:
        logging.error(f"Error in async initialization: {e}")
        raise


async def create_agent(path):
    """
    Factory method to create and initialize an agent orchestrator asynchronously.
    
    Returns:
        Initialized AgentOrchestrator instance
    """
    orchestrator = await initialize(path)
    return orchestrator

