#from db_connect import Database
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
import os
from dotenv import load_dotenv
from mistralai import Mistral
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import logging

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
# Qdrant connection setup
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)

class Configs:
    def __init__(self):
        self.model = None
        self.embeddings = None
        self.vector_store = None
        self.client = None
        self.mistral_model = None
        self.qdrant_client = None
        self.llm = None
        #self.report_collection = None

    async def initialize(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #self.report_collection = await Database.connect(self.embeddings)
        self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
        self.client = Mistral(api_key=api_key)
        self.mistral_model = "mistral-small-latest"
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="security_reports",
            embedding=self.embeddings,
            content_payload_key="chunk_text"
        )

    def get_client(self):
        return self.client, self.mistral_model
    
    def get_llm(self):
        return self.llm

    def get_embeddings(self):
        return self.embeddings

    def get_vector_store(self):
        return self.qdrant_client, self.vector_store
    
    #def get_report_collection(self):
        #return self.report_collection

configs = Configs()

async def initialize_configs():
    await configs.initialize()
    logging.info("configs initialized")