from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from dotenv import load_dotenv
load_dotenv()

#Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    _client = None
    _db = None
    _resume_collection = None

    @classmethod
    async def connect(cls, embeddings):
        if cls._client is None:
            try:
                uri = os.environ.get("MONGO_URI")
                if uri is None:
                    raise ValueError("MONGO_URI environment variable is not set.")
                print(f"Connecting to MongoDB")

                db_name = "reportDB"
                report_collection_name = "report-collection"

                cls._client = AsyncIOMotorClient(uri, maxPoolSize=50)
                cls._db = cls._client[db_name]
                cls._report_collection = cls._db[report_collection_name]

                logger.info("Connected to MongoDB and initialized database")
            except ConnectionError as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        return cls._report_collection
    
    @classmethod
    async def close(cls):
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            cls._report_collection = None
            logger.info("Disconnected from MongoDB")