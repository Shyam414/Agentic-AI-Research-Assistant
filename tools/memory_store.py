import logging
import os
from datetime import datetime, timezone
import certifi
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from utils.llm import embed_text as create_embedding, get_embedding_model_name

load_dotenv()

MONGO_SERVER_TIMEOUT = 5000
DEFAULT_SIMILARITY_THRESHOLD = 0.90

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

def _close_client(client):
    if client:
        try:
            client.close()
        except Exception:
            pass

def _get_collection():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MongoDB is DOWN")
    client = None
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=MONGO_SERVER_TIMEOUT, connectTimeoutMS=MONGO_SERVER_TIMEOUT, tlsCAFile=certifi.where())
        client.admin.command("ping")
        db_name = os.getenv("MONGO_DB_NAME", "agentic_research")
        collection_name = os.getenv("MONGO_MEMORY_COLLECTION", "chat_memories")
        return client, client[db_name][collection_name]
    except Exception as error:
        _close_client(client)
        logger.error("MongoDB is DOWN")
        raise RuntimeError("MongoDB is DOWN") from error

def get_embedding(text):
    return create_embedding(text)

def _validate_query(query):
    if query is None:
        raise ValueError("Query cannot be None.")
    query = str(query).strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    return query

def _get_similarity_threshold():
    return float(os.getenv("MONGO_SIMILARITY_THRESHOLD", DEFAULT_SIMILARITY_THRESHOLD))

def _build_vector_pipeline(query_embedding):
    return [
        {
            "$vectorSearch": {
                "index": os.getenv("MONGO_VECTOR_INDEX", "chat_query_vector_index"),
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": int(os.getenv("MONGO_VECTOR_NUM_CANDIDATES", "50")),
                "limit": 1
            }
        },
        {
            "$project": {
                "_id": 1,
                "query": 1,
                "summary": 1,
                "report": 1,
                "created_at": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

def find_similar_memory(query):
    query = _validate_query(query)
    logger.info("MongoDB vector search started | query_length=%d", len(query))
    query_embedding = get_embedding(query)
    logger.info("Query embedding created | dimensions=%d", len(query_embedding))
    client = None
    try:
        client, collection = _get_collection()
        results = list(collection.aggregate(_build_vector_pipeline(query_embedding)))
        logger.info("MongoDB vector search completed | results=%d", len(results))
        if not results:
            logger.info("MongoDB memory MISS | no matching documents found")
            return None
        result = results[0]
        score = float(result.get("score", 0.0))
        logger.info("MongoDB best match | score=%.3f | query=%r", score, result.get("query", ""))
        return result
    except PyMongoError as error:
        logger.error("MongoDB is DOWN")
        raise RuntimeError("MongoDB is DOWN") from error
    finally:
        _close_client(client)

def retrieve_matching_memory(query):
    result = find_similar_memory(query)
    if not result:
        return None
    score = float(result.get("score", 0.0))
    threshold = _get_similarity_threshold()
    logger.info("MongoDB similarity check | score=%.3f | threshold=%.3f", score, threshold)
    if score < threshold:
        logger.info("MongoDB memory MISS | similarity=%.1f%%", score * 100)
        return None
    logger.info("MongoDB memory MATCH | similarity=%.1f%%", score * 100)
    return result

def build_semantic_memory_context(query):
    try:
        result = retrieve_matching_memory(query)
        if not result:
            return ""
        score = float(result.get("score", 0.0))
        return f"Matched previous query: {result.get('query', '')} score={score:.3f}\nSummary: {result.get('summary', '')}\nReport: {result.get('report', '')}"
    except RuntimeError as error:
        if str(error) == "MongoDB is DOWN":
            logger.error("MongoDB is DOWN")
            return "MongoDB is DOWN. Semantic memory unavailable."
        raise

def store_chat_memory(query, summary, report=""):
    if not query:
        raise ValueError("Memory query cannot be empty.")
    if not summary:
        raise ValueError("Memory summary cannot be empty.")
    embedding = get_embedding(query)
    client = None
    try:
        client, collection = _get_collection()
        collection.create_index("created_at")
        collection.insert_one({
            "query": query,
            "summary": summary,
            "report": report,
            "embedding": embedding,
            "embedding_model": get_embedding_model_name(),
            "created_at": datetime.now(timezone.utc)
        })
        logger.info("Chat memory stored successfully")
    except PyMongoError as error:
        logger.error("MongoDB is DOWN")
        raise RuntimeError("MongoDB is DOWN") from error
    finally:
        _close_client(client)

def get_memory(query):
    return retrieve_matching_memory(query)

def _check_mongodb_health():
    client = None
    if not os.getenv("MONGO_URI"):
        return {"available": False, "status": "MongoDB is DOWN"}
    try:
        client = MongoClient(os.getenv("MONGO_URI"), serverSelectionTimeoutMS=MONGO_SERVER_TIMEOUT, connectTimeoutMS=MONGO_SERVER_TIMEOUT, tlsCAFile=certifi.where())
        client.admin.command("ping")
        return {"available": True, "status": "MongoDB is UP"}
    except Exception:
        return {"available": False, "status": "MongoDB is DOWN"}
    finally:
        _close_client(client)

def check_database_health():
    return {"mongodb": _check_mongodb_health()}

if __name__ == "__main__":
    status = check_database_health()
    print(f"MongoDB: {'UP' if status['mongodb']['available'] else 'DOWN'}")