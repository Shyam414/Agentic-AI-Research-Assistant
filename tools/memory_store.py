import os
import hashlib
import json
import logging
from datetime import datetime, timezone
import certifi
import redis

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError, OperationFailure
from redis.exceptions import RedisError

from utils.llm import (
    embed_text as create_embedding,
    get_embedding_model_name,
)

load_dotenv()

DEFAULT_EMBEDDING_CACHE_TTL = 60 * 60 * 24 * 30
DEFAULT_RETRIEVAL_CACHE_TTL = 60 * 10

REDIS_CONNECT_TIMEOUT = 2
REDIS_SOCKET_TIMEOUT = 2
MONGO_SERVER_TIMEOUT = 5000


logging.basicConfig(
    level=os.getenv("LOG_LEVEL","INFO").upper(),
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)

# REDIS
def _get_redis_client():

    redis_url = os.getenv("REDIS_URL")

    if not redis_url:
        return None

    try:

        return redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT,
            socket_timeout=REDIS_SOCKET_TIMEOUT,
        )

    except Exception:

        logger.error(
            "⚠️ Redis is DOWN"
        )

        return None


def is_redis_available():

    client = _get_redis_client()

    if not client:

        return False

    try:

        client.ping()

        return True

    except RedisError:

        return False

    finally:

        try:
            client.close()
        except Exception:
            pass


def get_redis_status():

    if not os.getenv("REDIS_URL"):

        return {
            "available": False,
            "status": "Redis is DOWN",
        }

    if is_redis_available():

        return {
            "available": True,
            "status": "Redis is UP",
        }

    return {
        "available": False,
        "status": "Redis is DOWN",
    }

# REDIS CACHE GET
def _cache_get_json(key):

    client = _get_redis_client()

    if not client:
        return None

    try:

        value = client.get(key)

        if not value:
            return None

        return json.loads(value)

    except RedisError:

        logger.warning(
            "⚠️ Redis is DOWN"
        )

        return None

    except json.JSONDecodeError:

        logger.warning(
            "⚠️ Redis cache contains invalid data"
        )

        return None

    finally:

        try:
            client.close()
        except Exception:
            pass

# REDIS CACHE SET
def _cache_set_json(key,value,ttl):
    client = _get_redis_client()
    if not client:
        return False

    try:
        client.setex(
            key,
            ttl,
            json.dumps(value,default=str),
        )

        logger.debug(
            "Redis cache SET successful"
        )

        return True

    except RedisError:

        logger.warning(
            "⚠️ Redis is DOWN"
        )

        return False

    except TypeError:

        logger.warning(
            "⚠️ Redis cache serialization failed"
        )

        return False

    finally:

        try:
            client.close()
        except Exception:
            pass

# REDIS DELETE
def _cache_delete_pattern(pattern):

    client = _get_redis_client()

    if not client:
        return False

    try:

        keys = list(
            client.scan_iter(
                pattern,
                count=100,
            )
        )

        if keys:

            client.delete(
                *keys
            )

        return True

    except RedisError:

        logger.warning(
            "⚠️ Redis is DOWN"
        )

        return False

    finally:

        try:
            client.close()
        except Exception:
            pass

# CACHE KEY
def _cache_key(prefix,*parts):

    payload = "||".join(
        str(part).strip().lower()
        for part in parts
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return (
        f"agentic:"
        f"{prefix}:"
        f"{digest}"
    )

# MONGODB CONNECTION
def _get_collection():

    mongo_uri = os.getenv(
        "MONGO_URI"
    )

    if not mongo_uri:

        raise RuntimeError(
            "MongoDB is DOWN"
        )

    try:

        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=MONGO_SERVER_TIMEOUT,
            connectTimeoutMS=MONGO_SERVER_TIMEOUT,
            tlsCAFile=certifi.where(),
        )

        # Force connection immediately.
        client.admin.command(
            "ping"
        )

        db_name = os.getenv(
            "MONGO_DB_NAME",
            "agentic_research",
        )

        collection_name = os.getenv(
            "MONGO_MEMORY_COLLECTION",
            "chat_memories",
        )

        collection = client[
            db_name
        ][
            collection_name
        ]

        return client, collection

    except Exception:

        logger.error(
            "⚠️ MongoDB is DOWN"
        )

        raise RuntimeError(
            "MongoDB is DOWN"
        )

# EMBEDDING CACHE
def get_cached_embedding(text):

    model_name = get_embedding_model_name()

    cache_key = _cache_key(
        "embedding",
        model_name,
        text,
    )

    cached_embedding = _cache_get_json(cache_key)

    if cached_embedding:
        return cached_embedding
    embedding = create_embedding(
        text
    )

    _cache_set_json(
        cache_key,
        embedding,
        int(
            os.getenv(
                "REDIS_EMBEDDING_TTL_SECONDS",
                DEFAULT_EMBEDDING_CACHE_TTL,
            )
        ),
    )

    return embedding

# COSINE SIMILARITY
def _cosine_similarity(left,right):
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        return 0.0

    return sum(
        left_value * right_value
        for left_value, right_value
        in zip(
            left,
            right,
        )
    )

# FORMAT MEMORY RESULTS
def _format_memory_results(results):

    if not results:
        return ""
    lines = ["Related previous queries:"]

    for result in results:
        score = result.get( "score")
        if isinstance(score,(float, int)):
            score_text = (f" score={score:.3f}")
        else:
            score_text = ""
        lines.append(
            f"- Query: "
            f"{result.get('query', '')}"
            f"{score_text}\n"
            f"  Summary: "
            f"{result.get('summary', '')}"
        )

    return "\n".join(
        lines
    )

# MONGODB VECTOR RETRIEVAL
def retrieve_related_queries(query,limit=5):
    if query is None:
        raise ValueError("Query cannot be None.")
    query = str(query).strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    embedding_model = (get_embedding_model_name())
    db_name = os.getenv("MONGO_DB_NAME","agentic_research",)
    collection_name = os.getenv("MONGO_MEMORY_COLLECTION","chat_memories",)
    cache_key = _cache_key(
        "retrieval",
        embedding_model,
        db_name,
        collection_name,
        query,
        limit,
    )
    cached_results = _cache_get_json(cache_key)
    if cached_results is not None:
        logger.info("Retrieval cache HIT")
        return cached_results
    logger.info("Retrieval cache MISS")
    query_embedding = (get_cached_embedding(query))
    # MONGODB CONNECTION
    try:
        client, collection = (_get_collection())
    except RuntimeError:
        # Do NOT expose DNS / PyMongo traceback.
        raise
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": os.getenv(
                        "MONGO_VECTOR_INDEX",
                        "chat_query_vector_index",
                    ),
                    "path": "embedding",
                    "queryVector":
                        query_embedding,
                    "numCandidates":
                        max(
                            limit * 20,
                            20,
                        ),
                    "limit": limit,
                }
            },

            {
                "$project": {
                    "_id": 0,
                    "query": 1,
                    "summary": 1,
                    "report": 1,
                    "created_at": 1,
                    "score": {
                        "$meta":
                            "vectorSearchScore"
                    },
                }
            },
        ]

        results = list(
            collection.aggregate(
                pipeline
            )
        )
        logger.info( "MongoDB vector search returned %d results",len(results),)

    except OperationFailure:
        logger.warning(
            "MongoDB vector search unavailable; "
            "using cosine similarity fallback"
        )

        memories = collection.find(
            {
                "embedding": {
                    "$exists": True
                }
            },
            {
                "_id": 0,
                "query": 1,
                "summary": 1,
                "report": 1,
                "created_at": 1,
                "embedding": 1,
            },
            limit=200,
        )

        scored_memories = []

        for memory in memories:

            memory_embedding = (
                memory.get(
                    "embedding"
                )
            )

            if not memory_embedding:
                continue

            if len(memory_embedding) != len(
                query_embedding
            ):
                continue

            memory["score"] = (
                _cosine_similarity(
                    query_embedding,
                    memory_embedding,
                )
            )

            memory.pop(
                "embedding",
                None,
            )

            scored_memories.append(
                memory
            )

        results = sorted(
            scored_memories,
            key=lambda item:
                item["score"],
            reverse=True,
        )[:limit]

    except PyMongoError:

        logger.error(
            "⚠️ MongoDB is DOWN"
        )

        raise RuntimeError(
            "MongoDB is DOWN"
        )

    finally:

        client.close()

    # CACHE RETRIEVAL
    retrieval_ttl = int(
        os.getenv(
            "REDIS_RETRIEVAL_TTL_SECONDS",
            DEFAULT_RETRIEVAL_CACHE_TTL,
        )
    )

    _cache_set_json(
        cache_key,
        results,
        retrieval_ttl,
    )

    return results

# SEMANTIC MEMORY CONTEXT
def build_semantic_memory_context(query,limit=5,):
    try:
        results = (
            retrieve_related_queries(
                query,
                limit=limit,
            )
        )
        return _format_memory_results(results)

    except RuntimeError as error:
        # Only show our clean status.
        if str(error) == "MongoDB is DOWN":
            logger.error("⚠️ MongoDB is DOWN")
            return (
                "MongoDB is DOWN. "
                "Semantic memory unavailable."
            )
        raise
    except Exception:

        logger.error(
            "⚠️ MongoDB is DOWN"
        )
        return (
            "MongoDB is DOWN. "
            "Semantic memory unavailable."
        )


# STORE CHAT MEMORY
def store_chat_memory(query,summary,report="",):
    if not query:
        raise ValueError("Memory query cannot be empty.")
    if not summary:
        raise ValueError("Memory summary cannot be empty.")
    embedding = (get_cached_embedding(query))

    try:
        client, collection = (_get_collection())
    except RuntimeError:
        raise
    try:
        collection.create_index("created_at")
        collection.insert_one(
            {
                "query": query,
                "summary": summary,
                "report": report,
                "embedding": embedding,
                "embedding_model":
                    get_embedding_model_name(),
                "created_at":
                    datetime.now(
                        timezone.utc
                    ),
            }
        )
        logger.info("Chat memory stored successfully")
        _cache_delete_pattern("agentic:retrieval:*")

    except PyMongoError:
        logger.error("⚠️ MongoDB is DOWN")
        raise RuntimeError("MongoDB is DOWN")
    finally:
        client.close()


# DATABASE HEALTH CHECK
def check_database_health():
    redis_status = get_redis_status()
    mongo_status = {"available": False,"status": "MongoDB is DOWN",}
    client = None
    if os.getenv("MONGO_URI"):
        try:
            client = MongoClient(
                os.getenv("MONGO_URI"),
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                tlsCAFile=certifi.where(),
            )
            client.admin.command("ping")
            mongo_status = {
                "available": True,
                "status": "MongoDB is UP",
            }
        except Exception:
            mongo_status = {
                "available": False,
                "status": "MongoDB is DOWN",
            }

        finally:
            if client:

                try:
                    client.close()
                except Exception:
                    pass
    return {"redis": redis_status,"mongodb": mongo_status,}

# MAIN HEALTH CHECK
if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("DATABASE HEALTH CHECK")
    print("=" * 50)

    status = check_database_health()

    if status["redis"]["available"]:
        print("Redis:   ✅ UP")
    else:
        print("Redis:   ❌ DOWN")

    if status["mongodb"]["available"]:
        print("MongoDB: ✅ UP")
    else:
        print("MongoDB: ❌ DOWN")

    print("=" * 50)
