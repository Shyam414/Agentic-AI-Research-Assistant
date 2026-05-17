import os
import hashlib
import json
from datetime import datetime, timezone
from math import sqrt

import certifi
import redis
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from redis.exceptions import RedisError

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_CACHE_TTL = 60 * 60 * 24 * 30
DEFAULT_RETRIEVAL_CACHE_TTL = 60 * 10


def _get_redis_client():
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None

    return redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )


def _cache_get_json(key):
    client = _get_redis_client()
    if not client:
        return None

    try:
        value = client.get(key)
        return json.loads(value) if value else None
    except (RedisError, json.JSONDecodeError):
        return None


def _cache_set_json(key, value, ttl):
    client = _get_redis_client()
    if not client:
        return

    try:
        client.setex(key, ttl, json.dumps(value, default=str))
    except (RedisError, TypeError):
        return


def _cache_delete_pattern(pattern):
    client = _get_redis_client()
    if not client:
        return

    try:
        keys = list(client.scan_iter(pattern, count=100))
        if keys:
            client.delete(*keys)
    except RedisError:
        return


def _cache_key(prefix, *parts):
    payload = "||".join(str(part).strip().lower() for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"agentic:{prefix}:{digest}"


def _get_collection():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MONGO_URI not found in .env")

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=10000,
        tlsCAFile=certifi.where(),
    )
    db_name = os.getenv("MONGO_DB_NAME", "agentic_research")
    collection_name = os.getenv("MONGO_MEMORY_COLLECTION", "chat_memories")
    return client, client[db_name][collection_name]


def _normalize_embedding(raw_embedding):
    embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else raw_embedding

    if embedding and isinstance(embedding[0], list):
        dimensions = len(embedding[0])
        embedding = [
            sum(token_vector[index] for token_vector in embedding) / len(embedding)
            for index in range(dimensions)
        ]

    vector_length = sqrt(sum(value * value for value in embedding))
    if vector_length == 0:
        return embedding

    return [float(value / vector_length) for value in embedding]


def embed_text(text):
    embedding_model = os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    cache_key = _cache_key("embedding", embedding_model, text)
    cached_embedding = _cache_get_json(cache_key)
    if cached_embedding:
        return cached_embedding

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env")

    client = InferenceClient(token=token)
    raw_embedding = client.feature_extraction(
        text,
        model=embedding_model,
    )
    embedding = _normalize_embedding(raw_embedding)
    _cache_set_json(
        cache_key,
        embedding,
        int(os.getenv("REDIS_EMBEDDING_TTL_SECONDS", DEFAULT_EMBEDDING_CACHE_TTL)),
    )
    return embedding


def _cosine_similarity(left, right):
    return sum(left_value * right_value for left_value, right_value in zip(left, right))


def _format_memory_results(results):
    if not results:
        return ""

    lines = ["Related previous queries:"]
    for result in results:
        score = result.get("score")
        score_text = f" score={score:.3f}" if isinstance(score, float) else ""
        lines.append(
            f"- Query: {result.get('query', '')}{score_text}\n"
            f"  Summary: {result.get('summary', '')}"
        )
    return "\n".join(lines)


def retrieve_related_queries(query, limit=5):
    cache_key = _cache_key(
        "retrieval",
        os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        os.getenv("MONGO_DB_NAME", "agentic_research"),
        os.getenv("MONGO_MEMORY_COLLECTION", "chat_memories"),
        query,
        limit,
    )
    cached_results = _cache_get_json(cache_key)
    if cached_results is not None:
        return cached_results

    query_embedding = embed_text(query)
    client, collection = _get_collection()

    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": os.getenv("MONGO_VECTOR_INDEX", "chat_query_vector_index"),
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(limit * 20, 20),
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
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        results = list(collection.aggregate(pipeline))
    except OperationFailure:
        memories = collection.find(
            {"embedding": {"$exists": True}},
            {"_id": 0, "query": 1, "summary": 1, "report": 1, "created_at": 1, "embedding": 1},
            limit=200,
        )
        scored_memories = []
        for memory in memories:
            memory["score"] = _cosine_similarity(query_embedding, memory["embedding"])
            memory.pop("embedding", None)
            scored_memories.append(memory)
        results = sorted(scored_memories, key=lambda item: item["score"], reverse=True)[:limit]
    finally:
        client.close()

    _cache_set_json(
        cache_key,
        results,
        int(os.getenv("REDIS_RETRIEVAL_TTL_SECONDS", DEFAULT_RETRIEVAL_CACHE_TTL)),
    )
    return results


def build_semantic_memory_context(query, limit=5):
    try:
        return _format_memory_results(retrieve_related_queries(query, limit=limit))
    except Exception as error:
        return f"Semantic memory unavailable: {error}"


def store_chat_memory(query, summary, report=""):
    embedding = embed_text(query)
    client, collection = _get_collection()

    try:
        collection.create_index("created_at")
        collection.insert_one(
            {
                "query": query,
                "summary": summary,
                "report": report,
                "embedding": embedding,
                "embedding_model": os.getenv("HF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                "created_at": datetime.now(timezone.utc),
            }
        )
        _cache_delete_pattern("agentic:retrieval:*")
    finally:
        client.close()
