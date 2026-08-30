import os
import uuid
import logging

import redis
import certifi

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from redis.exceptions import RedisError


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# REDIS TEST
# =========================================================

def test_redis():
    print("\n" + "=" * 60)
    print("REDIS CONNECTION TEST")
    print("=" * 60)

    redis_url = os.getenv("REDIS_URL")

    # -----------------------------------------------------
    # ENV CHECK
    # -----------------------------------------------------

    if not redis_url:
        print("❌ REDIS_URL is missing from .env")
        return False

    print("✅ REDIS_URL found in .env")

    # Don't print the actual URL because it may contain a password.
    if redis_url.startswith("rediss://"):
        print("Protocol: rediss:// (TLS)")
    elif redis_url.startswith("redis://"):
        print("Protocol: redis://")
    else:
        print("⚠️ Unknown Redis URL protocol")

    # -----------------------------------------------------
    # CREATE CLIENT
    # -----------------------------------------------------

    try:

        client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

        print("✅ Redis client created")

    except Exception as error:

        print("❌ Could not create Redis client")
        print(f"Error: {error}")

        return False

    # -----------------------------------------------------
    # PING
    # -----------------------------------------------------

    try:

        response = client.ping()

        if response is True:
            print("✅ Redis PING successful")
        else:
            print("❌ Redis PING returned unexpected response")
            return False

    except RedisError as error:

        print("❌ Redis connection failed")
        print(f"Error: {error}")

        try:
            client.close()
        except Exception:
            pass

        return False

    # -----------------------------------------------------
    # SET / GET TEST
    # -----------------------------------------------------

    test_key = (
        f"agentic:test:"
        f"{uuid.uuid4().hex}"
    )

    test_value = "redis_connection_test"

    try:

        client.set(
            test_key,
            test_value,
            ex=60,
        )

        print("✅ Redis SET successful")

        value = client.get(test_key)

        if value == test_value:

            print("✅ Redis GET successful")
            print("✅ Redis SET/GET data is correct")

        else:

            print("❌ Redis GET returned incorrect value")
            print(f"Expected: {test_value}")
            print(f"Received: {value}")

            return False

        # -------------------------------------------------
        # DELETE
        # -------------------------------------------------

        client.delete(test_key)

        print("✅ Redis DELETE successful")

        # -------------------------------------------------
        # VERIFY DELETE
        # -------------------------------------------------

        deleted_value = client.get(test_key)

        if deleted_value is None:
            print("✅ Redis test key removed correctly")
        else:
            print("⚠️ Redis test key still exists")

    except RedisError as error:

        print("❌ Redis operation failed")
        print(f"Error: {error}")

        return False

    finally:

        try:
            client.close()
        except Exception:
            pass

    print("\n✅ REDIS IS WORKING CORRECTLY")

    return True


# =========================================================
# MONGODB TEST
# =========================================================

def test_mongodb():
    print("\n" + "=" * 60)
    print("MONGODB CONNECTION TEST")
    print("=" * 60)

    mongo_uri = os.getenv("MONGO_URI")

    # -----------------------------------------------------
    # ENV CHECK
    # -----------------------------------------------------

    if not mongo_uri:

        print("❌ MONGO_URI is missing from .env")
        return False

    print("✅ MONGO_URI found in .env")

    # -----------------------------------------------------
    # DATABASE CONFIG
    # -----------------------------------------------------

    db_name = os.getenv(
        "MONGO_DB_NAME",
        "agentic_research"
    )

    collection_name = os.getenv(
        "MONGO_MEMORY_COLLECTION",
        "chat_memories"
    )

    vector_index = os.getenv(
        "MONGO_VECTOR_INDEX",
        "chat_query_vector_index"
    )

    print(f"Database: {db_name}")
    print(f"Collection: {collection_name}")
    print(f"Vector index: {vector_index}")

    # -----------------------------------------------------
    # CONNECT
    # -----------------------------------------------------

    client = None

    try:

        client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            tlsCAFile=certifi.where(),
        )

        print("✅ MongoClient created")

        # -------------------------------------------------
        # PING
        # -------------------------------------------------

        client.admin.command("ping")

        print("✅ MongoDB PING successful")

        # -------------------------------------------------
        # DATABASE ACCESS
        # -------------------------------------------------

        db = client[db_name]

        collection = db[
            collection_name
        ]

        print("✅ Database selected")
        print("✅ Collection selected")

        # -------------------------------------------------
        # WRITE TEST
        # -------------------------------------------------

        test_id = uuid.uuid4().hex

        test_document = {
            "_test_id": test_id,
            "message": "mongodb_connection_test",
        }

        insert_result = collection.insert_one(
            test_document
        )

        print("✅ MongoDB INSERT successful")

        # -------------------------------------------------
        # READ TEST
        # -------------------------------------------------

        found = collection.find_one(
            {
                "_test_id": test_id
            }
        )

        if found:

            print("✅ MongoDB FIND successful")
            print("✅ MongoDB returned correct document")

        else:

            print(
                "❌ MongoDB could not find "
                "the inserted document"
            )

            return False

        # -------------------------------------------------
        # DELETE TEST
        # -------------------------------------------------

        delete_result = collection.delete_one(
            {
                "_test_id": test_id
            }
        )

        if delete_result.deleted_count == 1:

            print("✅ MongoDB DELETE successful")

        else:

            print("⚠️ MongoDB test document was not deleted")

        # -------------------------------------------------
        # LIST COLLECTIONS
        # -------------------------------------------------

        collections = db.list_collection_names()

        print(
            f"✅ Database accessible "
            f"| collections={len(collections)}"
        )

        # -------------------------------------------------
        # CHECK VECTOR INDEX
        # -------------------------------------------------

        print("\nChecking MongoDB vector indexes...")

        try:

            indexes = list(
                collection.list_search_indexes()
            )

            if not indexes:

                print(
                    "⚠️ No MongoDB Search/Vector indexes found"
                )

            else:

                print(
                    f"✅ Found {len(indexes)} "
                    "MongoDB search index(es)"
                )

                for index in indexes:

                    print(
                        f"Index: "
                        f"{index.get('name', 'unknown')}"
                    )

        except Exception as error:

            print(
                "⚠️ Could not inspect vector indexes"
            )

            print(
                f"Details: {error}"
            )

        print("\n✅ MONGODB IS WORKING CORRECTLY")

        return True

    except PyMongoError as error:

        print("❌ MongoDB connection/operation failed")
        print(f"Error: {error}")

        return False

    except Exception as error:

        print("❌ Unexpected MongoDB error")
        print(f"Error: {error}")

        return False

    finally:

        if client:

            try:
                client.close()
            except Exception:
                pass


# =========================================================
# ENVIRONMENT TEST
# =========================================================

def test_environment():

    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLE TEST")
    print("=" * 60)

    redis_url = os.getenv("REDIS_URL")
    mongo_uri = os.getenv("MONGO_URI")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if redis_url:
        print("✅ REDIS_URL found")
    else:
        print("❌ REDIS_URL missing")

    if mongo_uri:
        print("✅ MONGO_URI found")
    else:
        print("❌ MONGO_URI missing")

    if tavily_key:
        print("✅ TAVILY_API_KEY found")
    else:
        print("⚠️ TAVILY_API_KEY missing")

    print(
        "\nSecurity: connection strings and API keys "
        "are NOT printed."
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("       DATABASE CONNECTION TEST SUITE")
    print("=" * 60)

    test_environment()

    redis_ok = test_redis()

    mongo_ok = test_mongodb()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print(
        f"Redis:   {'✅ WORKING' if redis_ok else '❌ FAILED'}"
    )

    print(
        f"MongoDB: {'✅ WORKING' if mongo_ok else '❌ FAILED'}"
    )

    if redis_ok and mongo_ok:

        print(
            "\n🎉 Redis and MongoDB are both working!"
        )

    else:

        print(
            "\n⚠️ One or more database services "
            "need attention."
        )