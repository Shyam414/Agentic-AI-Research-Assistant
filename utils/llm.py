import logging
import os
import time
from math import sqrt
from types import SimpleNamespace
import ollama
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:v1.5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MAX_RETRIES = 2


def get_chat_model_name():
    model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    if not model:
        raise ValueError("OLLAMA_MODEL cannot be empty.")
    return model


def get_embedding_model_name():
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL).strip()
    if not model:
        raise ValueError("OLLAMA_EMBEDDING_MODEL cannot be empty.")
    return model


def _max_tokens():
    value = os.getenv("OLLAMA_MAX_NEW_TOKENS", str(DEFAULT_MAX_TOKENS))
    try:
        value = int(value)
    except ValueError as error:
        raise ValueError("OLLAMA_MAX_NEW_TOKENS must be an integer.") from error
    if value <= 0:
        raise ValueError("OLLAMA_MAX_NEW_TOKENS must be greater than 0.")
    return value


def _max_retries():
    value = os.getenv("OLLAMA_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
    try:
        value = int(value)
    except ValueError as error:
        raise ValueError("OLLAMA_MAX_RETRIES must be an integer.") from error
    if value < 0:
        raise ValueError("OLLAMA_MAX_RETRIES cannot be negative.")
    return value


def check_ollama_connection():
    try:
        models = ollama.list()
        logger.info("Ollama connection successful.")
        return models
    except Exception as error:
        raise RuntimeError("Cannot connect to Ollama. Make sure Ollama is running.") from error


def _get_available_models(models):
    available_models = []

    if hasattr(models, "models"):
        models = models.models
    elif isinstance(models, dict):
        models = models.get("models", [])
    else:
        models = []

    for model in models:
        if hasattr(model, "model"):
            name = model.model
        elif isinstance(model, dict):
            name = model.get("name") or model.get("model")
        else:
            name = None

        if name:
            available_models.append(name)

    return available_models


def check_model_available(model_name, models=None):
    try:
        if models is None:
            models = ollama.list()

        available_models = _get_available_models(models)

        if model_name in available_models:
            return True

        return any(available == model_name or available.startswith(f"{model_name}:") for available in available_models)
    except Exception as error:
        logger.warning("Could not check Ollama model: %s", error)
        return False


def _normalize_embedding(raw_embedding):
    embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else raw_embedding

    if not embedding:
        raise RuntimeError("Ollama returned an empty embedding.")

    if isinstance(embedding[0], list):
        dimensions = len(embedding[0])
        if dimensions == 0:
            raise RuntimeError("Embedding has zero dimensions.")
        embedding = [sum(vector[index] for vector in embedding) / len(embedding) for index in range(dimensions)]

    try:
        embedding = [float(value) for value in embedding]
    except (TypeError, ValueError) as error:
        raise RuntimeError("Embedding contains invalid values.") from error

    magnitude = sqrt(sum(value * value for value in embedding))
    if magnitude == 0:
        raise RuntimeError("Embedding has zero magnitude.")

    return [value / magnitude for value in embedding]


def _extract_embeddings(response):
    if hasattr(response, "embeddings"):
        return response.embeddings
    if isinstance(response, dict):
        return response.get("embeddings")
    return None


def embed_text(text):
    text = str(text or "").strip()
    if not text:
        raise ValueError("Text to embed cannot be empty.")

    model_name = get_embedding_model_name()
    logger.info("Embedding request started | model=%s | text_length=%d", model_name, len(text))

    models = check_ollama_connection()
    if not check_model_available(model_name, models):
        raise RuntimeError(f"Ollama embedding model '{model_name}' is not available locally. Run: ollama pull {model_name}")

    try:
        response = ollama.embed(model=model_name, input=text)
        embeddings = _extract_embeddings(response)

        if not embeddings:
            raise RuntimeError("Ollama returned no embeddings.")

        embedding = _normalize_embedding(embeddings[0])
        logger.info("Embedding created successfully | dimensions=%d", len(embedding))
        return embedding
    except Exception as error:
        logger.exception("Embedding generation failed: %s", error)
        raise


class OllamaLLM:
    def __init__(self, temperature=0, max_new_tokens=DEFAULT_MAX_TOKENS):
        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number.")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        self.model_id = get_chat_model_name()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = _max_retries()

        models = check_ollama_connection()
        if not check_model_available(self.model_id, models):
            raise RuntimeError(f"Ollama chat model '{self.model_id}' is not available locally. Run: ollama pull {self.model_id}")

        logger.info(
            "OllamaLLM initialized | model=%s | temperature=%s | max_tokens=%d | retries=%d",
            self.model_id,
            self.temperature,
            self.max_new_tokens,
            self.max_retries,
        )

    def invoke(self, prompt, json_mode=False):
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if not isinstance(json_mode, bool):
            raise TypeError("json_mode must be a boolean.")

        logger.info("LLM request started | model=%s | prompt_length=%d", self.model_id, len(prompt))

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = self._complete(prompt, json_mode=json_mode)
                if not result:
                    raise RuntimeError("Ollama returned an empty response.")

                result = str(result).strip()
                logger.info("LLM request successful | response_length=%d", len(result))
                return SimpleNamespace(content=result)
            except Exception as error:
                last_error = error
                logger.warning(
                    "LLM request failed | attempt=%d/%d | error=%s",
                    attempt + 1,
                    self.max_retries + 1,
                    error,
                )

                if attempt < self.max_retries:
                    delay = 2 ** attempt
                    logger.info("Retrying in %d seconds...", delay)
                    time.sleep(delay)

        raise RuntimeError(f"Ollama LLM request failed after {self.max_retries + 1} attempts.") from last_error

    def _complete(self, prompt, json_mode=False):
        request = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": self.temperature, "num_predict": self.max_new_tokens},
        }

        if json_mode:
            request["format"] = "json"

        response = ollama.chat(**request)

        if hasattr(response, "message"):
            return response.message.content
        if isinstance(response, dict):
            return response.get("message", {}).get("content")
        return None


def get_llm(temperature=0):
    return OllamaLLM(temperature=temperature, max_new_tokens=_max_tokens())