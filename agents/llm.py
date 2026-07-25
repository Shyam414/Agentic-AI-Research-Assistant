import os
from types import SimpleNamespace

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_MODEL_ID = "llama-3.1-8b-instant"

class GroqChatLLM:
    def __init__(self, model_id, api_key, temperature=0, max_new_tokens=1024):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.client = Groq(api_key=api_key)

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return SimpleNamespace(content=response.choices[0].message.content)


def get_llm(temperature=0):
    api_key = os.getenv("GROQ_TOKEN") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_TOKEN or GROQ_API_KEY not found in .env")

    return GroqChatLLM(
        model_id=os.getenv("GROQ_MODEL_ID", DEFAULT_MODEL_ID),
        api_key=api_key,
        temperature=temperature,
        max_new_tokens=int(os.getenv("GROQ_MAX_NEW_TOKENS", "1024")),
    )
