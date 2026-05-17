import os
from types import SimpleNamespace

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()


class HuggingFaceChatLLM:
    def __init__(self, model_id, token, temperature=0, max_new_tokens=1024):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.client = InferenceClient(token=token)

    def invoke(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return SimpleNamespace(content=response.choices[0].message.content)


def get_llm(temperature=0):
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN not found in .env")

    return HuggingFaceChatLLM(
        model_id=os.getenv(
            "HF_MODEL_ID",
            "meta-llama/Llama-3.2-1B-Instruct",
        ),
        token=token,
        temperature=temperature,
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "1024")),
    )
