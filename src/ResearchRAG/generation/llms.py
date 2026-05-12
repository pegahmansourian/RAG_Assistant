import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import ollama

from ResearchRAG.config import ROOT_DIR, LLM_MODELS, DEFAULT_LLM_KEY

load_dotenv(ROOT_DIR / ".env")

def is_model_available(model_name):
    models_response = ollama.list()
    models = models_response.get("models", [])

    for model in models:
        if model.get("model") == model_name or model.get("name") == model_name:
            return True

    return False


def ensure_model_available(model_name):
    if not is_model_available(model_name):
        ollama.pull(model_name)

def get_llm_config(llm_key=None):
    if llm_key is None:
        model_key = DEFAULT_LLM_KEY

    if llm_key not in LLM_MODELS:
        raise ValueError(
            f"Unknown llm model key: {llm_key}. "
            f"Available keys: {list(LLM_MODELS.keys())}"
        )

    return LLM_MODELS[llm_key]

def build_llm(llm_key=None, temperature=0):
    model_config = get_llm_config(llm_key)

    provider = model_config["provider"]
    model_name = model_config["model_name"]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key
        )

    if provider == "cohere":
        api_key = os.getenv("COHERE_API_KEY")
        return ChatCohere(
            model=model_name,
            temperature=temperature,
            cohere_api_key=api_key
        )

    if provider == "ollama":
        ensure_model_available(model_name)
        return ChatOllama(
            model=model_name,
            temperature=temperature
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")