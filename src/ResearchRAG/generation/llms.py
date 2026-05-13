import os
import logging

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import ollama

from ResearchRAG.config import ROOT_DIR, LLM_MODELS, DEFAULT_LLM_KEY

logger = logging.getLogger(__name__)

load_dotenv(ROOT_DIR / ".env")

def ensure_model_available(model_name):
    logger.info("Checking Ollama model availability: %s", model_name)

    try:
        models_response = ollama.list()
        models = models_response.get("models", [])

        for model in models:
            if model.get("model") == model_name:
                logger.info("Ollama model found: %s", model_name)
                return

            if model.get("name") == model_name:
                logger.info("Ollama model found: %s", model_name)
                return

        logger.warning("Ollama model not found: %s", model_name)
        logger.info("Pulling Ollama model: %s", model_name)
        ollama.pull(model_name)
        logger.info("Successfully pulled Ollama model: %s", model_name)

    except Exception:
        logger.exception("Failed to ensure Ollama model availability: %s", model_name)
        raise

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

    logger.info("Initializing LLM | provider=%s | model=%s", provider, model_name)

    try:
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
        logger.error("Unsupported LLM provider: %s", provider)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    except Exception:
        logger.exception("Failed to initialize LLM: %s", model_name)
        raise