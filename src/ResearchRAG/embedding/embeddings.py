from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
import numpy as np
from ResearchRAG.config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_KEY
import logging

logger = logging.getLogger(__name__)

def get_embedding_config(model_key=None):
    if model_key is None:
        model_key = DEFAULT_EMBEDDING_KEY

    logger.info("Loading embedding config for key: %s", model_key)

    if model_key not in EMBEDDING_MODELS:
        logger.error("Unknown embedding model key: %s", model_key)
        raise ValueError(
            f"Unknown embedding model key: {model_key}. "
            f"Available keys: {list(EMBEDDING_MODELS.keys())}"
        )

    return EMBEDDING_MODELS[model_key]

def build_embedding_model(model_key=None):
    model_config = get_embedding_config(model_key)

    provider = model_config["provider"]
    model_name = model_config["model_name"]

    logger.info("Initializing embedding model | provider=%s | model=%s",provider,model_name,)

    try:
        if provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=model_name)

        if provider == "cohere":
            return CohereEmbeddings(model=model_name)

        logger.error("Unsupported embedding provider: %s", provider)
        raise ValueError(f"Unsupported embedding provider: {provider}")

    except Exception:
        logger.exception("Failed to initialize embedding model: %s", model_name)
        raise ValueError(f"Failed to initialize embedding model: {model_name}")