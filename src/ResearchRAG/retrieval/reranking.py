import logging

from ResearchRAG.config import RERANKER_MODEL_NAME, RERANK_TOP_N

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

logger = logging.getLogger(__name__)

def build_cross_encoder_model(model_name=RERANKER_MODEL_NAME):
    logger.info("Initializing cross-encoder reranker model: %s", model_name)
    try:
        model = HuggingFaceCrossEncoder(model_name=model_name)
        logger.info("Cross-encoder model initialized successfully")
        return model

    except Exception:
        logger.exception("Failed to initialize cross-encoder model: %s", model_name)
        raise


def build_cross_encoder_reranker(model_name=RERANKER_MODEL_NAME, top_n=RERANK_TOP_N):
    logger.info("Building reranker | model=%s | top_n=%d", model_name, top_n)
    try:
        model = build_cross_encoder_model(model_name=model_name)
        reranker = CrossEncoderReranker(model=model, top_n=top_n)
        logger.info("Cross-encoder reranker created successfully")
        return reranker

    except Exception:
        logger.exception("Failed to build cross-encoder reranker")
        raise


def build_rerank_retriever(base_retriever, model_name=RERANKER_MODEL_NAME, top_n=RERANK_TOP_N):
    logger.info("Building rerank retriever")
    try:
        reranker = build_cross_encoder_reranker(
            model_name=model_name,
            top_n=top_n
        )

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )

        logger.info("Rerank retriever created successfully")

        return compression_retriever
    except Exception:
        logger.exception("Failed to build rerank retriever")
        raise