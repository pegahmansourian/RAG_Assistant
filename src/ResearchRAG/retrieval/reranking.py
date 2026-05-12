from ResearchRAG.config import RERANKER_MODEL_NAME, RERANK_TOP_N

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


def build_cross_encoder_model(model_name=RERANKER_MODEL_NAME):
    model = HuggingFaceCrossEncoder(model_name=model_name)
    return model


def build_cross_encoder_reranker(model_name=RERANKER_MODEL_NAME, top_n=RERANK_TOP_N):
    model = build_cross_encoder_model(model_name=model_name)
    reranker = CrossEncoderReranker(model=model, top_n=top_n)
    return reranker


def build_rerank_retriever(base_retriever, model_name=RERANKER_MODEL_NAME, top_n=RERANK_TOP_N):
    reranker = build_cross_encoder_reranker(
        model_name=model_name,
        top_n=top_n
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever
    )

    return compression_retriever