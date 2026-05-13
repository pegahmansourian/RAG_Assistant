import logging

from ResearchRAG.config import RETRIEVAL_K

logger = logging.getLogger(__name__)

def build_retriever(vectorstore, search_type="mmr", k=RETRIEVAL_K, fetch_k=20):
    logger.info("Building retriever | type=%s | k=%d", search_type, k)

    try:
        search_kwargs = {"k": k}

        if search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k

        retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        logger.info("Retriever created successfully")

        return retriever

    except Exception:
        logger.exception("Failed to build retriever | type=%s", search_type)
        raise

def retrieve_documents(query, retriever):
    logger.info("Retrieving documents")
    try:
        documents = retriever.invoke(query)
        logger.info("Retrieved %d documents", len(documents))
        return documents

    except Exception:
        logger.exception("Failed to retrieve documents")
        raise