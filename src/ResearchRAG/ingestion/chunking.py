import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from ResearchRAG.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def create_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return text_splitter

def split_text(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # Split page text into paragraph-like units.

    if not docs:
        logger.warning("No documents provided for chunking")
        return []

    logger.info("Splitting %d documents into chunks", len(docs))
    try:
        splitter = create_splitter(chunk_size, chunk_overlap)
        chunks = splitter.split_documents(docs)
        logger.info("Created %d chunks", len(chunks))
        return chunks

    except Exception:
        logger.exception("Failed to split documents into chunks")
        raise