import logging
from langchain_community.vectorstores import FAISS
from pathlib import Path
from ResearchRAG.config import INDEX_DIR

logger = logging.getLogger(__name__)

def build_database(chunks, embedding):
    try:
        faissdb = FAISS.from_documents(chunks, embedding)
        logger.info("FAISS database created successfully")
        return faissdb

    except Exception:
        logger.exception("Failed to build FAISS database")
        raise

def save_faiss_index(vectorstore, index_name):
    save_path = Path(INDEX_DIR) / index_name
    logger.info("Saving FAISS index to %s", save_path)

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_path))
        logger.info("FAISS index saved successfully")

    except Exception:
        logger.exception("Failed to save FAISS index: %s", index_name)
        raise


def load_faiss_index(index_name, embedding_model):
    load_path = Path(INDEX_DIR) / index_name
    logger.info("Loading FAISS index from %s", load_path)

    try:
        vectorstore = FAISS.load_local(
            str(load_path),
            embedding_model,
            allow_dangerous_deserialization=True
        )

        logger.info("FAISS index loaded successfully")

        return vectorstore

    except Exception:
        logger.exception("Failed to load FAISS index: %s", index_name)
        raise


def update_faiss_index(index_name, new_chunks, embedding_model):
    logger.info("Updating FAISS index with %d new chunks", len(new_chunks))
    try:
        vectorstore = load_faiss_index(index_name, embedding_model)
        vectorstore.add_documents(new_chunks)
        save_faiss_index(vectorstore, index_name)
        logger.info("FAISS index updated successfully")
        return vectorstore

    except Exception:
        logger.exception("Failed to update FAISS index: %s", index_name)
        raise


def delete_from_faiss_index(index_name, doc_to_delete, embedding_model):
    logger.info("Deleting document from FAISS index: %s", doc_to_delete)

    try:
        vectorstore = load_faiss_index(index_name, embedding_model)
        ids_to_delete = []

        for docstore_id, doc in vectorstore.docstore._dict.items():
            source = doc.metadata.get("source")
            if Path(source).name == Path(doc_to_delete).name:
                ids_to_delete.append(docstore_id)

        if ids_to_delete:
            logger.info("Deleting %d chunks from FAISS index", len(ids_to_delete))
            vectorstore.delete(ids=ids_to_delete)
            save_faiss_index(vectorstore, index_name)
            logger.info("Document deleted successfully")

        else:
            logger.warning("No matching chunks found for %s",doc_to_delete)

        return vectorstore

    except Exception:
        logger.exception(
            "Failed to delete document from FAISS index: %s",
            doc_to_delete
        )
        raise