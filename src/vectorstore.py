from langchain_community.vectorstores import FAISS
from pathlib import Path
from .config import INDEX_DIR

def build_database(chunks, embedding):
    faissdb = FAISS.from_documents(chunks, embedding)

    return faissdb

def save_faiss_index(vectorstore, index_name):
    save_path = Path(INDEX_DIR) / index_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_path))


def load_faiss_index(index_name, embedding_model):
    load_path = Path(INDEX_DIR) / index_name

    vectorstore = FAISS.load_local(
        str(load_path),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore