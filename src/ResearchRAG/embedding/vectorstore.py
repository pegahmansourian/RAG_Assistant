from langchain_community.vectorstores import FAISS
from pathlib import Path
from ResearchRAG.config import INDEX_DIR

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


def update_faiss_index(index_name, new_chunks, embedding_model):
    vectorstore = load_faiss_index(index_name, embedding_model)
    vectorstore.add_documents(new_chunks)
    save_faiss_index(vectorstore, index_name)
    return vectorstore

def delete_from_faiss_index(index_name, doc_to_delete, embedding_model):
    vectorstore = load_faiss_index(index_name, embedding_model)

    ids_to_delete = []

    for docstore_id, doc in vectorstore.docstore._dict.items():
        source = doc.metadata.get("source")
        if Path(source).name == Path(doc_to_delete).name:
            ids_to_delete.append(docstore_id)

    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
        save_faiss_index(vectorstore, index_name)

    return vectorstore