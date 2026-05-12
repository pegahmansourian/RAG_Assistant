from ResearchRAG.config import RETRIEVAL_K


def build_similarity_retriever(vectorstore, k=RETRIEVAL_K):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def build_mmr_retriever(vectorstore, k=RETRIEVAL_K, fetch_k=20):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k
        }
    )
    return retriever


def retrieve_documents(query, retriever):
    documents = retriever.invoke(query)
    return documents