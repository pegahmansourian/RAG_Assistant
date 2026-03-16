import streamlit as st

from src.loaders import parse_pdf_folder
from src.chunking import split_text
from src.embeddings import build_embedding_model
from src.vectorstore import build_database, load_faiss_index, save_faiss_index
from src.retriever import build_similarity_retriever
from src.rag_chain import run_rag
from src.config import RAW_PDF_DIR
from src.llms import build_llm

from langchain_openai import ChatOpenAI


st.set_page_config(page_title="Technical PDF RAG Assistant", layout="wide")

st.title("Technical PDF RAG Assistant")
st.write("Ask questions over your ML and security PDF collection.")

st.sidebar.header("Settings")

embedding_key = st.sidebar.selectbox(
    "Embedding model",
    ["miniLM", "mpnet", "bge_base", "cohere_v3"]
)

llm_key = st.sidebar.selectbox(
    "LLM model name",
    options=["mistral", "miniLM", "cohere", "llama3", "qwen3"]
)

index_name = st.sidebar.text_input(
    "Index name",
    value=f"faiss_{embedding_key}"
)

rebuild_index = st.sidebar.button("Rebuild Index")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


def initialize_pipeline():
    embedding_model = build_embedding_model(embedding_key)

    if rebuild_index:
        documents = parse_pdf_folder(RAW_PDF_DIR)
        chunked_documents = split_text(documents)

        vectorstore = build_database(chunked_documents, embedding_model)
        save_faiss_index(vectorstore, index_name)
    else:
        try:
            vectorstore = load_faiss_index(index_name, embedding_model)
        except Exception:
            documents = parse_pdf_folder(RAW_PDF_DIR)
            chunked_documents = split_text(documents)

            vectorstore = build_database(chunked_documents, embedding_model)
            save_faiss_index(vectorstore, index_name)

    retriever = build_similarity_retriever(vectorstore)

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever


if st.session_state.retriever is None or rebuild_index:
    with st.spinner("Initializing pipeline..."):
        initialize_pipeline()

query = st.text_area(
    "Enter your question",
    placeholder="Example: What are common attacks in CAVs?"
)

ask_button = st.button("Ask")

if ask_button and query.strip():
    with st.spinner("Preparing LLM model..."):
        llm = build_llm(llm_key)

    with st.spinner("Retrieving documents and generating answer..."):
        result = run_rag(
            query=query,
            retriever=st.session_state.retriever,
            llm=llm
        )

    st.subheader("Answer")
    st.write(result["answer"])

    if "sources" in result:
        st.subheader("Sources")
        for source in result["sources"]:
            st.write(
                f"- {source.get('title', 'unknown')} | "
                f"by {source.get('author', 'unknown')} | "
                f"page {source.get('page', 'unknown')}"
            )

    st.subheader("Retrieved Chunks")
    for i, doc in enumerate(result["retrieved_documents"], start=1):
        with st.expander(
            f"Chunk {i} | {doc.metadata.get('title', 'unknown')} | "
            f"Author: {doc.metadata.get('author', 'unknown')} |"
            f"page {doc.metadata.get('page', 'unknown')}"
        ):
            st.write(doc.page_content)