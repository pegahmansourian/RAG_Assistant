import logging
from pathlib import Path
import asyncio

import streamlit as st

from ResearchRAG.ingestion.loaders import parse_pdf_folder, parse_pdf, sync_parsed_pdfs
from ResearchRAG.ingestion.chunking import split_text
from ResearchRAG.embedding.embeddings import build_embedding_model
from ResearchRAG.embedding.vectorstore import build_database, load_faiss_index, save_faiss_index, update_faiss_index, delete_from_faiss_index
from ResearchRAG.retrieval.retriever import build_retriever
from ResearchRAG.generation.rag_chain import run_rag
from ResearchRAG.config import RAW_PDF_DIR, RERANK_BASE_K, RERANK_TOP_N, RETRIEVAL_K, PROCESSED_DIR, RAGAS_METRICS_FOR_UI
from ResearchRAG.generation.llms import build_llm
from ResearchRAG.retrieval.reranking import build_rerank_retriever
from ResearchRAG.evaluation.evaluation import evaluate_rag_response


from ResearchRAG.utils.logging_config import setup_logging

RAGAS_METRIC_DESCRIPTIONS = {
    "faithfulness":
        "Measures whether the generated answer "
        "is supported by the retrieved context "
        "and avoids hallucinations.",

    "answer_relevancy":
        "Measures how relevant the generated "
        "answer is to the user question.",
}

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Technical PDF RAG Assistant", layout="wide")
st.title("Technical PDF RAG Assistant")

# ── Sidebar: Settings only ───────────────────────────────────────────────────
st.sidebar.header("Settings")

embedding_key = st.sidebar.selectbox(
    "Embedding model",
    ["miniLM", "mpnet", "bge_base", "cohere_v3"]
)

llm_key = st.sidebar.selectbox(
    "LLM model name",
    options=["mistral", "miniLM", "cohere", "llama3", "qwen3"]
)

use_reranker = st.sidebar.checkbox("Use reranker", value=False)

index_name = st.sidebar.text_input(
    "Index name",
    value=f"faiss_{embedding_key}"
)

rebuild_index = st.sidebar.button("Rebuild Index")

st.sidebar.markdown("---")
st.sidebar.subheader("Evaluation")

selected_metric = st.sidebar.selectbox(
    "RAGAS metric",
    options=list(RAGAS_METRICS_FOR_UI.keys()),
    index=0,
)

st.session_state.selected_metric = selected_metric

st.sidebar.caption(RAGAS_METRIC_DESCRIPTIONS[selected_metric])

evaluate_button = st.sidebar.button(
    "Evaluate Response",
    use_container_width=True
)

# ── Session state ────────────────────────────────────────────────────────────
for key in ("vectorstore", "retriever", "use_reranker", "current_embedding_key", "llm", "selected_metric", "last_query", "last_result"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Helpers ──────────────────────────────────────────────────────────────────
def format_authors(authors):
    if not authors:
        return "unknown"
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    return authors[0] if len(authors) == 1 else f"{authors[0]} et al."

def build_pipeline_retriever(vectorstore, use_reranker=False):
    if use_reranker:
        base = build_retriever(vectorstore, search_type="similarity", k=RERANK_BASE_K)
        return build_rerank_retriever(base,top_n=RERANK_TOP_N)

    return build_retriever(vectorstore, search_type="similarity")

# ── Pipeline initialization ──────────────────────────────────────────────────
def initialize_pipeline():
    logger.info("Initializing RAG pipeline")
    sync_parsed_pdfs()

    embedding_model = build_embedding_model(embedding_key)

    if rebuild_index:
        logger.info("Rebuilding FAISS index")
        documents = parse_pdf_folder()
        chunked_documents = split_text(documents)
        vectorstore = build_database(chunked_documents, embedding_model)
        save_faiss_index(vectorstore, index_name)
    else:
        try:
            vectorstore = load_faiss_index(index_name, embedding_model)
        except Exception:
            logger.warning("Failed to load existing index, rebuilding")
            documents = parse_pdf_folder()
            chunked_documents = split_text(documents)
            vectorstore = build_database(chunked_documents, embedding_model)
            save_faiss_index(vectorstore, index_name)

    retriever = build_pipeline_retriever(vectorstore, use_reranker=use_reranker)

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.use_reranker = use_reranker
    st.session_state.current_embedding_key = embedding_key

    logger.info("Pipeline initialized successfully")

pipeline_needs_update = (
    st.session_state.retriever is None
    or rebuild_index
    or st.session_state.get("current_embedding_key") != embedding_key
    or st.session_state.get("use_reranker") != use_reranker
)

if pipeline_needs_update:
    with st.spinner("Initializing pipeline..."):
        initialize_pipeline()

# ── Main: two-column layout ──────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

# ── Left column: PDF Library ─────────────────────────────────────────────────
with left_col:
    st.subheader("📚 PDF Library")

    with st.form("upload_form", clear_on_submit=True):

        uploaded_files = st.file_uploader(
            "Upload PDF(s)",
            type="pdf",
            accept_multiple_files=True,
        )

        upload_submitted = st.form_submit_button(
            "Upload"
        )

    if upload_submitted and uploaded_files:
        # Snapshot BEFORE uploads
        existing_files = {
            p.name for p in Path(RAW_PDF_DIR).glob("*.pdf")
        }

        saved = []
        already_exists = []

        for f in uploaded_files:
            if f.name in existing_files:
                already_exists.append(f.name)
                continue

            dest = Path(RAW_PDF_DIR) / f.name
            if dest.exists():
                already_exists.append(f.name)
                continue

            with open(dest, "wb") as out:
                out.write(f.read())
            saved.append((f.name, dest))
            logger.info("Uploaded PDF: %s", f.name)

        if already_exists:
            st.info(
                "Already in library: "
                + ", ".join(already_exists)
            )
        if saved:
            with st.spinner(f"Processing {len(saved)} new PDF(s)..."):
                new_docs = []
                failed = []

                for name, path in saved:
                    try:
                        new_doc = parse_pdf(path)
                        new_docs.extend(new_doc)
                    except Exception as e:
                        failed.append(name)
                        path.unlink(missing_ok=True)
                        logger.exception("Failed to process uploaded PDF: %s", name)
                        st.warning(f"Could not process {name}: {e}")

                if new_docs:
                    chunked = split_text(new_docs)

                    # Add to existing index
                    embedding_model = build_embedding_model(
                        st.session_state.get("current_embedding_key", embedding_key)
                    )
                    st.session_state.vectorstore = update_faiss_index(
                        index_name, chunked, embedding_model
                    )
                    # Rebuild retriever against updated vectorstore
                    st.session_state.retriever = build_pipeline_retriever(st.session_state.vectorstore, use_reranker)

                    added = [n for n, _ in saved if n not in failed]
                    if added:
                        st.success(f"Added to index: {', '.join(added)}")
                        logger.info("Added %d PDFs to vector index",len(added))

    pdf_files = sorted(Path(RAW_PDF_DIR).glob("*.pdf"))
    if pdf_files:
        st.markdown(f"**{len(pdf_files)} PDF(s) available:**")
        for pdf in pdf_files:
            col1, col2 = st.columns([5, 1])
            col1.markdown(f"📄 {pdf.stem}", help=pdf.name)
            if col2.button("🗑", key=f"del_{pdf.name}"):
                # Remove PDF and its processed JSON
                logger.info("Deleting PDF: %s", pdf.name)
                pdf.unlink(missing_ok=True)

                # Removing from existing index
                embedding_model = build_embedding_model(
                    st.session_state.get("current_embedding_key", embedding_key)
                )
                st.session_state.vectorstore = delete_from_faiss_index(
                    index_name, pdf.name, embedding_model
                )
                # Rebuild retriever against updated vectorstore
                st.session_state.retriever = build_pipeline_retriever(st.session_state.vectorstore, use_reranker)
                logger.info("Deleted PDF successfully: %s", pdf.name)
                st.rerun()

    else:
        st.info("No PDFs yet. Upload one above.")

# ── Right column: Q&A ────────────────────────────────────────────────────────
with right_col:
    st.subheader("💬 Ask a Question")

    query = st.text_area(
        "Enter your question",
        placeholder="Example: What are common attacks in CAVs?",
        label_visibility="collapsed",
    )

    ask_button = st.button("Ask", use_container_width=True)

    if ask_button and query.strip():
        logger.info("Received user query")
        with st.spinner("Preparing LLM model..."):
            st.session_state.llm = build_llm(llm_key)

        with st.spinner("Retrieving documents and generating answer..."):
            result = run_rag(
                query=query,
                retriever=st.session_state.retriever,
                llm=st.session_state.llm
            )
        st.session_state.last_query = query
        st.session_state.last_result = result
        logger.info("RAG query completed successfully")

    if st.session_state.last_result is not None:
        result = st.session_state.last_result

        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(
                result["retrieved_documents"],
                start=1
        ):
            authors = format_authors(
                doc.metadata.get("authors", "unknown")
            )

            with st.expander(
                    f"Chunk {i} | "
                    f"{doc.metadata.get('title', 'unknown')} | "
                    f"Author(S): {authors} | "
                    f"section "
                    f"{doc.metadata.get('section_header', 'unknown')}"
            ):
                st.write(doc.page_content)

        if evaluate_button:
            try:
                with st.spinner("Running RAGAS evaluation..."):

                    eval_result = asyncio.run(
                        evaluate_rag_response(
                            question=st.session_state.last_query,
                            response=st.session_state.last_result["answer"],
                            retrieved_documents=st.session_state.last_result["retrieved_documents"],
                            metric_name=st.session_state.selected_metric,
                        )
                    )

                st.markdown("## 📊 Evaluation Result")

                metric_name = (eval_result["metric"].replace("_", " ").title())

                st.markdown(
                    f"### {metric_name}: "
                    f"`{eval_result['score']:.4f}`"
                )

            except Exception as e:
                logger.exception("RAGAS evaluation failed")

                st.error(f"Evaluation failed: {str(e)}")