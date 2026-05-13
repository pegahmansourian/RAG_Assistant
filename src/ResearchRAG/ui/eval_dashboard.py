import json
from pathlib import Path

import streamlit as st

from ResearchRAG.ingestion.loaders import parse_pdf_folder
from ResearchRAG.ingestion.chunking import split_text
from ResearchRAG.embedding.embeddings import build_embedding_model
from ResearchRAG.embedding.vectorstore import build_database, load_faiss_index, save_faiss_index
from ResearchRAG.retrieval.retriever import build_similarity_retriever, build_mmr_retriever
from ResearchRAG.generation.rag_chain import run_rag
from ResearchRAG.evaluation import load_eval_data, summarize_retrieval_results, normalize_expected_sources
from ResearchRAG.generation.llms import build_llm
from ResearchRAG.config import RAW_PDF_DIR, EVAL_DIR


st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

st.title("RAG Evaluation Dashboard")
st.write("Compare expected evaluation data with retrieved sources and generated RAG answers.")

st.sidebar.header("Settings")

embedding_options = ["miniLM", "mpnet", "bge_base", "cohere_v3"]
embedding_key = st.sidebar.selectbox("Embedding model", embedding_options, index=0)

retriever_type = st.sidebar.selectbox("Retriever type", ["similarity", "mmr"], index=0)

llm_options = ["mistral", "llama3", "qwen3"]
llm_key = st.sidebar.selectbox("LLM", llm_options, index=1)

index_name = st.sidebar.text_input("Index name", value=f"faiss_{embedding_key}")

rebuild_index = st.sidebar.button("Rebuild Index")

eval_files = sorted([p.name for p in Path(EVAL_DIR).glob("*.json")])
selected_eval_file = st.sidebar.selectbox("Eval file", eval_files) if eval_files else None

run_all_button = st.sidebar.button("Run All Questions")

FEEDBACK_FILE = Path(EVAL_DIR) / "human_feedback.json"

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "eval_results" not in st.session_state:
    st.session_state.eval_results = []

if "current_index_name" not in st.session_state:
    st.session_state.current_index_name = None

if "current_embedding_key" not in st.session_state:
    st.session_state.current_embedding_key = None

if "current_retriever_type" not in st.session_state:
    st.session_state.current_retriever_type = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def extract_sources_from_docs(documents):
    sources = []
    for doc in documents:
        sources.append({
            "source_file": doc.metadata.get("title", "unknown"),
            "page_num": doc.metadata.get("pages", "unknown"),
        })
    return sources

def load_human_feedback():
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_human_feedback(feedback_data):
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=2, ensure_ascii=False)


def build_feedback_key(eval_file, question):
    return f"{eval_file}|||{question}"


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

    if retriever_type == "similarity":
        retriever = build_similarity_retriever(vectorstore)
    else:
        retriever = build_mmr_retriever(vectorstore)

    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.current_index_name = index_name
    st.session_state.current_embedding_key = embedding_key
    st.session_state.current_retriever_type = retriever_type


needs_reload = (
    st.session_state.retriever is None
    or rebuild_index
    or st.session_state.current_index_name != index_name
    or st.session_state.current_embedding_key != embedding_key
    or st.session_state.current_retriever_type != retriever_type
)

if needs_reload:
    with st.spinner("Initializing retrieval pipeline..."):
        initialize_pipeline()

if not selected_eval_file:
    st.warning("No evaluation JSON files found in data/eval/")
    st.stop()

eval_data = load_eval_data(selected_eval_file)

if not eval_data:
    st.warning("Selected evaluation file is empty.")
    st.stop()

llm = build_llm(llm_key=llm_key, temperature=0)

question_labels = [
    f"{i+1}. {item.get('question', '')[:100]}"
    for i, item in enumerate(eval_data)
]
selected_index = st.selectbox(
    "Choose an evaluation question",
    options=list(range(len(question_labels))),
    format_func=lambda i: question_labels[i]
)

selected_item = eval_data[selected_index]
selected_question = selected_item.get("question", "")
selected_answer = selected_item.get("answer", "")
feedback_store = load_human_feedback()
feedback_key = build_feedback_key(selected_eval_file, selected_question)
existing_feedback = feedback_store.get(feedback_key, {})

col1, col2 = st.columns(2)

with col1:
    st.subheader("Expected Evaluation Data")
    st.markdown("**Question**")
    st.write(selected_question)
    st.markdown("**Answer**")
    st.write(selected_answer)

    st.markdown("**Expected Sources**")
    expected_sources = normalize_expected_sources(selected_item.get("evidence", []))
    if expected_sources:
        for src in expected_sources:
            if isinstance(src, dict):
                st.write(f"- {src.get('source_file', 'unknown')} | page {src.get('page_num', 'any')}")
                st.write(f" {src.get('support', 'unknown')}")
            else:
                st.write(f"- {src}")
    else:
        st.write("No expected sources provided.")

with col2:
    st.subheader("RAG Output")
    st.markdown("**Question**")
    st.write(selected_item.get("question", ""))
    if st.button("Run Selected Question"):
        with st.spinner("Running RAG..."):
            result = run_rag(
                query=selected_item.get("question", ""),
                retriever=st.session_state.retriever,
                llm=llm
            )

        st.session_state.last_result = result

    if "last_result" in st.session_state:
        result = st.session_state.last_result

        st.markdown("**Generated Answer**")
        st.write(result["answer"])

        st.markdown("**Retrieved Sources**")
        for src in result.get("sources", []):
            st.write(f"- {src.get('title', 'unknown')} | page {src.get('page', 'unknown')}")

        st.markdown("**Retrieved Chunks**")
        for i, doc in enumerate(result["retrieved_documents"], start=1):
            with st.expander(
                f"Chunk {i} | {doc.metadata.get('title', 'unknown')} | page {doc.metadata.get('page', 'unknown')}"
            ):
                st.write(doc.page_content)

st.divider()

st.subheader("Human Feedback")

feedback_col1, feedback_col2, feedback_col3 = st.columns(3)

with feedback_col1:
    correctness_score = st.selectbox(
        "Correctness",
        options=[1, 2, 3, 4, 5],
        index=max(0, existing_feedback.get("correctness", 3) - 1),
        help="1 = very poor, 5 = excellent"
    )

with feedback_col2:
    hallucination_score = st.selectbox(
        "Hallucination",
        options=[1, 2, 3, 4, 5],
        index=max(0, existing_feedback.get("hallucination", 3) - 1),
        help="1 = severe hallucination, 5 = no hallucination"
    )

with feedback_col3:
    completeness_score = st.selectbox(
        "Completeness",
        options=[1, 2, 3, 4, 5],
        index=max(0, existing_feedback.get("completeness", 3) - 1),
        help="1 = very incomplete, 5 = fully complete"
    )

feedback_comment = st.text_area(
    "Comments",
    value=existing_feedback.get("comment", ""),
    placeholder="Optional notes about answer quality, missing points, wrong citations, etc."
)

if st.button("Save Feedback"):
    answer_text = ""
    retrieved_sources = []

    if st.session_state.last_result is not None:
        answer_text = st.session_state.last_result.get("answer", "")
        retrieved_sources = st.session_state.last_result.get("sources", [])

    feedback_store[feedback_key] = {
        "eval_file": selected_eval_file,
        "question": selected_question,
        "correctness": correctness_score,
        "hallucination": hallucination_score,
        "completeness": completeness_score,
        "comment": feedback_comment,
        "expected_sources": expected_sources,
        "saved_answer": answer_text,
        "retrieved_sources": retrieved_sources,
        "embedding_key": embedding_key,
        "retriever_type": retriever_type,
        "llm_key": llm_key,
    }

    save_human_feedback(feedback_store)
    st.success("Feedback saved.")

st.divider()


st.subheader("Run Full Evaluation")

if run_all_button:
    results = []

    with st.spinner("Running evaluation on all questions..."):
        for item in eval_data:
            question = item.get("question", "")
            expected_sources = item.get("expected_sources", [])
            expected_keywords = item.get("expected_keywords", [])

            result = run_rag(
                query=question,
                retriever=st.session_state.retriever,
                llm=llm
            )

            results.append({
                "question": question,
                "expected_sources": expected_sources,
                "expected_keywords": expected_keywords,
                "answer": result["answer"],
                "retrieved_sources": result.get("sources", extract_sources_from_docs(result["retrieved_documents"])),
                "retrieved_documents": result["retrieved_documents"],
            })

    st.session_state.eval_results = results

if st.session_state.eval_results:
    st.subheader("Evaluation Results Summary")

    summary_ready_results = []
    for item in st.session_state.eval_results:
        summary_ready_results.append({
            "question": item["question"],
            "answer": item["answer"],
            "expected_sources": item["expected_sources"],
            "retrieved_sources": item["retrieved_sources"],
            "expected_keywords": item["expected_keywords"],
            "hit_at_k": 0,
            "recall_at_k": 0.0,
            "keyword_coverage": 0.0,
        })

    summary = summarize_retrieval_results(summary_ready_results)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Questions", summary["num_questions"])
    metric_col2.metric("Hit Rate", f"{summary['hit_rate']:.2f}")
    metric_col3.metric("Avg Keyword Coverage", f"{summary['average_keyword_coverage']:.2f}")

    st.subheader("Per-Question Results")
    for i, item in enumerate(st.session_state.eval_results, start=1):
        with st.expander(f"{i}. {item['question']}"):
            left, right = st.columns(2)

            with left:
                st.markdown("**Expected Sources**")
                for src in item["expected_sources"]:
                    if isinstance(src, dict):
                        st.write(f"- {src.get('source_file', 'unknown')} | page {src.get('page_num', 'any')}")
                    else:
                        st.write(f"- {src}")

                st.markdown("**Expected Keywords**")
                for kw in item["expected_keywords"]:
                    st.write(f"- {kw}")

            with right:
                st.markdown("**Generated Answer**")
                st.write(item["answer"])

                st.markdown("**Retrieved Sources**")
                for src in item["retrieved_sources"]:
                    st.write(f"- {src.get('source_file', 'unknown')} | page {src.get('page_num', 'unknown')}")