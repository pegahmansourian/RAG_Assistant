import argparse
from pathlib import Path
import json
import time
import yaml
import mlflow

from src.loaders import parse_pdf_folder
from src.chunking import split_text
from src.embeddings import build_embedding_model
from src.vectorstore import build_database, load_faiss_index, save_faiss_index
from src.retriever import build_similarity_retriever
from src.rag_chain import run_rag
from src.config import RAW_PDF_DIR, OUTPUTS_DIR, INDEX_DIR
from src.llms import build_llm
from src.reranking import build_rerank_retriever
from src.evaluation import normalize_expected_sources, compute_hit_at_k, compute_recall_at_k, extract_retrieved_sources


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(base_path, exp_path):
    base_config = load_yaml(base_path)
    exp_config = load_yaml(exp_path)

    config = base_config.copy()
    config.update(exp_config)
    return config


def load_eval_set(eval_path):
    with open(eval_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_index_name(config):
    return "{}_chunk{}_overlap{}".format(
        config["embedding_model"],
        config["chunk_size"],
        config["chunk_overlap"]
    )


def build_pipeline(config):
    embedding_key = config["embedding_model"]
    llm_key = config["llm"]
    use_reranker = config["use_reranker"]
    retriever_k = config["retriever_k"]
    rerank_top_n = config["rerank_top_n"]
    rerank_base_k = config["rerank_base_k"]
    index_name = make_index_name(config)

    embedding_model = build_embedding_model(embedding_key)

    index_path = INDEX_DIR / index_name

    if index_path.exists():
        vectorstore = load_faiss_index(index_name, embedding_model)
    else:
        documents = parse_pdf_folder(RAW_PDF_DIR)
        chunked_documents = split_text(
            documents,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        vectorstore = build_database(chunked_documents, embedding_model)
        save_faiss_index(vectorstore, index_name)

    if use_reranker:
        base_retriever = build_similarity_retriever(vectorstore, k=rerank_base_k)
        retriever = build_rerank_retriever(base_retriever, top_n=rerank_top_n)
    else:
        retriever = build_similarity_retriever(vectorstore, k=retriever_k)

    llm = build_llm(llm_key)

    pipeline = {
        "vectorstore": vectorstore,
        "retriever": retriever,
        "llm": llm,
        "embedding_key": embedding_key,
        "llm_key": llm_key,
        "use_reranker": use_reranker,
        "index_name": index_name,
    }

    return pipeline


def answer_question(pipeline, question_record):
    query = question_record["question"]

    result = run_rag(
        query=query,
        retriever=pipeline["retriever"],
        llm=pipeline["llm"]
    )

    predicted_answer = result["answer"]

    retrieved_contexts = []
    source_documents = result.get("retrieved_documents", [])

    for doc in source_documents:
        retrieved_contexts.append({
            "source_file": doc.metadata.get("title", ""),
            "page_num": doc.metadata.get("page", "")
        })

    expected_sources = normalize_expected_sources(question_record.get("evidence", []))
    expected_contexts = []
    if expected_sources:
        for src in expected_sources:
            expected_contexts.append({
                "source_file": src.get('source_file', ""),
                "page_num": src.get('page_num', "")
            })

    return {
        "question": query,
        "ground_truth_answer": question_record.get("answer", ""),
        "ground_truth_retrieved_contexts": expected_contexts,
        "predicted_answer": predicted_answer,
        "retrieved_contexts": retrieved_contexts
    }


def compute_metrics(predictions, total_time):
    num_questions = len(predictions)

    total_hit = 0.0
    total_recall = 0.0
    for p in predictions:
        retrieved_sources = p.get("retrieved_contexts", [])
        expected_sources = p.get("ground_truth_retrieved_contexts", [])

        hit_at_k = compute_hit_at_k(retrieved_sources, expected_sources)
        recall_at_k = compute_recall_at_k(retrieved_sources, expected_sources)

        total_hit += hit_at_k
        total_recall += recall_at_k

    if num_questions > 0:
        avg_hit = total_hit / num_questions
        avg_recall = total_recall / num_questions
        avg_latency = total_time / num_questions
    else:
        avg_hit = 0.0
        avg_recall = 0.0
        avg_latency = 0.0

    return {
        "num_questions": num_questions,
        "avg_hit": avg_hit,
        "avg_recall": avg_recall,
        "avg_latency_sec": avg_latency,
    }


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment config yaml, for example configs/exp_reranker.yaml"
    )
    parser.add_argument(
        "--base_config",
        default="configs/base.yaml",
        help="Path to base config yaml"
    )
    args = parser.parse_args()

    config = load_config(args.base_config, args.config)
    eval_set = load_eval_set(config["eval_set"])

    pipeline = build_pipeline(config)

    predictions = []
    start_time = time.perf_counter()

    for question_record in eval_set:
        result = answer_question(pipeline, question_record)
        predictions.append(result)

    total_time = time.perf_counter() - start_time
    metrics = compute_metrics(predictions, total_time)

    mlflow.set_experiment("technical_pdf_rag")

    with mlflow.start_run(run_name=config["experiment_name"]):
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)

        output_dir = Path(OUTPUTS_DIR) / "experiments/"
        output_dir = output_dir / config["experiment_name"]
        save_json(config, output_dir / "config.json")
        save_json(predictions, output_dir / "predictions.json")
        save_json(metrics, output_dir / "metrics.json")

        mlflow.log_artifact(str(output_dir / "config.json"))
        mlflow.log_artifact(str(output_dir / "predictions.json"))
        mlflow.log_artifact(str(output_dir / "metrics.json"))

    print("Experiment finished.")
    print("Experiment name:", config["experiment_name"])
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()

