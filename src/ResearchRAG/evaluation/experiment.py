import argparse
from pathlib import Path
import json
import time
import yaml
import mlflow
import logging
import os

from ResearchRAG.ingestion.loaders import parse_pdf_folder
from ResearchRAG.ingestion.chunking import split_text
from ResearchRAG.embedding.embeddings import build_embedding_model
from ResearchRAG.embedding.vectorstore import build_database, load_faiss_index, save_faiss_index
from ResearchRAG.retrieval.retriever import build_retriever
from ResearchRAG.config import RAW_PDF_DIR, OUTPUTS_DIR, INDEX_DIR, EVAL_DIR, ROOT_DIR
from ResearchRAG.generation.llms import build_llm
from ResearchRAG.retrieval.reranking import build_rerank_retriever
from ResearchRAG.evaluation.evaluation import normalize_expected_sources, evaluate_rag_response
from ResearchRAG.utils.logging_config import setup_logging

from ragas import Dataset
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness


setup_logging()
logger = logging.getLogger(__name__)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(base_path, exp_path):
    logger.info("Loading config | base=%s | exp=%s", base_path, exp_path)
    base_config = load_yaml(base_path)
    exp_config = load_yaml(exp_path)
    config = base_config.copy()
    config.update(exp_config)
    logger.info("Config loaded | experiment=%s", config.get("experiment_name", "unknown"))
    return config


def load_eval_data(eval_file):
    eval_path = Path(eval_file)

    if not eval_path.is_absolute():
        eval_path = EVAL_DIR / eval_file

    logger.info("Loading evaluation dataset | path=%s", eval_path)

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Evaluation dataset loaded | samples=%d", len(data))
        return data

    except Exception:
        logger.exception("Failed to load evaluation dataset | path=%s", eval_path)
        raise


def build_ragas_dataset(eval_data):
    logger.info("Building RAGAS dataset | samples=%d", len(eval_data))

    try:
        dataset = Dataset(
            name="eval",
            backend="local/csv",
            root_dir="data/eval",
        )

        for item in eval_data:
            dataset.append({
                "question": item["question"],
                "reference_answer": item["answer"],
                "difficulty": item.get("difficulty"),
                "type": item.get("type"),
            })

        dataset.save()
        logger.info("RAGAS dataset built and saved")
        return dataset

    except Exception:
        logger.exception("Failed to build RAGAS dataset")
        raise


def build_ragas_metric(**kwargs):
    metric_name = kwargs.pop("metric_name")
    logger.info("Building RAGAS metric | metric=%s", metric_name)

    try:
        match metric_name:
            case "faithfulness":
                return Faithfulness(**kwargs)
            case "answer_relevancy":
                return AnswerRelevancy(**kwargs)
            case "context_precision":
                return ContextPrecision(**kwargs)
            case "context_recall":
                return ContextRecall(**kwargs)
            case "factual_correctness":
                return FactualCorrectness(**kwargs)
            case _:
                logger.warning("Unknown metric name | metric=%s | falling back to AnswerRelevancy", metric_name)
                return AnswerRelevancy(**kwargs)

    except Exception:
        logger.exception("Failed to build RAGAS metric | metric=%s", metric_name)
        raise


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
    retriever_search_type = config["search_type"]
    index_name = make_index_name(config)

    logger.info(
        "Building pipeline | embedding=%s | llm=%s | reranker=%s | search_type=%s | index=%s",
        embedding_key, llm_key, use_reranker, retriever_search_type, index_name
    )

    try:
        embedding_model = build_embedding_model(embedding_key)
        index_path = INDEX_DIR / index_name

        if index_path.exists():
            logger.info("Loading existing FAISS index | index=%s", index_name)
            vectorstore = load_faiss_index(index_name, embedding_model)
        else:
            logger.info("Index not found, building from PDFs | index=%s", index_name)
            documents = parse_pdf_folder(RAW_PDF_DIR)
            chunked_documents = split_text(
                documents,
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )
            logger.info("Documents chunked | chunks=%d", len(chunked_documents))
            vectorstore = build_database(chunked_documents, embedding_model)
            save_faiss_index(vectorstore, index_name)
            logger.info("FAISS index built and saved | index=%s", index_name)

        if use_reranker:
            logger.info("Building rerank retriever | base_k=%d | top_n=%d", rerank_base_k, rerank_top_n)
            base_retriever = build_retriever(vectorstore, k=rerank_base_k, search_type=retriever_search_type)
            retriever = build_rerank_retriever(base_retriever, top_n=rerank_top_n)
        else:
            logger.info("Building standard retriever | k=%d", retriever_k)
            retriever = build_retriever(vectorstore, k=retriever_k, search_type=retriever_search_type)

        llm = build_llm(llm_key)
        logger.info("Pipeline built successfully")

        return {
            "vectorstore": vectorstore,
            "retriever": retriever,
            "llm": llm,
            "embedding_key": embedding_key,
            "llm_key": llm_key,
            "use_reranker": use_reranker,
            "index_name": index_name,
        }

    except Exception:
        logger.exception("Failed to build pipeline | embedding=%s | llm=%s", embedding_key, llm_key)
        raise

def save_json(data, path):
    logger.info("Saving JSON | path=%s", path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("JSON saved successfully | path=%s", path)
    except Exception:
        logger.exception("Failed to save JSON | path=%s", path)
        raise


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config yaml")
    parser.add_argument("--base_config", default="configs/base.yaml", help="Path to base config yaml")
    args = parser.parse_args()

    config = load_config(args.base_config, args.config)
    eval_set = load_eval_data(config["eval_set"])
    ragas_eval_set = build_ragas_dataset(eval_set)
    pipeline = build_pipeline(config)

    logger.info("Starting experiment | name=%s | metric=%s", config["experiment_name"], config["metric_name"])

    start_time = time.perf_counter()
    result, df = await evaluate_rag_response(
        ragas_eval_set, pipeline, config["metric_name"],
        ragas_llm="llama3", ragas_embedding="miniLM", reference_answer=None
    )
    total_time = time.perf_counter() - start_time
    avg_score = df[config["metric_name"]].mean()
    metrics = {config["metric_name"]: avg_score}
    logger.info("Experiment completed | name=%s | metric=%s | score=%.4f | elapsed=%.2fs",
                config["experiment_name"], config["metric_name"], avg_score, total_time)

    logger.info("Logging results to MLflow | experiment=technical_pdf_rag | run=%s", config["experiment_name"])

    mlflow.set_experiment("technical_pdf_rag")
    with mlflow.start_run(run_name=config["experiment_name"]):
        mlflow.log_params(config)
        mlflow.log_metrics(metrics)

        output_dir = Path(OUTPUTS_DIR) / "experiments" / config["experiment_name"]
        save_json(config, output_dir / "config.json")
        save_json(df.to_dict(orient="records"), output_dir / "results.json")
        save_json(metrics, output_dir / "metrics.json")

        mlflow.log_artifact(str(output_dir / "config.json"))
        mlflow.log_artifact(str(output_dir / "results.json"))
        mlflow.log_artifact(str(output_dir / "metrics.json"))

    logger.info("MLflow run complete | name=%s", config["experiment_name"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())