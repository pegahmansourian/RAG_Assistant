import json
import logging
from pathlib import Path

from ragas import experiment, Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


from ResearchRAG.retrieval.retriever import retrieve_documents
from ResearchRAG.config import EVAL_DIR, EVAL_RESULTS_DIR, RAGAS_METRICS_FOR_EXP
from ResearchRAG.generation.rag_chain import run_rag
from ResearchRAG.generation.llms import build_llm
from ResearchRAG.embedding.embeddings import build_embedding_model

logger = logging.getLogger(__name__)


def load_eval_data(eval_file):
    eval_path = Path(eval_file)

    if not eval_path.is_absolute():
        eval_path = EVAL_DIR / eval_file

    logger.info("Loading evaluation dataset: %s", eval_path)

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info("Loaded %d evaluation samples", len(data))

        return data

    except Exception:
        logger.exception("Failed to load evaluation dataset: %s",eval_path)
        raise


def build_ragas_dataset(eval_data):

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

    return dataset


def extract_retrieved_sources(documents):
    sources = []

    for doc in documents:
        source_file = doc.metadata.get("title", "unknown")
        section = doc.metadata.get("section_header", "unknown")

        sources.append({
            "source_file": source_file,
            "section": section,
        })

    return sources


def normalize_expected_sources(expected_sources):
    normalized = []

    for item in expected_sources:
        if isinstance(item, str):
            normalized.append({
                "source_file": item,
                "page_num": None,
            })
        elif isinstance(item, dict):
            for section in item.get("section_header", []):
                normalized.append({
                    "source_file": item.get("paper"),
                    "section": section,
                    "support": item.get("support"),
                })

    return normalized

def save_evaluation_results(results, output_file):
    output_path = Path(output_file)

    if not output_path.is_absolute():
        output_path = EVAL_RESULTS_DIR / output_file

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("Saved evaluation results to %s", output_path)
    except Exception:
        logger.exception("Failed to save evaluation results: %s", output_path)
        raise


async def evaluate_rag_response(question, response, retrieved_documents, metric_name, ragas_llm="llama3", ragas_embedding="miniLM", reference_answer=None):
    if metric_name not in RAGAS_METRICS_FOR_EXP:
        logger.error("Unsupported RAGAS metric: %s", metric_name)
        raise ValueError(f"Unsupported metric: {metric_name}")

    logger.info("Running RAGAS evaluation | metric=%s", metric_name)

    try:
        metric = RAGAS_METRICS_FOR_EXP[metric_name]
        metric.llm = LangchainLLMWrapper(build_llm(ragas_llm))
        metric.embeddings = LangchainEmbeddingsWrapper(build_embedding_model(ragas_embedding))

        retrieved_contexts = [doc.page_content for doc in retrieved_documents]

        #retrieved_metadata = extract_retrieved_sources(retrieved_documents)

        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        if reference_answer is not None:
            sample.reference = reference_answer

        score = await metric.single_turn_ascore(sample)

        logger.info("RAGAS evaluation completed | metric=%s | score=%.4f", metric_name, score)

        return {
            "question": question,
            "response": response,
            "reference_answer": reference_answer,
            "retrieved_contexts": retrieved_contexts,
            "metric": metric_name,
            "score": score,
        }
    except Exception:
        logger.exception("Failed RAGAS evaluation | metric=%s", metric_name)
        raise