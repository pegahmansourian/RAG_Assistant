import json
import logging
from pathlib import Path

from ragas import experiment, Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, FactualCorrectness
from ragas.llms import llm_factory
from ragas.embeddings import HuggingFaceEmbeddings

from litellm import AsyncOpenAI as LiteLLMAsync
import instructor


from ResearchRAG.retrieval.retriever import retrieve_documents
from ResearchRAG.config import EVAL_DIR, EVAL_RESULTS_DIR
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

def build_ragas_metric(*args, **kwargs):
    match kwargs["metric_name"]:
        case "faithfulness":
            return Faithfulness(*args, **kwargs)
        case "answer_relevancy":
            return AnswerRelevancy(*args, **kwargs)
        case "context_precision":
            return ContextPrecision(*args, **kwargs)
        case "context_recall":
            return ContextRecall(*args, **kwargs)
        case "factual_correctness":
            return FactualCorrectness(*args, **kwargs)
        case _:
            return AnswerRelevancy(*args, **kwargs)

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


async def evaluate_rag_response(dataset, pipeline, metric_name, reference_answer=None):
    logger.info("Running RAGAS experiment | metric=%s ", metric_name)

    try:
        litellm_async_client = LiteLLMAsync(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        # Wrap with instructor as AsyncInstructor — _check_client_async returns True for this
        async_instructor_client = instructor.from_openai(
            litellm_async_client,
            mode=instructor.Mode.JSON
        )

        evaluator_llm = llm_factory(
            model="mistral:latest",
            provider="ollama",
            client=async_instructor_client,
            adapter="litellm"
        )
        evaluator_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-base-en-v1.5")

        metric_args = {
            "metric_name": metric_name,
            "llm": evaluator_llm,
            "embeddings": evaluator_embeddings,
        }

        match metric_name:
            case "answer_relevancy":
                metric_args["strictness"] = 2
            case "faithfulness":
                metric_args.pop("embeddings", None)

        metric = build_ragas_metric(**metric_args)

        @experiment()
        async def run_experiment(row):
            result = run_rag(
                query=row["question"],
                retriever=pipeline["retriever"],
                llm=pipeline["llm"]
            )

            logger.info("Scoring sample | metric=%s", metric_name)

            match metric_name:
                case "faithfulness":
                    score = await metric.ascore(
                        user_input=row["question"],
                        response=result["answer"],
                        retrieved_contexts=[doc.page_content for doc in result["retrieved_documents"]],
                    )
                case _:  # answer_relevancy and others
                    score = await metric.ascore(
                        user_input=row["question"],
                        response=result["answer"],
                    )

            logger.info("Sample scored | metric=%s | score=%.4f", metric_name, score)

            return {
                **row,
                "response": result.get("answer", ""),
                metric_name: score.value,
            }

        result = await run_experiment.arun(dataset)

    except Exception:
        logger.exception("Failed RAGAS experiment | metric=%s", metric_name)
        raise

    # Convert to DataFrame for analysis
    df = result.to_pandas()

    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("Experiment Results")
    logger.info("=" * 40)
    logger.info(f"\nDataFrame shape: {df.shape}")
    logger.info(f"\n{df.to_string()}")

    metric_columns = [
        "answer_relevancy",
    ]
    for column in metric_columns:
        if column in df.columns:
            logger.info(f"Average {column}: {df[column].mean():.4f}")

    return result, df