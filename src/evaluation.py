import json
from pathlib import Path

from .retriever import retrieve_documents
from .config import EVAL_DIR, EVAL_RESULTS_DIR


def load_eval_data(eval_file):
    eval_path = Path(eval_file)

    if not eval_path.is_absolute():
        eval_path = EVAL_DIR / eval_file

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def extract_retrieved_sources(documents):
    sources = []

    for doc in documents:
        source_file = doc.metadata.get("title", "unknown")
        page_num = doc.metadata.get("page", "unknown")

        sources.append({
            "source_file": source_file,
            "page_num": page_num,
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
            for page_num in item.get("pages", []):
                normalized.append({
                    "source_file": item.get("paper"),
                    "page_num": page_num,
                    "support": item.get("support"),
                })

    return normalized


def source_matches(retrieved_source, expected_source):
    same_file = retrieved_source.get("source_file") == expected_source.get("source_file")

    expected_page = expected_source.get("page_num")
    retrieved_page = retrieved_source.get("page_num")

    if expected_page is None:
        return same_file

    return same_file and retrieved_page == expected_page


def compute_hit_at_k(retrieved_sources, expected_sources):
    # This checks:
    # “Did I retrieve at least one correct source in the top k results?”
    # If any retrieved source matches any expected source, it returns 1

    for expected_source in expected_sources:
        for retrieved_source in retrieved_sources:
            if source_matches(retrieved_source, expected_source):
                return 1
    return 0


def compute_recall_at_k(retrieved_sources, expected_sources):
    # This checks:
    # “Out of all expected relevant sources, how many did I retrieve?”
    # So this is a fraction, not just yes/no.

    if not expected_sources:
        return 0.0

    matched = 0

    for expected_source in expected_sources:
        found = False
        for retrieved_source in retrieved_sources:
            if source_matches(retrieved_source, expected_source):
                found = True
                break
        if found:
            matched += 1

    return matched / len(expected_sources)


def evaluate_retrieval(eval_data, retriever):
    results = []

    for item in eval_data:
        question = item.get("question", "")
        expected_sources = normalize_expected_sources(item.get("evidence", []))

        retrieved_documents = retrieve_documents(question, retriever)
        retrieved_sources = extract_retrieved_sources(retrieved_documents)

        hit_at_k = compute_hit_at_k(retrieved_sources, expected_sources)
        recall_at_k = compute_recall_at_k(retrieved_sources, expected_sources)

        results.append({
            "question": question,
            "expected_sources": expected_sources,
            "retrieved_sources": retrieved_sources,
            "hit_at_k": hit_at_k,
            "recall_at_k": recall_at_k,
        })

    return results


def summarize_retrieval_results(results):
    if not results:
        return {
            "num_questions": 0,
            "hit_rate": 0.0,
            "average_recall": 0.0,
        }

    num_questions = len(results)
    total_hits = sum(item["hit_at_k"] for item in results)
    total_recall = sum(item["recall_at_k"] for item in results)

    summary = {
        "num_questions": num_questions,
        "hit_rate": total_hits / num_questions,
        "average_recall": total_recall / num_questions,
    }

    return summary


def save_evaluation_results(results, summary, output_file):
    output_path = Path(output_file)

    if not output_path.is_absolute():
        output_path = EVAL_RESULTS_DIR / output_file

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    print("This module is intended to be imported and used from a notebook or script.")