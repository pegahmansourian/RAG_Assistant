import json
import logging
from pathlib import Path

from ResearchRAG.config import PROCESSED_DIR, RAW_PDF_DIR, ROOT_DIR, SUPPORTED_PDF_GLOB
from .pdf_cleaning import clean_pdf

logger = logging.getLogger(__name__)

def _resolve_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR.joinpath(path)


def _output_paths(pdf_path, output_dir):
    return {
        "text_path": output_dir / f"{pdf_path.stem}.txt",
        "json_path": output_dir / f"{pdf_path.stem}.json",
    }


def run_pdf_etl_for_file(pdf_path, output_dir=PROCESSED_DIR, overwrite=False):
    # Extract from one PDF, transform the content, and load artifacts to data/processed.
    pdf_file = _resolve_path(pdf_path)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _output_paths(pdf_file, output_dir)
    if not overwrite and paths["text_path"].exists() and paths["json_path"].exists():
        logger.info("Skipping already processed PDF: %s", pdf_file.name)
        return {
            "pdf_path": str(pdf_file),
            "text_path": str(paths["text_path"]),
            "json_path": str(paths["json_path"]),
            "status": "skipped",
        }
    logger.info("Running PDF ETL for %s", pdf_file.name)
    try:
        result = clean_pdf(pdf_file)
        paths["text_path"].write_text(result["cleaned_text"], encoding="utf-8")

        payload = {
            "title": result["title"],
            "authors": result["authors"],
            "source": str(pdf_file),
            "chunk_count": len(result["chunks"]),
            "chunks": result["chunks"],
        }
        paths["json_path"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("PDF ETL completed | file=%s | chunks=%d", pdf_file.name, len(result["chunks"]))

        return {
            "pdf_path": str(pdf_file),
            "text_path": str(paths["text_path"]),
            "json_path": str(paths["json_path"]),
            "title": result["title"],
            "chunk_count": len(result["chunks"]),
            "status": "processed",
        }
    except Exception:
        logger.exception("Failed PDF ETL for %s", pdf_file.name)
        raise


def run_pdf_etl(input_dir=RAW_PDF_DIR, output_dir=PROCESSED_DIR, suffix=SUPPORTED_PDF_GLOB, overwrite=False):
    # Run the PDF ETL job for every PDF in a directory and write a manifest.
    source_dir = _resolve_path(input_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if not source_dir.exists():
        logger.error("Input PDF folder not found: %s", source_dir)
        return results

    logger.info("Running batch PDF ETL from %s", source_dir)

    for pdf_file in sorted(source_dir.glob(suffix)):
        try:
            results.append(run_pdf_etl_for_file(pdf_file, output_dir=output_dir, overwrite=overwrite))
        except Exception as exc:
            logger.exception("Failed batch ETL for %s",pdf_file.name)
            results.append(
                {
                    "pdf_path": str(pdf_file),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    logger.info(
        "Batch PDF ETL completed | processed=%d | failed=%d",
        sum(1 for item in results if item["status"] == "processed"),
        sum(1 for item in results if item["status"] == "failed")
    )

    return results
