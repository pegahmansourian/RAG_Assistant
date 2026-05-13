import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from ResearchRAG.config import PROCESSED_DIR, SUPPORTED_PDF_GLOB, RAW_PDF_DIR
from ResearchRAG.ingestion.pdf_etl import run_pdf_etl_for_file

logger = logging.getLogger(__name__)

def parse_pdf(pdf_path):
    # Load one cleaned PDF from its processed JSON chunks.
    pdf_file = Path(pdf_path)
    sync_parsed_pdfs()

    if not pdf_file.exists():
        logger.error("PDF not found: %s", pdf_file)
        return []

    json_path = PROCESSED_DIR / f"{pdf_file.stem}.json"

    if not json_path.exists():
        logger.error("Processed JSON not found for: %s", pdf_file.name)
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        documents = []
        for idx, chunk in enumerate(payload.get("chunks", [])):
            documents.append(
                Document(
                    page_content=chunk.get("content", ""),
                    metadata={
                        "source": payload.get("source", str(pdf_file)),
                        "title": payload.get("title", pdf_file.stem),
                        "authors": payload.get("authors", []),
                        "section_header": chunk.get("header", "Document"),
                        "chunk_id": idx,
                    },
                )
            )
        logger.info("Loaded %d chunks from %s", len(documents),pdf_file.name)
        return documents
    except Exception:
        logger.exception("Failed to parse PDF: %s", pdf_file.name)
        raise


def parse_pdf_folder():
    # Load all cleaned PDFs in a folder from processed JSON chunks.
    parsed_pdfs = []
    pdf_path = Path(RAW_PDF_DIR)
    if not pdf_path.exists():
        logger.error("PDF folder not found: %s", pdf_path)
        return parsed_pdfs

    logger.info("Parsing PDFs from %s", pdf_path)

    try:
        for pdf_file in pdf_path.glob(SUPPORTED_PDF_GLOB):
            parsed_pdfs.extend(parse_pdf(pdf_file))

        logger.info("Loaded %d parsed document chunks", len(parsed_pdfs))

        return parsed_pdfs

    except Exception:
        logger.exception("Failed to parse PDF folder")
        raise


def sync_parsed_pdfs():
    logger.info("Synchronizing processed PDF files")
    try:
        for pdf_path in Path(RAW_PDF_DIR).glob(SUPPORTED_PDF_GLOB):
            json_path = PROCESSED_DIR / f"{pdf_path.stem}.json"

            if not json_path.exists():
                logger.info("Running ETL for missing PDF: %s", pdf_path.name)
                run_pdf_etl_for_file(pdf_path)

        logger.info("PDF synchronization completed")

    except Exception:
        logger.exception("Failed to synchronize processed PDFs")
        raise