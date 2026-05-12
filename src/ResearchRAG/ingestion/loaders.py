import json
from pathlib import Path
from langchain_core.documents import Document
from ResearchRAG.config import PROCESSED_DIR, ROOT_DIR, SUPPORTED_PDF_GLOB
from ResearchRAG.ingestion.pdf_etl import run_pdf_etl_for_file


def parse_pdf(pdf_path):
    # Load one cleaned PDF from its processed JSON chunks.
    pdf_file = Path(pdf_path)
    if not pdf_file.is_absolute():
        pdf_file = ROOT_DIR.joinpath(pdf_file)

    json_path = PROCESSED_DIR / f"{pdf_file.stem}.json"
    if not json_path.exists():
        if not pdf_file.exists():
            print(f"PDF not found: {pdf_file}")
            return []
        run_pdf_etl_for_file(pdf_file)

    if not json_path.exists():
        print(f"ETL did not produce output for: {pdf_file.name}")
        return []

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

    return documents


def parse_pdf_folder(folder_path, suffix=SUPPORTED_PDF_GLOB):
    # Load all cleaned PDFs in a folder from processed JSON chunks.

    parsed_pdfs = []
    pdf_path = ROOT_DIR.joinpath(folder_path)
    if pdf_path.exists():
        for pdf_file in list(pdf_path.glob(suffix)):
            parsed_pdfs.extend(parse_pdf(pdf_file))
    else:
        print('Folder not found!')
    return parsed_pdfs
