import json
import logging
from pathlib import Path

from ResearchRAG.config import PROCESSED_DIR, RAW_PDF_DIR, ROOT_DIR, SUPPORTED_PDF_GLOB
from .pdf_cleaning import clean_pdf

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling_core.types.doc import SectionHeaderItem, TextItem, PictureItem, DocItemLabel, TableItem


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


def _build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.generate_picture_images = True
    opts.images_scale = 2.0
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def run_pdf_docling_etl_for_file(pdf_path, converter=None, output_dir=PROCESSED_DIR, overwrite=False):
    # Extract from one PDF, transform the content, and load artifacts to data/processed.
    pdf_file = _resolve_path(pdf_path)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _output_paths(pdf_file, output_dir)
    json_path = paths["json_path"]
    text_path = paths["text_path"]

    if converter is None:
        converter = _build_converter()

    if not overwrite and text_path.exists() and json_path.exists():
        logger.info("Skipping already processed PDF: %s", pdf_file.name)
        return {
            "pdf_path": str(pdf_file),
            "text_path": str(text_path),
            "json_path": str(json_path),
            "status": "skipped",
        }
    logger.info("Running docling PDF ETL for %s", pdf_file.name)
    try:
        paper_id = pdf_file.stem
        img_dir = output_dir / "images" / paper_id

        result = extract_with_docling(pdf_file, converter=converter, paper_id=paper_id, img_dir=img_dir)

        cleaned_text = "\n\n".join(c["text"] for c in result["chunks"])
        text_path.write_text(cleaned_text, encoding="utf-8")

        payload = {
            "paper_id": paper_id,
            "title": result["title"],
            "authors": result["authors"],
            "source": str(pdf_file),
            "chunk_count": len(result["chunks"]),
            "chunks": result["chunks"],
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Docling ETL completed | file=%s | chunks=%d", pdf_file.name, len(result["chunks"]))

        return {
            "pdf_path": str(pdf_file),
            "text_path": str(text_path),
            "json_path": str(json_path),
            "title": result["title"],
            "chunk_count": len(result["chunks"]),
            "status": "processed",
        }
    except Exception:
        logger.exception("Failed docling ETL for %s", pdf_file.name)
        raise

def _walk_sections(doc, paper_id, img_dir):
    img_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    current = {"heading": "preamble", "text_blocks": [], "images": [], "tables": []}
    fig_count = 0
    table_count = 0

    for item, _ in doc.iterate_items():
        if isinstance(item, SectionHeaderItem) or item.label == DocItemLabel.SECTION_HEADER:
            if current["text_blocks"] or current["images"]:
                sections.append(_finalize_section(current))
            current = {"heading": item.text, "text_blocks": [], "images": [], "tables": []}

        elif isinstance(item, TextItem):
            current["text_blocks"].append(item.text)

        elif isinstance(item, TableItem):
            table_count += 1
            page_no = item.prov[0].page_no if item.prov else None
            current["tables"].append({
                "table_id": f"{paper_id}_table{table_count}",
                "markdown": item.export_to_markdown(doc),  # readable text form for embedding
                "dataframe": item.export_to_dataframe(doc).to_dict(orient="records"),  # structured form if you need it
                "page_no": page_no,
            })

        elif isinstance(item, PictureItem):
            fig_count += 1
            fname = f"{paper_id}_fig{fig_count}.png"
            item.get_image(doc).save(img_dir / fname)
            page_no = item.prov[0].page_no if item.prov else None
            current["images"].append({
                "image_id": f"{paper_id}_fig{fig_count}",
                "image_path": str(img_dir / fname),
                "caption": item.caption_text(doc),
                "page_no": page_no,
            })

    if current["text_blocks"] or current["images"] or current["tables"]:
        sections.append(_finalize_section(current))

    return sections


def _finalize_section(section):
    return {
        "heading": section["heading"],
        "text": "\n".join(section["text_blocks"]),
        "images": section["images"],
        "tables": section["tables"],
    }

def extract_with_docling(pdf_path, converter, paper_id, img_dir):
    result = converter.convert(str(pdf_path))
    doc = result.document

    title = doc.name or pdf_path.stem  # docling gives filename-derived name; see note below
    title = title.replace("_", " ")
    chunks = _walk_sections(doc, paper_id, img_dir)  # your section-walker from earlier

    return {
        "title": title,
        "authors": None,  # see note below
        "chunks": chunks,
    }

def run_pdf_docling_etl(input_dir=RAW_PDF_DIR, output_dir=PROCESSED_DIR, suffix=SUPPORTED_PDF_GLOB, overwrite=False):
    # Run the PDF ETL job for every PDF in a directory and write a manifest.
    source_dir = _resolve_path(input_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _converter = _build_converter()

    results = []
    if not source_dir.exists():
        logger.error("Input PDF folder not found: %s", source_dir)
        return results

    logger.info("Running batch Docling PDF ETL from %s", source_dir)

    for pdf_file in sorted(source_dir.glob(suffix)):
        try:
            results.append(run_pdf_docling_etl_for_file(pdf_file, converter=_converter, output_dir=output_dir, overwrite=overwrite))
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