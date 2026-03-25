import json
from pathlib import Path

from .config import PROCESSED_DIR, RAW_PDF_DIR, ROOT_DIR, SUPPORTED_PDF_GLOB
from .pdf_cleaning import clean_pdf


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
        return {
            "pdf_path": str(pdf_file),
            "text_path": str(paths["text_path"]),
            "json_path": str(paths["json_path"]),
            "status": "skipped",
        }

    result = clean_pdf(pdf_file)
    paths["text_path"].write_text(result["cleaned_text"], encoding="utf-8")

    payload = {
        "title": result["title"],
        "source": str(pdf_file),
        "chunk_count": len(result["chunks"]),
        "chunks": result["chunks"],
    }
    paths["json_path"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "pdf_path": str(pdf_file),
        "text_path": str(paths["text_path"]),
        "json_path": str(paths["json_path"]),
        "title": result["title"],
        "chunk_count": len(result["chunks"]),
        "status": "processed",
    }


def run_pdf_etl(input_dir=RAW_PDF_DIR, output_dir=PROCESSED_DIR, suffix=SUPPORTED_PDF_GLOB, overwrite=False):
    # Run the PDF ETL job for every PDF in a directory and write a manifest.
    source_dir = _resolve_path(input_dir)
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if not source_dir.exists():
        print("Folder not found!")
        return results

    for pdf_file in sorted(source_dir.glob(suffix)):
        try:
            results.append(run_pdf_etl_for_file(pdf_file, output_dir=output_dir, overwrite=overwrite))
        except Exception as exc:
            results.append(
                {
                    "pdf_path": str(pdf_file),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    manifest = {
        "input_dir": str(source_dir),
        "output_dir": str(output_dir),
        "total_files": len(results),
        "processed_files": sum(1 for item in results if item["status"] == "processed"),
        "skipped_files": sum(1 for item in results if item["status"] == "skipped"),
        "failed_files": sum(1 for item in results if item["status"] == "failed"),
        "files": results,
    }
    manifest_path = output_dir / "processed_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the PDF cleaning ETL job.")
    parser.add_argument("--input-dir", type=str, default=str(RAW_PDF_DIR), help="Directory containing PDF files")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DIR), help="Directory for ETL outputs")
    parser.add_argument("--suffix", type=str, default=SUPPORTED_PDF_GLOB, help="Glob pattern for source PDFs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt and .json outputs")
    args = parser.parse_args()

    results = run_pdf_etl(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        suffix=args.suffix,
        overwrite=args.overwrite,
    )
    print(f"Processed {sum(1 for item in results if item['status'] == 'processed')} PDF files.")
    print(f"Skipped {sum(1 for item in results if item['status'] == 'skipped')} PDF files.")
    print(f"Failed {sum(1 for item in results if item['status'] == 'failed')} PDF files.")
