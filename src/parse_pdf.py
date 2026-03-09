import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
import fitz


def clean_text(text):
    # Take raw extracted text and do light cleanup.
    # Keep this conservative so you do not damage technical content.

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Correct "fi" misreading
    text = re.sub("ﬁ", "fi", text)

    return text.strip()


def parse_pdf(pdf_path):
    # Parse one PDF and also keep document-level metadata.

    with fitz.open(pdf_path) as doc:
        pages = []

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            page_text = clean_text(page.get_text("text"))

            pages.append({
                "source_file": pdf_path.name,
                "page_num": page_index + 1,
                "text": page_text,
            })

    return [{
        "source_file": pdf_path.name,
        "source_path": str(pdf_path),
        "num_pages": len(pages),
        "pages": pages,
    }]


def parse_pdf_folder(folder_path, suffix="*.pdf"):
    # Parse all PDF files in a folder.

    parsed_pdfs = []
    ROOT_DIR = Path(__file__).resolve().parents[1]
    pdf_path = ROOT_DIR.joinpath(folder_path)
    if pdf_path.exists():
        for pdf_file in list(pdf_path.glob(suffix)):
            parsed_pdfs.extend(parse_pdf(pdf_file))
    else:
        print('Folder not found!')
    return parsed_pdfs


def save_parsed_output(data, output_path):
    # Save parsed data to a JSON file.

    ROOT_DIR = Path(__file__).resolve().parents[1]
    json_path = ROOT_DIR.joinpath(output_path)
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=2)


if __name__ == "__main__":
    folder_path = 'data/raw/'
    files = parse_pdf_folder(folder_path)
    save_parsed_output(files, "data/processed/processed.json")
    pass