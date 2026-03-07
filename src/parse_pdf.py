import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
import fitz


def clean_text(text):
    # Take raw extracted text and do light cleanup.
    # Keep this conservative so you do not damage technical content.
    #
    # Possible steps:
    # - normalize line endings
    # - remove repeated spaces
    # - reduce excessive blank lines
    # - strip leading/trailing whitespace
    return text


def parse_pdf(pdf_path):
    # Parse one PDF file page by page.
    #
    # Steps:
    # 1. convert input path to Path object
    # 2. open the PDF with fitz
    # 3. loop through all pages
    # 4. extract text for each page
    # 5. create one record per page
    #
    # Each record should look like:
    # {
    #     "source_file": pdf file name,
    #     "page_num": page number starting from 1,
    #     "text": extracted page text
    # }
    #
    # Return a list of page-level records.
    return []


def parse_pdf_with_metadata(pdf_path):
    # Parse one PDF and also keep document-level metadata.

    with fitz.open(pdf_path) as doc:
        metadata = doc.metadata or {}
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
        "pdf_metadata": metadata,
        "pages": pages,
    }]


def parse_pdf_folder(folder_path, suffix="*.pdf"):
    # Parse all PDF files in a folder.
    #
    # Steps:
    # 1. convert folder path to Path object
    # 2. find all matching PDF files
    # 3. loop through them
    # 4. parse each PDF
    # 5. combine all page-level records into one list
    #
    # Return one flat list containing records from all PDFs.
    parsed_pdfs = []
    ROOT_DIR = Path(__file__).resolve().parents[1]
    pdf_path = ROOT_DIR.joinpath(folder_path)
    if pdf_path.exists():
        for pdf_file in list(pdf_path.glob(suffix)):
            parsed_pdfs.extend(parse_pdf_with_metadata(pdf_file))
    else:
        print('Folder not found!')
    return parsed_pdfs


def save_parsed_output(data, output_path):
    # Save parsed data to a JSON file.
    #
    # Steps:
    # 1. convert output path to Path object
    # 2. create parent directory if needed
    # 3. open file in write mode with utf-8 encoding
    # 4. dump JSON with indentation
    pass


if __name__ == "__main__":
    # Use this block for quick local testing.
    #
    # Suggested flow:
    # 1. define a sample PDF path from data/raw/
    # 2. check whether the file exists
    # 3. parse the PDF
    # 4. save parsed output to data/processed/
    # 5. print a short success message
    #
    # This block is only for manual testing, not core pipeline use.
    folder_path = 'data/raw/'
    files = parse_pdf_folder(folder_path)
    print(files[0]['pages'][0]['text'])
    pass