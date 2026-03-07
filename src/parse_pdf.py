import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def clean_text(text):
    # Take raw extracted text and do light cleanup.
    # Keep this conservative so you do not damage technical content.
    #
    # Possible steps:
    # - normalize line endings
    # - remove repeated spaces
    # - reduce excessive blank lines
    # - strip leading/trailing whitespace
    pass


def extract_page_text(page):
    # Extract raw text from one PDF page.
    #
    # You can start with page.get_text("text").
    # Later, if needed, you can test other extraction modes.
    #
    # Return cleaned page text.
    pass


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
    #
    # In addition to page records, collect things like:
    # - source file name
    # - full source path
    # - number of pages
    # - built-in PDF metadata from the document
    #
    # Suggested return format:
    # {
    #     "source_file": ...,
    #     "source_path": ...,
    #     "num_pages": ...,
    #     "pdf_metadata": ...,
    #     "pages": [...]
    # }
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load_and_split()
    return []


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
    print(files)
    pass