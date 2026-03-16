import json
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from .config import ROOT_DIR, SUPPORTED_PDF_GLOB


def parse_pdf(pdf_path):
    # Parse one PDF and also keep document-level metadata.
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()

    return data


def parse_pdf_folder(folder_path, suffix=SUPPORTED_PDF_GLOB):
    # Parse all PDF files in a folder.

    parsed_pdfs = []
    pdf_path = ROOT_DIR.joinpath(folder_path)
    if pdf_path.exists():
        for pdf_file in list(pdf_path.glob(suffix)):
            parsed_pdfs.extend(parse_pdf(pdf_file))
    else:
        print('Folder not found!')
    return parsed_pdfs
