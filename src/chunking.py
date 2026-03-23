from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP
import json
import re
from .loaders import parse_pdf_folder


def create_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    return text_splitter

def split_text(docs):
    # Split page text into paragraph-like units.

    if not docs:
        return []

    splitter = create_splitter()
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["page"] = chunk.metadata["page"] + 1

    return chunks


if __name__ == "__main__":
    # Use this block for quick manual testing.
    #
    folder_path = 'data/raw/'
    files = parse_pdf_folder(folder_path)
    chunks = split_text(files)
    pass