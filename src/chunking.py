from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
import json
import re


def split_into_paragraphs(text):
    # Split page text into paragraph-like units.

    if not text:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    return chunks



if __name__ == "__main__":
    # Use this block for quick manual testing.
    #
    # Suggested flow:
    # 1. load a parsed JSON file from data/processed/
    # 2. extract page records
    # 3. chunk them
    # 4. save chunked output back to data/processed/
    # 5. print number of created chunks
    pass