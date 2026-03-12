import re

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

    # Remove broken line hyphens
    text = re.sub("-\n", "", text)

    return text.strip()