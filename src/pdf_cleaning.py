from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fitz  # PyMuPDF
import pymupdf4llm
from langchain_core.documents import Document


@dataclass
class CleaningConfig:
    # Header-based chunking
    min_section_chars: int = 120

    # Cropping areas used to ignore page furniture
    top_margin_ratio: float = 0.08
    bottom_margin_ratio: float = 0.06
    left_margin_ratio: float = 0.05
    right_margin_ratio: float = 0.05

    # Repeated line detection for page furniture
    repeated_line_min_pages: int = 2
    repeated_line_max_len: int = 220

    # Section removal
    remove_references: bool = True
    remove_biography: bool = True


class PDFCleaner:
    """
    Clean IEEE-style scientific PDFs for RAG.

    What this does:
    - loads the PDF
    - keeps the document title separately
    - removes title/authors/front matter from body text
    - removes repeated headers/footers/page furniture
    - removes figure/table/equation captions and common float remnants
    - removes References and Biography/Biographies sections
    - chunks by markdown headers
    - merges sections spanning multiple pages into one chunk

    Important note:
    This is heuristic. Scientific PDFs vary a lot, so expect to tune regexes and
    margins for your papers.
    """

    HEADER_PATTERNS = [
        re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE),
        re.compile(r"^\*\*(abstract|keywords?)\.?\*\*\s*", re.IGNORECASE | re.MULTILINE),
    ]

    DROP_LINE_PATTERNS = [
        # Common journal / publisher furniture
        re.compile(r"(?i)^.*ieee.*$"),
        re.compile(r"(?i)^.*digital object identifier.*$"),
        re.compile(r"(?i)^.*personal use is permitted.*$"),
        re.compile(r"(?i)^.*copyright.*$"),
        re.compile(r"(?i)^.*published by.*$"),
        re.compile(r"(?i)^.*vol\.?\s+\d+.*no\.?\s+\d+.*$"),
        re.compile(r"(?i)^.*manuscript received.*$"),
        re.compile(r"(?i)^.*recommended by.*$"),
        re.compile(r"(?i)^.*this work was supported.*$"),
        # Page numbers or isolated number markers
        re.compile(r"^\s*\d+\s*$"),
    ]

    CAPTION_PATTERNS = [
        re.compile(r"(?im)^\s*fig\.?\s*\d+[\.:]?\s+.*$"),
        re.compile(r"(?im)^\s*figure\s*\d+[\.:]?\s+.*$"),
        re.compile(r"(?im)^\s*table\s*[ivxlcdm\d]+[\.:]?\s+.*$"),
        re.compile(r"(?im)^\s*tab\.?\s*[ivxlcdm\d]+[\.:]?\s+.*$"),
        re.compile(r"(?im)^\s*eq\.?\s*\(?\d+\)?[\.:]?\s+.*$"),
        re.compile(r"(?im)^\s*(\*\*)?algorithm\s*\d+(\*\*)?[\.:]?\s+.*$"),
    ]

    BIO_SECTION_PATTERNS = [
        re.compile(r"(?im)^#{1,6}\s*(biography|biographies|author biography|about the authors)\s*$"),
        re.compile(r"(?im)^\*\*(biography|biographies|author biography|about the authors)\*\*\s*$"),
    ]


    REFERENCE_SECTION_PATTERNS = [
        re.compile(r"(?im)^#{1,6}\s*references\s*$"),
        re.compile(r"(?im)^\*\*references\*\*\s*$"),
        re.compile(r"(?im)^\s*references\s*$"),
        re.compile(r"(?im)^\s*[ivxlcdm\d]+\.?\s*references\s*$"),
        re.compile(r"(?im)^\s*\*\*[ivxlcdm\d]+\.?\s*references\*\*\s*$"),
    ]

    REFERENCE_ENTRY_PATTERNS = [
        re.compile(r"^\[\d{1,3}\]\s+\S+"),
        re.compile(r"^\[\d{1,3}\]\s*$"),
        re.compile(r"^\d{1,3}\)\s+\S+"),
    ]

    def __init__(self, pdf_path: str | Path, config: CleaningConfig | None = None):
        self.pdf_path = Path(pdf_path)
        self.config = config or CleaningConfig()
        self.doc = fitz.open(self.pdf_path)
        self.title = (self.doc.metadata or {}).get("title", "") or self._guess_title_from_first_page()

    def _guess_title_from_first_page(self) -> str:
        page = self.doc[0]
        blocks = page.get_text("dict").get("blocks", [])
        spans: list[tuple[float, str]] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if text:
                        spans.append((float(span.get("size", 0)), text))

        if not spans:
            return self.pdf_path.stem

        max_size = max(s for s, _ in spans)
        title_lines = [t for s, t in spans if s >= max_size - 0.5 and len(t) > 5]
        title = " ".join(title_lines).strip()
        title = re.sub(r"\s+", " ", title)
        return title or self.pdf_path.stem

    def _page_text_without_margins(self, page: fitz.Page) -> str:
        rect = page.rect
        clip = fitz.Rect(
            rect.x0 + rect.width * self.config.left_margin_ratio,
            rect.y0 + rect.height * self.config.top_margin_ratio,
            rect.x1 - rect.width * self.config.right_margin_ratio,
            rect.y1 - rect.height * self.config.bottom_margin_ratio,
        )
        return page.get_text("text", clip=clip)

    def _find_repeated_margin_lines(self, pages: list[dict[str, Any]]) -> set[str]:
        candidates: dict[str, set[int]] = {}

        # Raw PDF text near page edges
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")
            lines = [self._normalize_line(x) for x in text.splitlines() if self._normalize_line(x)]
            if not lines:
                continue

            for line in lines[:6] + lines[-6:]:
                if len(line) <= self.config.repeated_line_max_len:
                    candidates.setdefault(line, set()).add(page_num)

        # Markdown-extracted text near page edges
        for i, page in enumerate(pages):
            text = page.get("text", "")
            lines = [self._normalize_line(x) for x in text.splitlines() if self._normalize_line(x)]
            if not lines:
                continue

            for line in lines[:8] + lines[-8:]:
                if len(line) <= self.config.repeated_line_max_len:
                    candidates.setdefault(line, set()).add(i)

        repeated = {
            line
            for line, page_ids in candidates.items()
            if len(page_ids) >= self.config.repeated_line_min_pages
        }

        if self.title:
            repeated.update(self._title_fragments(self.title))

        return repeated

    @staticmethod
    def _normalize_line(line: str) -> str:
        line = re.sub(r"\*\*", "", line)
        line = re.sub(r"\s+", " ", line).strip(" -:\t")
        return line

    def _title_fragments(self, title: str) -> set[str]:
        normalized = self._normalize_line(title)
        if not normalized:
            return set()

        fragments = {normalized}
        words = normalized.split()

        # Build long n-gram fragments from the title so partial running headers match too
        max_n = min(8, len(words))
        min_n = min(4, max_n)

        for n in range(max_n, min_n - 1, -1):
            for i in range(0, len(words) - n + 1):
                frag = " ".join(words[i:i + n])
                if len(frag) >= 25:
                    fragments.add(frag)

        return fragments

    def _extract_markdown_pages(self) -> list[dict[str, Any]]:
        return pymupdf4llm.to_markdown(
            str(self.pdf_path),
            page_chunks=True,
            extract_words=True,
            show_progress=False,
        )

    def _clean_page_markdown(self, text: str, repeated_margin_lines: set[str], page_number: int) -> str:
        lines = text.splitlines()
        cleaned: list[str] = []

        for raw_line in lines:
            line = raw_line.rstrip()
            normalized = self._normalize_line(line)
            if not normalized:
                cleaned.append("")
                continue

            if normalized in repeated_margin_lines:
                continue

            if self.title:
                title_fragments = self._title_fragments(self.title)
                if any(fragment in normalized for fragment in title_fragments if len(fragment) >= 25):
                    continue

            if re.search(r"(?i)et al\.?\s*:", normalized):
                continue

            if "picture [" in normalized and "intentionally omitted" in normalized:
                continue

            if any(p.match(normalized) for p in self.DROP_LINE_PATTERNS):
                continue

            if any(p.match(normalized) for p in self.CAPTION_PATTERNS):
                continue


            # Remove likely equation-only lines: math-heavy and low alphabetic content
            if self._looks_like_equation_or_formula(normalized):
                continue

            # Remove image placeholders / markdown image links
            if re.match(r"!\[.*?\]\(.*?\)", normalized):
                continue

            cleaned.append(line)

        page_text = "\n".join(cleaned)

        # Remove front matter from first page: title, authors, affiliations, emails.
        if page_number == 0:
            page_text = self._remove_front_matter(page_text)

        # Remove common markdown code fences occasionally added around titles / affiliations
        page_text = re.sub(r"(?m)^```\s*$", "", page_text)

        # Collapse excessive whitespace
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        return page_text

    def _remove_front_matter(self, text: str) -> str:
        candidates = []
        for pattern in [
            re.compile(r"(?im)^[-* ]*\*\*abstract\*\*"),
            re.compile(r"(?im)^\s*abstract\s*$"),
            re.compile(r"(?im)^#{1,6}\s*abstract\s*$"),
            re.compile(r"(?im)^\*\*i\.\s*introduction\*\*\s*$"),
            re.compile(r"(?im)^\s*[ivxlcdm\d]+\.\s*introduction\s*$"),
            re.compile(r"(?im)^#{1,6}\s*introduction\s*$"),
        ]:
            m = pattern.search(text)
            if m:
                candidates.append(m.start())

        if candidates:
            text = text[min(candidates):]

        if self.title:
            escaped_title = re.escape(self.title.strip())
            text = re.sub(rf"(?im)^\s*{escaped_title}\s*$", "", text)

        text = re.sub(r"(?im)^.*@[A-Za-z0-9._-]+\.[A-Za-z]{2,}.*$", "", text)
        text = re.sub(r"(?im)^\s*\d+\s*(department|school|faculty|institute|college|lab|laboratory)\b.*$", "", text)
        text = re.sub(r"(?im)^\s*(department|school|faculty|institute|college|lab|laboratory|university)\b.*$", "",
                      text)
        text = re.sub(r"(?im)^\s*[A-Z][A-Z\s,0-9]+$", "", text)

        text = re.sub(r"(?im)^[-* ]*\*\*abstract\*\*", "## Abstract\n", text)
        text = re.sub(r"(?im)^[-* ]*\*\*index terms\*\*", "## Index Terms\n", text)

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    @staticmethod
    def _looks_like_equation_or_formula(line: str) -> bool:
        # Skip ordinary short headings
        if line.startswith("#"):
            return False

        symbol_count = sum(ch in "=+-/*^_<>∑∫√≈≜≤≥±[]{}()|\\" for ch in line)
        alpha_count = sum(ch.isalpha() for ch in line)
        digit_count = sum(ch.isdigit() for ch in line)
        total = len(line)

        if total < 8:
            return False

        if re.match(r"^\(?\d+\)?$", line):
            return True

        # Likely equation if symbols dominate and natural language is sparse
        return (symbol_count >= 3 and alpha_count / max(total, 1) < 0.45) or (
            digit_count > 0 and symbol_count > 0 and alpha_count / max(total, 1) < 0.30
        )

    def _merge_pages(self) -> str:
        pages = self._extract_markdown_pages()
        repeated_margin_lines = self._find_repeated_margin_lines(pages)
        cleaned_pages = []

        for i, page in enumerate(pages):
            page_text = page.get("text", "")
            page_num = page.get("metadata", {}).get("page", i)
            cleaned = self._clean_page_markdown(page_text, repeated_margin_lines, page_num)
            if cleaned:
                cleaned_pages.append(cleaned)

        merged = "\n\n".join(cleaned_pages)
        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
        return merged

    def _remove_sections(self, text: str) -> str:
        # Normalize headings first so plain/bold/numbered References gets caught
        text = self._promote_numbered_headings(text)

        cut_points: list[int] = []

        if self.config.remove_references:
            for p in self.REFERENCE_SECTION_PATTERNS:
                m = p.search(text)
                if m:
                    cut_points.append(m.start())

            named_reference_start = self._find_named_section_start(
                text,
                {"references", "reference", "bibliography"},
            )
            if named_reference_start is not None:
                cut_points.append(named_reference_start)

            reference_block_start = self._find_reference_block_start(text)
            if reference_block_start is not None:
                cut_points.append(reference_block_start)

        if self.config.remove_biography:
            for p in self.BIO_SECTION_PATTERNS:
                m = p.search(text)
                if m:
                    cut_points.append(m.start())

            biography_start = self._find_named_section_start(
                text,
                {"biography", "biographies", "author biography", "about the authors"},
            )
            if biography_start is not None:
                cut_points.append(biography_start)

        if cut_points:
            text = text[: min(cut_points)]

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _promote_numbered_headings(self, text: str) -> str:
        lines = text.splitlines()
        out = []
        for line in lines:
            stripped = line.strip()

            if re.match(r"^(\d+|[IVXLC]+)\.?\s+[A-Z][A-Za-z0-9 ,\-/()]+$", stripped):
                out.append(f"## {stripped}")
                continue

            if re.match(r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,\-/()]+$", stripped):
                out.append(f"### {stripped}")
                continue

            if re.match(
                    r"^(Abstract|Introduction|Conclusion|Conclusions|Method|Methods|Experiments|Results|Discussion|Keywords|References)\s*$",
                    stripped,
                    re.I,
            ):
                if not stripped.startswith("#"):
                    out.append(f"## {stripped}")
                    continue

            if re.match(
                    r"^\*\*(Abstract|Introduction|Conclusion|Conclusions|Method|Methods|Experiments|Results|Discussion|Keywords|References)\*\*\s*$",
                    stripped,
                    re.I,
            ):
                out.append(f"## {stripped.strip('*')}")
                continue

            out.append(line)

        return "\n".join(out)

    @staticmethod
    def _normalize_heading_candidate(line: str) -> str:
        candidate = re.sub(r"\*\*", "", line).strip()
        candidate = re.sub(r"^#{1,6}\s*", "", candidate)
        candidate = re.sub(r"^(?:[IVXLC]+|\d+|[A-Z])[\.\)]\s+", "", candidate, flags=re.I)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .:-").lower()
        return candidate

    def _find_named_section_start(self, text: str, section_names: set[str]) -> int | None:
        offset = 0
        for line in text.splitlines(keepends=True):
            normalized = self._normalize_heading_candidate(line)
            if normalized in section_names:
                return offset
            offset += len(line)
        return None

    def _find_reference_block_start(self, text: str) -> int | None:
        lines = text.splitlines(keepends=True)
        if not lines:
            return None

        offsets: list[int] = []
        offset = 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)

        # IEEE references are usually a dense block of numbered entries near the end.
        scan_start = int(len(lines) * 0.6)
        for i in range(scan_start, len(lines)):
            stripped = lines[i].strip()
            if not stripped:
                continue

            if not any(pattern.match(stripped) for pattern in self.REFERENCE_ENTRY_PATTERNS):
                continue

            window_matches = 0
            for candidate in lines[i:i + 12]:
                candidate_stripped = candidate.strip()
                if any(pattern.match(candidate_stripped) for pattern in self.REFERENCE_ENTRY_PATTERNS):
                    window_matches += 1

            if window_matches >= 3:
                return offsets[i]

        return None


    def _chunk_by_headers(self, text: str) -> list[dict[str, str]]:
        text = self._promote_numbered_headings(text)

        heading_regex = re.compile(r"(?m)^(#{2,6})\s+(.+?)\s*$")
        matches = list(heading_regex.finditer(text))

        if not matches:
            fallback = text.strip()
            return [{"header": "Document", "content": fallback}] if fallback else []

        chunks: list[dict[str, str]] = []

        # Keep anything before first heading only if useful
        preamble = text[: matches[0].start()].strip()
        if preamble and len(preamble) >= self.config.min_section_chars:
            chunks.append({"header": "Preamble", "content": preamble})

        for i, match in enumerate(matches):
            header = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if not content:
                continue
            chunks.append({"header": header, "content": content})

        return chunks

    def load_and_clean(self) -> dict[str, Any]:
        merged_text = self._merge_pages()
        merged_text = self._remove_sections(merged_text)
        chunks = self._chunk_by_headers(merged_text)

        docs = []
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "source": str(self.pdf_path),
                        "title": self.title,
                        "section_header": chunk["header"],
                        "chunk_id": idx,
                    },
                )
            )

        return {
            "title": self.title,
            "cleaned_text": merged_text,
            "chunks": chunks,
            "documents": docs,
        }


def clean_pdf(pdf_path: str | Path, config: CleaningConfig | None = None) -> dict[str, Any]:
    cleaner = PDFCleaner(pdf_path=pdf_path, config=config)
    return cleaner.load_and_clean()


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Clean a scientific PDF for RAG.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--save-text", type=str, default="", help="Optional path to save cleaned text")
    parser.add_argument("--save-json", type=str, default="", help="Optional path to save chunk metadata as JSON")
    args = parser.parse_args()

    result = clean_pdf(args.pdf_path)

    print(f"Title: {result['title']}")
    print(f"Chunks: {len(result['chunks'])}")

    if args.save_text:
        Path(args.save_text).write_text(result["cleaned_text"], encoding="utf-8")

    if args.save_json:
        payload = {
            "title": result["title"],
            "chunks": result["chunks"],
        }
        Path(args.save_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
