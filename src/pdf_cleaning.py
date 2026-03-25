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
    remove_acknowledgments: bool = True


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
        re.compile(r"(?i)^.*creative commons attribution 4\.0 license.*$"),
        re.compile(r"(?i)^.*creativecommons\.org/licenses/by/4\.0/.*$"),
        re.compile(r"(?i)^.*vol\.?\s+\d+.*no\.?\s+\d+.*$"),
        re.compile(r"(?i)^.*manuscript received.*$"),
        re.compile(r"(?i)^.*date of publication.*$"),
        re.compile(r"(?i)^.*date of current version.*$"),
        re.compile(r"(?i)^.*recommended by.*$"),
        re.compile(r"(?i)^.*the review of this article was coordinated by.*$"),
        re.compile(r"(?i)^.*corresponding author.*$"),
        re.compile(r"(?i)^.*this work was supported.*$"),
        re.compile(r"(?i)^.*e-mail:.*$"),
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

    ACKNOWLEDGMENT_SECTION_PATTERNS = [
        re.compile(r"(?im)^#{1,6}\s*(acknowledgment|acknowledgments|acknowledgement|acknowledgements)\s*$"),
        re.compile(r"(?im)^\*\*(acknowledgment|acknowledgments|acknowledgement|acknowledgements)\*\*\s*$"),
    ]

    REFERENCE_SECTION_PATTERNS = [
        re.compile(r"(?im)^#{1,6}\s*references\s*$"),
        re.compile(r"(?im)^\*\*references\*\*\s*$"),
        re.compile(r"(?im)^\s*references\s*$"),
        re.compile(r"(?im)^\s*[ivxlcdm\d]+\.?\s*references\s*$"),
        re.compile(r"(?im)^\s*\*\*[ivxlcdm\d]+\.?\s*references\*\*\s*$"),
    ]

    ALGORITHM_HEADING_PATTERNS = [
        re.compile(r"(?im)^#{1,6}\s*\**algorithm\s+\d+.*$"),
        re.compile(r"(?im)^\**algorithm\s+\d+.*$"),
    ]

    REFERENCE_ENTRY_PATTERNS = [
        re.compile(r"^\[\d{1,3}\]\s+\S+"),
        re.compile(r"^\[\d{1,3}\]\s*$"),
        re.compile(r"^\d{1,3}\)\s+\S+"),
    ]

    INLINE_FRONT_MATTER_PATTERNS = [
        re.compile(
            r"(?is)manuscript received .*?date of current version .*?\.",
        ),
        re.compile(
            r"(?is)manuscript received .*?e-mail:.*?\)",
        ),
        re.compile(
            r"(?is)the review of this article was coordinated by .*?\.",
        ),
        re.compile(
            r"(?is)\(corresponding author:.*?\)",
        ),
        re.compile(
            r"(?is)[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:\s*;\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})*",
        ),
        re.compile(
            r"(?is)digital object identi\w* .*?(?=\n|$)",
        ),
    ]

    def __init__(self, pdf_path: str | Path, config: CleaningConfig | None = None):
        """Open the PDF, load runtime config, and resolve a title for metadata."""
        self.pdf_path = Path(pdf_path)
        self.config = config or CleaningConfig()
        self.doc = fitz.open(self.pdf_path)
        self.title = (self.doc.metadata or {}).get("title", "") or self._guess_title_from_first_page()

    def _guess_title_from_first_page(self) -> str:
        """Infer a title from the largest text spans on page one when metadata is missing."""
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
        """Return raw page text clipped to the inner body area, excluding outer margins."""
        rect = page.rect
        clip = fitz.Rect(
            rect.x0 + rect.width * self.config.left_margin_ratio,
            rect.y0 + rect.height * self.config.top_margin_ratio,
            rect.x1 - rect.width * self.config.right_margin_ratio,
            rect.y1 - rect.height * self.config.bottom_margin_ratio,
        )
        return page.get_text("text", clip=clip)

    def _find_repeated_margin_lines(self, pages: list[dict[str, Any]]) -> set[str]:
        """Collect lines that repeat across pages so headers and footers can be dropped."""
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
        """Normalize markdown markers and whitespace so line comparisons are consistent."""
        line = re.sub(r"\*\*", "", line)
        line = re.sub(r"\s+", " ", line).strip(" -:\t")
        return line

    def _title_fragments(self, title: str) -> set[str]:
        """Generate long title fragments for detecting running headers based on the paper title."""
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
        """Extract per-page markdown-like text using pymupdf4llm."""
        return pymupdf4llm.to_markdown(
            str(self.pdf_path),
            page_chunks=True,
            extract_words=True,
            show_progress=False,
        )

    def _clean_page_markdown(self, text: str, repeated_margin_lines: set[str], page_number: int) -> str:
        """Remove page-level noise such as furniture, captions, and formula-only lines."""
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
        for pattern in self.INLINE_FRONT_MATTER_PATTERNS:
            page_text = pattern.sub("", page_text)

        # Remove front matter from first page: title, authors, affiliations, emails.
        if page_number == 0:
            page_text = self._remove_front_matter(page_text)

        # Remove common markdown code fences occasionally added around titles / affiliations
        page_text = re.sub(r"(?m)^```\s*$", "", page_text)

        # Collapse excessive whitespace
        page_text = re.sub(r"\n{3,}", "\n\n", page_text).strip()
        return page_text

    def _remove_front_matter(self, text: str) -> str:
        """Trim title-page metadata and normalize Abstract/Index Terms into chunkable headings."""
        abstract_candidates = []
        for pattern in [
            re.compile(r"(?im)^[-* ]*\*\*abstract\*\*"),
            re.compile(r"(?im)^\s*abstract\s*$"),
            re.compile(r"(?im)^#{1,6}\s*abstract\s*$"),
            re.compile(r"(?im)_\*\*abstract\*\*_"),
        ]:
            m = pattern.search(text)
            if m:
                abstract_candidates.append(m.start())

        if abstract_candidates:
            text = text[min(abstract_candidates):]

        if self.title:
            escaped_title = re.escape(self.title.strip())
            text = re.sub(rf"(?im)^\s*{escaped_title}\s*$", "", text)

        text = text.replace("â€”", "—").replace("â€“", "–")

        # Normalize inline IEEE-style abstract/index-term labels on the first page.
        text = re.sub(
            r"(?is)_\*\*abstract\*\*_\s*\*\*[—\-–]\s*",
            "## Abstract\n",
            text,
        )
        text = re.sub(
            r"(?is)\*\*\s*_\*\*index terms\*\*_\s*\*\*[—\-–]\s*",
            "\n## Index Terms\n",
            text,
        )
        text = re.sub(r"(?im)^abstract\s*[—\-–:]\s*", "## Abstract\n", text)
        text = re.sub(r"(?im)^index terms\s*[—\-–:]\s*", "## Index Terms\n", text)

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
        """Heuristically detect standalone equation lines so they do not pollute text chunks."""
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
        """Clean each extracted page and merge them into one document string."""
        pages = self._extract_markdown_pages()
        repeated_margin_lines = self._find_repeated_margin_lines(pages)
        cleaned_pages = []

        for i, page in enumerate(pages):
            page_text = page.get("text", "")
            cleaned = self._clean_page_markdown(page_text, repeated_margin_lines, i)
            if cleaned:
                cleaned_pages.append(cleaned)

        merged = "\n\n".join(cleaned_pages)
        merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
        return merged

    def _remove_sections(self, text: str) -> str:
        """Cut trailing sections such as references, biographies, and acknowledgments."""
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

        if self.config.remove_acknowledgments:
            for p in self.ACKNOWLEDGMENT_SECTION_PATTERNS:
                m = p.search(text)
                if m:
                    cut_points.append(m.start())

            acknowledgment_start = self._find_named_section_start(
                text,
                {"acknowledgment", "acknowledgments", "acknowledgement", "acknowledgements"},
            )
            if acknowledgment_start is not None:
                cut_points.append(acknowledgment_start)

        if cut_points:
            text = text[: min(cut_points)]

        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text

    def _remove_algorithm_blocks(self, text: str) -> str:
        """Remove algorithm headings and their body blocks from the merged document text."""
        lines = text.splitlines()
        cleaned: list[str] = []
        skipping_algorithm = False

        for line in lines:
            stripped = line.strip()

            if any(pattern.match(stripped) for pattern in self.ALGORITHM_HEADING_PATTERNS):
                skipping_algorithm = True
                continue

            if skipping_algorithm and re.match(r"^#{2,6}\s+.+$", stripped):
                skipping_algorithm = False

            if skipping_algorithm:
                continue

            cleaned.append(line)

        cleaned_text = "\n".join(cleaned)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
        return cleaned_text

    def _promote_numbered_headings(self, text: str) -> str:
        """Convert plain numbered or bold headings into markdown headers for chunk splitting."""
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
                    r"^(Abstract|Introduction|Conclusion|Conclusions|Method|Methods|Experiments|Results|Discussion|Keywords|References|Acknowledgment|Acknowledgments|Acknowledgement|Acknowledgements)\s*$",
                    stripped,
                    re.I,
            ):
                if not stripped.startswith("#"):
                    out.append(f"## {stripped}")
                    continue

            if re.match(
                    r"^\*\*(Abstract|Introduction|Conclusion|Conclusions|Method|Methods|Experiments|Results|Discussion|Keywords|References|Acknowledgment|Acknowledgments|Acknowledgement|Acknowledgements)\*\*\s*$",
                    stripped,
                    re.I,
            ):
                out.append(f"## {stripped.strip('*')}")
                continue

            out.append(line)

        return "\n".join(out)

    @staticmethod
    def _normalize_heading_candidate(line: str) -> str:
        """Normalize a heading-like line so different heading styles can be compared uniformly."""
        candidate = re.sub(r"\*\*", "", line).strip()
        candidate = re.sub(r"^#{1,6}\s*", "", candidate)
        candidate = re.sub(r"^(?:[IVXLC]+|\d+|[A-Z])[\.\)]\s+", "", candidate, flags=re.I)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .:-").lower()
        return candidate

    def _find_named_section_start(self, text: str, section_names: set[str]) -> int | None:
        """Find the character offset where a named section heading first appears."""
        offset = 0
        for line in text.splitlines(keepends=True):
            normalized = self._normalize_heading_candidate(line)
            if normalized in section_names:
                return offset
            offset += len(line)
        return None

    def _find_reference_block_start(self, text: str) -> int | None:
        """Fallback detector for bibliography blocks when the References heading is malformed."""
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
        """Split cleaned document text into section chunks using markdown-style headings."""
        text = self._promote_numbered_headings(text)

        heading_regex = re.compile(r"(?m)^(#{2,6})\s+(.+?)\s*$")
        matches = list(heading_regex.finditer(text))

        if not matches:
            fallback = text.strip()
            return [{"header": "Document", "content": fallback}] if fallback else []

        chunks: list[dict[str, str]] = []

        # Keep anything before first heading only if useful
        preamble = text[: matches[0].start()].strip()
        if preamble:
            front_chunks = self._extract_front_matter_chunks_flexible(preamble)
            if front_chunks:
                chunks.extend(front_chunks)
            elif len(preamble) >= self.config.min_section_chars:
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

    def _extract_inline_front_matter_chunks(self, preamble: str) -> list[dict[str, str]]:
        """Convert inline Abstract/Index Terms preamble text into normal section chunks."""
        working = (
            preamble.replace("â€”", "—")
            .replace("â€“", "–")
            .replace("Ã¢â‚¬â€", "—")
            .replace("Ã¢â‚¬â€œ", "–")
        )

        patterns = [
            ("Abstract", r"(?is)_?\*\*abstract\*\*_?\s*\*\*(?:—|–|-|:)\s*"),
            ("Index Terms", r"(?is)_?\*\*index terms\*\*_?\s*\*\*(?:—|–|-|:)\s*"),
        ]

        matches: list[tuple[str, re.Match[str]]] = []
        for header, pattern in patterns:
            match = re.search(pattern, working)
            if match:
                matches.append((header, match))

        if not matches:
            return []

        matches.sort(key=lambda item: item[1].start())
        chunks: list[dict[str, str]] = []

        first_header, first_match = matches[0]
        if first_match.start() > 0:
            leading = working[:first_match.start()].strip()
            leading = re.sub(r"^\*+\s*", "", leading)
            leading = re.sub(r"\s*\*+$", "", leading)
            leading = re.sub(r"\n{3,}", "\n\n", leading).strip()
            if leading and len(leading) >= self.config.min_section_chars and first_header == "Index Terms":
                chunks.append({"header": "Abstract", "content": leading})

        for i, (header, match) in enumerate(matches):
            start = match.end()
            end = matches[i + 1][1].start() if i + 1 < len(matches) else len(working)
            content = working[start:end].strip()
            content = re.sub(r"^\*+\s*", "", content)
            content = re.sub(r"\s*\*+$", "", content)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            if content:
                chunks.append({"header": header, "content": content})

        return chunks

    def _extract_front_matter_chunks_flexible(self, text: str) -> list[dict[str, str]]:
        """Extract Abstract and Index Terms from either markdown-style or plain first-page text."""
        working = (
            text.replace("â€”", "—")
            .replace("â€“", "–")
            .replace("Ã¢â‚¬â€", "—")
            .replace("Ã¢â‚¬â€œ", "–")
            .replace("ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â", "—")
            .replace("ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“", "–")
        )

        patterns = [
            ("Abstract", r"(?im)(?:_?\*\*abstract\*\*_?\s*\*\*|^abstract)\s*(?:—|–|-|:)\s*"),
            ("Index Terms", r"(?im)(?:_?\*\*index terms\*\*_?\s*\*\*|^index terms)\s*(?:—|–|-|:)\s*"),
        ]

        matches: list[tuple[str, re.Match[str]]] = []
        for header, pattern in patterns:
            match = re.search(pattern, working)
            if match:
                matches.append((header, match))

        if not matches:
            return []

        matches.sort(key=lambda item: item[1].start())
        chunks: list[dict[str, str]] = []

        first_header, first_match = matches[0]
        if first_match.start() > 0:
            leading = working[:first_match.start()].strip()
            leading = re.sub(r"^\*+\s*", "", leading)
            leading = re.sub(r"\s*\*+$", "", leading)
            leading = re.sub(r"\n{3,}", "\n\n", leading).strip()
            if leading and len(leading) >= self.config.min_section_chars and first_header == "Index Terms":
                chunks.append({"header": "Abstract", "content": leading})

        for i, (header, match) in enumerate(matches):
            start = match.end()
            end = matches[i + 1][1].start() if i + 1 < len(matches) else len(working)
            content = working[start:end].strip()
            content = re.sub(r"^\*+\s*", "", content)
            content = re.sub(r"\s*\*+$", "", content)
            content = re.sub(r"\n{3,}", "\n\n", content).strip()
            if content:
                chunks.append({"header": header, "content": content})

        return chunks

    def _extract_first_page_front_matter_chunks(self) -> list[dict[str, str]]:
        """Recover Abstract and Index Terms directly from raw first-page text when extraction misses them."""
        first_page_text = self._page_text_without_margins(self.doc[0])
        if not first_page_text:
            return []

        working = (
            first_page_text.replace("â€”", "—")
            .replace("â€“", "–")
            .replace("Ã¢â‚¬â€", "—")
            .replace("Ã¢â‚¬â€œ", "–")
        )
        for pattern in self.INLINE_FRONT_MATTER_PATTERNS:
            working = pattern.sub("", working)
        if self.title:
            working = re.sub(rf"(?im)^\s*{re.escape(self.title.strip())}\s*$", "", working)
        working = re.sub(r"(?im)^.*@[A-Za-z0-9._-]+\.[A-Za-z]{2,}.*$", "", working)
        working = re.sub(r"(?im)^\s*\d+\s*(department|school|faculty|institute|college|lab|laboratory)\b.*$", "", working)
        working = re.sub(r"(?im)^\s*(department|school|faculty|institute|college|lab|laboratory|university)\b.*$", "", working)
        working = re.sub(r"(?im)^\s*[A-Z][A-Z\s,0-9]+$", "", working)
        working = re.sub(r"\n{3,}", "\n\n", working).strip()

        intro_match = re.search(r"(?im)^(?:[IVXLC]+\.\s*)?introduction\s*$", working)
        front_text = working[:intro_match.start()].strip() if intro_match else working.strip()
        if not front_text:
            return []

        chunks = self._extract_front_matter_chunks_flexible(front_text)
        if chunks:
            return chunks

        plain_index_match = re.search(
            r"(?im)^index terms\s*(?:—|–|-|:)\s*",
            front_text,
        )
        if plain_index_match:
            leading = front_text[:plain_index_match.start()].strip()
            trailing = front_text[plain_index_match.end():].strip()
            recovered_chunks: list[dict[str, str]] = []
            if len(leading) >= self.config.min_section_chars:
                recovered_chunks.append({"header": "Abstract", "content": leading})
            if trailing:
                trailing = re.sub(r"^\*+\s*", "", trailing)
                trailing = re.sub(r"\s*\*+$", "", trailing).strip()
                recovered_chunks.append({"header": "Index Terms", "content": trailing})
            return recovered_chunks

        index_match = re.search(r"(?is)_?\*\*index terms\*\*_?\s*\*\*(?:—|–|-|:)\s*", front_text)
        if index_match:
            leading = front_text[:index_match.start()].strip()
            trailing = front_text[index_match.end():].strip()
            recovered_chunks: list[dict[str, str]] = []
            if len(leading) >= self.config.min_section_chars:
                recovered_chunks.append({"header": "Abstract", "content": leading})
            if trailing:
                trailing = re.sub(r"^\*+\s*", "", trailing)
                trailing = re.sub(r"\s*\*+$", "", trailing).strip()
                recovered_chunks.append({"header": "Index Terms", "content": trailing})
            return recovered_chunks

        return []

    def _repair_front_matter_chunks(self, chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        """Insert missing Abstract/Index Terms chunks recovered from the raw first page."""
        if not chunks:
            return chunks

        existing_headers = {self._normalize_heading_candidate(chunk["header"]) for chunk in chunks}
        recovered = self._extract_first_page_front_matter_chunks()
        if not recovered:
            return chunks

        missing = [
            chunk for chunk in recovered
            if self._normalize_heading_candidate(chunk["header"]) not in existing_headers
        ]
        if not missing:
            return chunks

        insert_at = 0
        if chunks and self._normalize_heading_candidate(chunks[0]["header"]) == "preamble":
            insert_at = 1

        return chunks[:insert_at] + missing + chunks[insert_at:]

    def _inject_missing_front_matter(self, merged_text: str) -> str:
        """Prepend recovered Abstract/Index Terms text to the merged document when absent."""
        recovered = self._extract_first_page_front_matter_chunks()
        if not recovered:
            return merged_text

        existing_headers = {
            self._normalize_heading_candidate(match.group(2))
            for match in re.finditer(r"(?m)^(#{2,6})\s+(.+?)\s*$", merged_text)
        }
        missing_blocks = []
        for chunk in recovered:
            header_key = self._normalize_heading_candidate(chunk["header"])
            if header_key in existing_headers:
                continue
            missing_blocks.append(f"## {chunk['header']}\n{chunk['content']}")

        if not missing_blocks:
            return merged_text

        return "\n\n".join(missing_blocks + [merged_text]).strip()

    @staticmethod
    def _normalize_for_overlap(text: str) -> str:
        """Normalize text for approximate overlap checks during intro repair."""
        text = re.sub(r"\*\*", "", text)
        text = re.sub(r"_", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _canonical_alnum(text: str) -> str:
        """Strip a string down to lowercase alphanumeric characters for loose prefix checks."""
        return re.sub(r"[^a-z0-9]+", "", text.lower())

    @staticmethod
    def _normalize_anchor_line(text: str) -> str:
        """Normalize a candidate anchor line so cleaned bullets match raw PDF list items."""
        text = re.sub(r"\*\*", "", text)
        text = re.sub(r"_", "", text).strip()
        text = re.sub(r"^[-*]\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _rewrap_pdf_lines(text: str) -> str:
        """Join wrapped PDF lines back into paragraphs while preserving larger paragraph breaks."""
        text = re.sub(r"(?<!-)\n(?=\S)", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_first_page_introduction_text(self) -> str:
        """Read the first-page introduction directly from clipped raw text for recovery purposes."""
        first_page_text = self._page_text_without_margins(self.doc[0])
        if not first_page_text:
            return ""

        for pattern in self.INLINE_FRONT_MATTER_PATTERNS:
            first_page_text = pattern.sub("", first_page_text)
        first_page_text = re.sub(r"(?im)^.*@[A-Za-z0-9._-]+\.[A-Za-z]{2,}.*$", "", first_page_text)
        first_page_text = re.sub(
            r"(?im)^\s*(department|school|faculty|institute|college|lab|laboratory|university)\b.*$",
            "",
            first_page_text,
        )
        first_page_text = re.sub(r"\n{3,}", "\n\n", first_page_text).strip()

        start_match = re.search(
            r"(?im)^(?:#\s*)?(?:[IVXLC]+\.\s*)?introduction\s*$",
            first_page_text,
        )
        if not start_match:
            return ""

        remainder = first_page_text[start_match.end():].strip()
        end_match = re.search(
            r"(?im)^(?:[A-Z]\.\s+[A-Z].*|[IVXLC]+\.\s+[A-Z].*|#{1,6}\s+.+)$",
            remainder,
        )
        if end_match:
            remainder = remainder[:end_match.start()].strip()

        return self._rewrap_pdf_lines(remainder)

    def _extract_raw_section_text(self, header: str) -> str:
        """Extract a section body from raw clipped pages using a heading match as an anchor."""
        target = self._normalize_heading_candidate(header)
        for page in self.doc:
            page_text = self._page_text_without_margins(page)
            if not page_text:
                continue

            lines = page_text.splitlines()
            start_idx = None
            for idx, line in enumerate(lines):
                if self._normalize_heading_candidate(line) == target:
                    start_idx = idx + 1
                    break

            if start_idx is None:
                continue

            section_lines: list[str] = []
            for line in lines[start_idx:]:
                normalized = self._normalize_heading_candidate(line)
                if normalized and normalized != target:
                    if re.match(r"^(?:[A-Z]\.|[IVXLC]+\.)\s+", line.strip(), re.I):
                        break
                section_lines.append(line)

            section_text = "\n".join(section_lines).strip()
            if section_text:
                for pattern in self.INLINE_FRONT_MATTER_PATTERNS:
                    section_text = pattern.sub("", section_text)
                section_text = re.sub(r"(?im)^.*@[A-Za-z0-9._-]+\.[A-Za-z]{2,}.*$", "", section_text)
                section_text = re.sub(r"\n{3,}", "\n\n", section_text).strip()
                return self._rewrap_pdf_lines(section_text)

        return ""

    def _repair_section_prefixes(self, chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        """Prepend missing opening text when a chunk starts in the middle of a raw page section."""
        repaired_chunks: list[dict[str, str]] = []
        for chunk in chunks:
            header_key = self._normalize_heading_candidate(chunk["header"])
            if header_key in {"abstract", "index terms"} or header_key.endswith("introduction"):
                repaired_chunks.append(chunk)
                continue

            raw_section = self._extract_raw_section_text(chunk["header"])
            if not raw_section:
                repaired_chunks.append(chunk)
                continue

            current_lines = [line.strip() for line in chunk["content"].splitlines() if line.strip()]
            anchor_line = current_lines[0] if current_lines else ""
            normalized_anchor = self._normalize_anchor_line(anchor_line)
            anchor_index = raw_section.find(normalized_anchor) if normalized_anchor else -1

            numbered_item_match = re.match(r"^(?:[-*]\s*)?((?:\d+|[A-Z]))\)\s+", anchor_line)
            if anchor_index < 0 and numbered_item_match:
                item_label = numbered_item_match.group(1)
                marker_match = re.search(rf"(?<!\w){re.escape(item_label)}\)\s+", raw_section)
                if marker_match:
                    anchor_index = marker_match.start()

            if anchor_index < 0 and normalized_anchor:
                list_match = re.match(r"^(?:\d+|[A-Z])\)\s+(.+)$", normalized_anchor)
                if list_match:
                    numbered_anchor = normalized_anchor
                    if numbered_anchor in raw_section:
                        anchor_index = raw_section.find(numbered_anchor)

            if anchor_index > 0:
                prefix = raw_section[:anchor_index].strip()
                if prefix:
                    chunk = {
                        "header": chunk["header"],
                        "content": f"{self._rewrap_pdf_lines(prefix)}\n\n{chunk['content']}",
                    }
            else:
                normalized_raw = self._normalize_for_overlap(raw_section)
                normalized_current = self._normalize_for_overlap(chunk["content"])
                if normalized_current and normalized_current in normalized_raw:
                    prefix = normalized_raw.split(normalized_current, 1)[0].strip()
                    if prefix:
                        chunk = {
                            "header": chunk["header"],
                            "content": f"{self._rewrap_pdf_lines(prefix)}\n\n{chunk['content']}",
                        }

            repaired_chunks.append(chunk)

        return repaired_chunks

    def _repair_first_introduction_chunk(self, chunks: list[dict[str, str]]) -> list[dict[str, str]]:
        """Patch a truncated first introduction chunk by prepending missing first-page text."""
        first_page_intro = self._extract_first_page_introduction_text()
        if not first_page_intro:
            return chunks

        normalized_intro = self._normalize_for_overlap(first_page_intro)
        if not normalized_intro:
            return chunks

        repaired_chunks: list[dict[str, str]] = []
        intro_repaired = False

        for chunk in chunks:
            header_key = self._normalize_heading_candidate(chunk["header"])
            if not intro_repaired and header_key.endswith("introduction"):
                current_content = chunk["content"].strip()
                normalized_current = self._normalize_for_overlap(current_content)
                canonical_current = self._canonical_alnum(current_content[:300])
                canonical_intro = self._canonical_alnum(first_page_intro[:300])

                if canonical_intro and (
                    canonical_current.startswith(canonical_intro[:60])
                    or canonical_current.startswith(canonical_intro[1:61])
                    or canonical_current.startswith(canonical_intro[2:62])
                ):
                    intro_repaired = True
                    repaired_chunks.append(chunk)
                    continue

                if normalized_current and normalized_current in normalized_intro:
                    prefix = normalized_intro.split(normalized_current, 1)[0].strip()
                    if prefix:
                        chunk = {
                            "header": chunk["header"],
                            "content": f"{prefix}\n\n{current_content}",
                        }
                elif normalized_current and not normalized_current.startswith(normalized_intro[:80]):
                    chunk = {
                        "header": chunk["header"],
                        "content": f"{first_page_intro}\n\n{current_content}",
                    }

                intro_repaired = True

            repaired_chunks.append(chunk)

        return repaired_chunks

    def load_and_clean(self) -> dict[str, Any]:
        """Run the full PDF cleaning pipeline and return text, chunks, and LangChain documents."""
        merged_text = self._merge_pages()
        merged_text = self._remove_sections(merged_text)
        merged_text = self._remove_algorithm_blocks(merged_text)
        merged_text = self._inject_missing_front_matter(merged_text)
        chunks = self._chunk_by_headers(merged_text)
        chunks = self._repair_front_matter_chunks(chunks)
        chunks = self._repair_first_introduction_chunk(chunks)
        chunks = self._repair_section_prefixes(chunks)

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
    """Convenience wrapper that cleans one PDF without instantiating the class manually."""
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
