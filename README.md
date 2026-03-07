# Technical PDF RAG Assistant with Hybrid Retrieval and Reranking

A citation-grounded Retrieval-Augmented Generation (RAG) assistant for technical machine learning and security PDFs.  
This project parses PDFs, chunks and indexes their content, retrieves relevant passages for a user query, optionally improves retrieval with hybrid search and reranking, and generates answers grounded in source text with citations.

The system is designed as a practical portfolio project for applied GenAI, ML, and cloud-oriented roles, with a strong focus on grounded generation, retrieval quality, and reproducible evaluation.

---

## Project Goals

This project aims to:

- Build a RAG pipeline over technical PDFs
- Preserve document metadata such as filename and page number for citation grounding
- Improve retrieval quality beyond a naive vector-search baseline
- Evaluate both retrieval and answer quality
- Package the system in a clean, modular structure suitable for extension and deployment
- Align the design with AWS-relevant cloud components for future production use

---

## Key Features

- PDF parsing with page-level metadata
- Document chunking for retrieval-ready indexing
- Dense retrieval using vector embeddings
- Optional hybrid retrieval with dense + sparse search
- Optional reranking of retrieved results
- Citation-grounded answer generation
- Evaluation pipeline for retrieval and answer quality
- Clear project structure for experimentation and deployment
- AWS-aligned architecture planning for scaling the system

---

## Example Use Case

A user asks:

> What are common defenses against prompt injection in RAG systems?

The assistant should:

1. Search across the indexed PDF collection
2. Retrieve the most relevant chunks
3. Optionally rerank the retrieved results
4. Generate an answer only from the retrieved evidence
5. Return citations such as filename and page number

---

## Project Structure

```text
rag-pdf-assistant/
│
├── data/
│   ├── raw_pdfs/
│   ├── processed/
│   └── eval/
│
├── notebooks/
│   ├── 01_pdf_parsing.ipynb
│   ├── 02_baseline_retrieval.ipynb
│   ├── 03_reranking_hybrid.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── ingest.py
│   ├── parse_pdf.py
│   ├── chunking.py
│   ├── embed.py
│   ├── retrieve.py
│   ├── rerank.py
│   ├── generate.py
│   ├── evaluate.py
│   ├── app.py
│   └── utils.py
│
├── outputs/
│   ├── index/
│   ├── logs/
│   ├── samples/
│   └── eval_results/
│
├── README.md
├── requirements.txt
└── .gitignore
```
---

## Folder and File Explanation

### `data/`

Stores the project inputs and prepared datasets.

#### `data/raw_pdfs/`
Contains the original PDF files used as the document corpus.

Examples:
- ML security papers
- RAG papers
- LLM safety papers
- cloud security or architecture whitepapers

These files are kept untouched as the raw source documents.

#### `data/processed/`
Stores intermediate processed artifacts generated from the raw PDFs.

Examples:
- extracted page-level text
- cleaned JSON records
- chunked documents with metadata
- serialized processed corpora

This avoids reparsing every PDF each time the project runs.

#### `data/eval/`
Contains the evaluation dataset used to measure system performance.

Examples:
- test questions
- expected relevant documents/pages
- gold evidence chunks
- expected answer points
- retrieval relevance labels

This folder supports reproducible evaluation and makes the project more than a simple demo.

### `notebooks/`

Contains exploratory notebooks for testing, experimentation, and analysis.

#### `01_pdf_parsing.ipynb`
Used to inspect and validate PDF parsing quality.

Typical checks:
- extracted text quality
- page boundary preservation
- metadata completeness
- parsing issues with figures, references, or tables

#### `02_baseline_retrieval.ipynb`
Implements and tests the first working retrieval pipeline.

Typical tasks:
- chunking
- embedding generation
- vector indexing
- top-k retrieval inspection

#### `03_reranking_hybrid.ipynb`
Used to test retrieval improvements over the baseline.

Possible comparisons:
- dense retrieval only
- BM25 only
- hybrid retrieval
- hybrid retrieval with reranking

#### `04_evaluation.ipynb`
Used for running evaluation experiments and visualizing results.

Possible outputs:
- Recall@k
- MRR
- hit rate
- answer-quality tables
- qualitative error analysis

### `src/`

Contains the core reusable Python code for the system.

#### `ingest.py`
Entry point for preparing the PDF corpus.

Responsibilities:
- scan the raw PDF directory
- call parsing and preprocessing steps
- prepare documents for indexing

#### `parse_pdf.py`
Extracts page-level text and metadata from PDFs.

Responsibilities:
- open PDF files
- read text page by page
- preserve source metadata such as filename and page number

#### `chunking.py`
Splits extracted text into retrieval-friendly chunks.

Responsibilities:
- create text chunks of controlled size
- maintain overlap where needed
- preserve source metadata for citation grounding

#### `embed.py`
Generates vector embeddings for chunks and queries.

Responsibilities:
- load the embedding model
- encode text chunks into vectors
- save or pass embeddings for indexing

#### `retrieve.py`
Implements the retrieval layer.

Responsibilities:
- perform dense retrieval
- optionally support hybrid retrieval with sparse search
- return top-k candidate chunks for a query

#### `rerank.py`
Improves the order of initially retrieved candidates.

Responsibilities:
- rescore retrieved chunks with a reranker
- reorder results based on query relevance

This is useful when the initial retrieval is broad but imperfect.

#### `generate.py`
Produces the final grounded answer.

Responsibilities:
- build prompts from retrieved evidence
- instruct the LLM to answer only from provided context
- include source-backed citations in the response

#### `evaluate.py`
Runs retrieval and answer-quality evaluation.

Responsibilities:
- process test questions
- compare outputs to reference labels
- compute metrics such as Recall@k and MRR
- support qualitative and quantitative analysis

#### `app.py`
Provides the user-facing interface.

Possible options:
- Streamlit app
- Gradio app
- lightweight local interface

Responsibilities:
- accept user questions
- trigger retrieval and generation
- display answer and citations

#### `utils.py`
Contains shared helper functions used across modules.

Examples:
- file loading utilities
- metadata formatting helpers
- text cleanup
- path and logging utilities

### `outputs/`

Stores generated outputs and artifacts produced by the system.

#### `outputs/index/`
Stores the retrieval index and related metadata.

Examples:
- FAISS index files
- chunk metadata mappings
- sparse retrieval structures

#### `outputs/logs/`
Stores run logs and debugging information.

Examples:
- ingestion logs
- retrieval logs
- evaluation logs
- error traces

#### `outputs/samples/`
Stores sample queries and outputs for demos or documentation.

Examples:
- saved answer examples
- retrieved context examples
- screenshots or JSON output snapshots

#### `outputs/eval_results/`
Stores evaluation artifacts.

Examples:
- CSV metric tables
- JSON summaries
- per-question evaluation results
- plots and charts

## End-to-End Pipeline

The project follows this high-level pipeline:

1. **Ingest PDFs** from `data/raw_pdfs/`
2. **Parse text and metadata** page by page
3. **Chunk documents** into smaller retrieval units
4. **Generate embeddings** for chunks
5. **Build retrieval index**
6. **Accept user query**
7. **Retrieve top-k candidate chunks**
8. **Optionally rerank results**
9. **Generate grounded answer**
10. **Return answer with citations**
11. **Evaluate system performance** on a labeled test set

## Baseline vs Improved Retrieval

The project is intended to compare multiple retrieval setups.

### Baseline
- dense vector retrieval only

### Improved Variants
- BM25 sparse retrieval
- hybrid retrieval combining sparse and dense methods
- cross-encoder reranking on top of retrieved results

This allows systematic comparison of retrieval quality instead of relying on intuition alone.

## Evaluation Plan

Evaluation is split into two parts.

### 1. Retrieval Evaluation
Measures whether the system retrieves relevant evidence.

Possible metrics:
- Recall@k
- Precision@k
- Hit@k
- Mean Reciprocal Rank (MRR)

### 2. Answer Evaluation
Measures the quality of the generated response.

Possible criteria:
- groundedness
- relevance
- completeness
- citation correctness
- faithfulness to retrieved context

The evaluation set should include technical questions paired with expected evidence and key answer points.

## Suggested Tech Stack

Core components may include:

- Python
- PyMuPDF or pdfplumber for PDF parsing
- sentence-transformers for embeddings
- FAISS for dense indexing
- BM25 for sparse retrieval
- cross-encoder reranker for reranking
- Streamlit or Gradio for the interface
- pandas / numpy / scikit-learn for evaluation and analysis

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd rag-pdf-assistant
pip install -r requirements.txt
```

## Author

**Pegah Mansourian**  
Applied ML / AI researcher focused on trustworthy AI, anomaly detection, and security-oriented machine learning systems.