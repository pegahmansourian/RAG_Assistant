# Technical PDF RAG Assistant with LangChain

A citation-grounded Retrieval-Augmented Generation (RAG) assistant for technical machine learning and security PDFs.

This project loads technical PDFs, preserves page-level metadata, chunks documents using LangChain text splitters, builds vector indexes with configurable embedding models, retrieves relevant passages for a user query, and generates grounded answers with source references.

The system is designed as a portfolio-ready applied GenAI project for ML, GenAI, and cloud-oriented roles, with a focus on modular design, retrieval quality, reproducible experiments, and deployable architecture.

---

## Project Goals

This project aims to:

- Build a LangChain-based RAG pipeline over technical PDFs
- Preserve source metadata such as filename and page number for grounded citations
- Compare different embedding models and retrieval strategies
- Support multiple LLM backends for answer generation
- Evaluate retrieval and answer quality in a reproducible way
- Package the system in a clean structure suitable for extension and deployment

---

## Key Features

- PDF loading with page-level metadata
- LangChain `Document`-based pipeline
- Chunking with `RecursiveCharacterTextSplitter`
- Configurable embedding backends
- FAISS vector store indexing
- Similarity and MMR retrieval
- Citation-grounded answer generation
- Swappable LLM backend
- Streamlit user interface
- Evaluation-ready project structure

---

## Example Use Case

A user asks:

> What are common defenses against prompt injection in RAG systems?

The assistant should:

1. Search across the indexed PDF collection
2. Retrieve the most relevant chunks
3. Pass only retrieved evidence to the LLM
4. Generate a grounded answer
5. Return sources such as filename and page number

---

## Project Structure

```text
rag-pdf-assistant/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── eval/
│
├── notebooks/
│   ├── 01_parse_and_inspect.ipynb
│   ├── 02_chunk_and_index.ipynb
│   ├── 03_retrieval_experiments.ipynb
│   └── 04_evaluation.ipynb
│
├── outputs/
│   ├── indexes/
│   └── experiments/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retrievers.py
│   ├── reranking.py
│   ├── llms.py
│   ├── prompts.py
│   ├── rag_chain.py
│   ├── evaluation.py
│   └── utils.py
├── app.py
├── eval_dashboard.py
├── experiment.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Folder and File Explanation

### `data/`

Stores project inputs and evaluation assets.

#### `data/raw_pdfs/`
Contains the original PDF files used as the document corpus.

Examples:
- RAG papers
- LLM safety papers
- ML security papers
- technical architecture documents

#### `data/processed/`
Stores optional intermediate artifacts generated during development.

Examples:
- serialized parsed documents
- chunk inspection files
- intermediate preprocessing outputs

#### `data/eval/`
Stores evaluation datasets.

Examples:
- benchmark questions
- expected sources
- expected keywords
- relevance labels

---

### `notebooks/`

Contains exploratory notebooks for step-by-step development and analysis.

#### `Evaluation.ipynb`
Used to compare retrieval and answer-quality evaluation results between retrieval-only and reranker.

#### `RAG Test.ipynb`
Used to test RAG pipeline from loading PDF files to RAG query.

---

### `outputs/`

Stores generated artifacts and experiment results.

#### `outputs/indexes/`
Stores saved FAISS indexes and related vector store files.

#### `outputs/experiments/`
Stores evaluation summaries, metrics, and experiment outputs.

---

### `src/`

Contains the core reusable code for the system.

#### `config.py`
Central configuration file.

#### `loaders.py`
Loads PDF files into LangChain `Document` objects.

Responsibilities:
- read PDF files from disk
- preserve page-level metadata
- prepare documents for downstream preprocessing and chunking

#### `preprocessing.py`
Handles light text cleanup before chunking.

Examples:
- whitespace normalization
- ligature cleanup
- extraction cleanup for technical PDFs

#### `chunking.py`
Chunks LangChain documents using text splitters.

Responsibilities:
- build `RecursiveCharacterTextSplitter`
- split documents into retrieval-friendly chunks
- preserve metadata for citation grounding

#### `embeddings.py`
Builds embedding backends.

Responsibilities:
- initialize embedding model by key
- support comparison of multiple embedding models
- isolate embedding-provider logic from the rest of the pipeline

#### `vectorstore.py`
Builds, saves, and loads FAISS vector stores.

Responsibilities:
- create vector store from chunked documents
- save indexes locally
- reload indexes for reuse

#### `retrievers.py`
Defines retrieval strategies.

Responsibilities:
- build similarity retrievers
- build MMR retrievers
- execute retrieval for a query

#### `reranking.py`
For second-stage reranking.

Responsibilities:
- rerank top retrieved chunks
- improve retrieval ordering with a cross-encoder or reranker model

#### `llms.py`
Builds the answer-generation LLM backend.

Responsibilities:
- initialize selected LLM provider
- support local or hosted LLM backends
- keep generation backend configurable

#### `rag_chain.py`
Core RAG orchestration module.

Responsibilities:
- retrieve relevant documents
- format context for the prompt
- run the LLM on retrieved evidence
- return answer plus sources

#### `evaluation.py`
Runs evaluation experiments.

Responsibilities:
- load evaluation questions
- run retrieval and generation
- compare outputs to expected references
- compute retrieval and answer-quality metrics

---

### `app.py`

Root-level Streamlit application for interactive question answering over the indexed PDF corpus.

### `eval_dashboard.py`

Streamlit application for evaluating generated answers based on the evaluation dataset. It supports human-based feedback for completeness, correctness, and hallucination.

### `experiment.py`

Logic for experimentation tracking using MLFlow.

---

## End-to-End Pipeline

The project follows this high-level flow:

1. Load PDFs from `data/raw_pdfs/`
2. Convert pages into LangChain `Document` objects with metadata
3. Apply light preprocessing
4. Chunk documents with LangChain text splitters
5. Build embeddings for chunks
6. Create and save a FAISS vector store
7. Accept a user query
8. Retrieve top-k relevant chunks
9. Pass retrieved context into the RAG prompt
10. Generate a grounded answer with source references
11. Evaluate retrieval and answer quality on a labeled test set

---

## Retrieval Configurations

The project is designed to support multiple retrieval setups.

### Baseline
- dense vector retrieval with FAISS similarity search

### Variants
- MMR retrieval
- alternative embedding models
- future reranking
- future hybrid retrieval

This structure makes it easy to compare retrieval quality systematically instead of relying only on manual inspection.

---

## Tech Stack

Core technologies used in this project include:

- Python
- LangChain
- PyMuPDF / LangChain PDF loaders
- sentence-transformers or other embedding backends
- FAISS
- Streamlit
- pandas / numpy / scikit-learn
- local or API-based LLM backends
- MLFlow

---

## Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd rag-pdf-assistant
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root for any required API keys.

Example:

```env
COHERE_API_KEY=your_cohere_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

You only need the keys for providers you actually use.

---

## Running the App

From the project root:

```bash
streamlit run app.py
```
This launches the Streamlit interface for querying the indexed PDF collection.

for evaluation dashboard:
```bash
streamlit run eval_dashboard.py
```
This launches the Streamlit interface for RAG evaluation dashboard.

For experiments:
```bash
mlflow ui
python experiment.py --config configs/exp_baseline.yaml 
```
This runs an experiment with configs defined in `configs/exp_baseline.yaml` and logs the results in MLFlow dashboard.

---

## Typical Development Workflow

1. Add PDFs to `data/raw_pdfs/`
2. Test loading and chunking in notebooks
3. Build embeddings and vector store
4. Test retrieval quality
5. Run the RAG chain end to end
6. Launch the Streamlit app
7. Run evaluation experiments
8. Compare embeddings and retrieval settings

---

## Why This Project Structure Matters

This structure is designed to support both learning and professional presentation:

- `src/` contains reusable application logic
- `data/` holds inputs and evaluation assets
- `outputs/` stores generated artifacts
- `notebooks/` support experimentation
- `app.py` provides a demo interface
- `config.py` keeps the system configurable
- `evaluation.py` makes the project measurable

This makes the project easier to debug, extend, compare experimentally, and present in interviews or on GitHub.

---

## Future Improvements

Potential extensions include:

- hybrid retrieval with BM25 + dense search
- richer source formatting in the UI
- IEEE paper style-specific data cleaning
- AWS-aligned deployment architecture
- automatic answer grading
- benchmark comparison across local and hosted LLMs

---

## Author

**Pegah Mansourian**  
Applied ML / AI researcher focused on trustworthy AI, anomaly detection, and security-oriented machine learning systems.

