# Technical PDF RAG Assistant with LangChain

A citation-grounded Retrieval-Augmented Generation (RAG) assistant for technical machine learning and security PDFs.

This project processes technical PDFs into structured retrieval-ready artifacts, builds configurable vector indexes, retrieves relevant context for user queries, and generates citation-grounded answers using multiple LLM backends.

The system is designed as a portfolio-ready applied GenAI project focused on:

- modular RAG architecture
- retrieval quality experimentation
- reproducible workflows
- production-style engineering practices
- scalable ingestion and indexing pipelines

---

## Project Goals

This project aims to:
- Build a modular LangChain-based RAG pipeline
- Support configurable embedding and LLM backends
- Improve retrieval quality through reranking and retrieval strategies
- Preserve structured metadata for grounded citations
- Support incremental indexing workflows
- Create a clean engineering-oriented codebase suitable for extension and deployment
- Enable reproducible experimentation and evaluation

---

## Key Features
- Modular ingestion, retrieval, and generation architecture
- PDF ETL preprocessing pipeline
- Section-aware document chunking
- Processed PDF caching
- Configurable embedding backends
- FAISS vector indexing
- Similarity and MMR retrieval
- Cross-encoder reranking
- Incremental FAISS index updates
- Citation-grounded answer generation
- Swappable LLM backends
- Streamlit user interface
- Structured application logging
- Automatic processed-PDF synchronization
- MLflow experiment tracking
- reproducible retrieval experiments
- evaluation-ready architecture

---

## Example Use Case

A user asks:

> What are common attacks on CAV networks?

The assistant will:

1. Retrieve relevant chunks from indexed technical PDFs
2. Optionally rerank retrieved passages
3. Pass grounded context into the LLM
4. Generate a citation-grounded answer
5. Return references linked to retrieved sources

---

## Project Structure

```text
rag-pdf-assistant/
│
├── data/
│   ├── eval/
│   ├── processed/
│   └── raw/
│
├── notebooks/
│   ├── Evaluation.ipynb
│   └── RAG Test.ipynb
│
├── outputs/
│   ├── experiments/
│   ├── indexes/
│   └── logs/
│
src/
├── ResearchRAG/
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   └── vectorstore.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluation.py
│   │   └── experiment.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llms.py
│   │   └── rag_chain.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── loaders.py
│   │   ├── pdf_cleaning.py
│   │   └── pdf_etl.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── reranking.py
│   │   └── retriever.py
│   │
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   └── eval_dashboard.py
│   │
│   └── utils/
│   │   ├── __init__.py
│   │   └── logging_config.py
│   │   
│   ├── __init__.py
│   └── config.py
│
└── README.md
```

---

## End-to-End Pipeline

The system follows this workflow:

1. Upload or place PDFs into `data/raw/`
2. Run PDF ETL preprocessing
3. Clean and normalize extracted text
4. Chunk content into structured retrieval units
5. Cache processed artifacts into JSON files
6. Generate embeddings
7. Create or update a FAISS vector index
8. Retrieve relevant chunks
9. Optionally rerank retrieved passages
10. Generate grounded answers with citations

---

## PDF ETL Pipeline

The ingestion system uses a lightweight ETL pipeline for converting raw PDFs into reusable structured artifacts.

The ETL process:

- extracts text from PDFs
- removes noisy sections and formatting artifacts
- repairs section structure
- chunks content using semantic headers
- stores processed outputs for reuse

Processed artifacts are cached under:

> data/processed/

The application automatically synchronizes missing processed files during startup.

---

## Retrieval Pipeline

The project supports multiple retrieval configurations.

### Supported Retrieval Modes
- similarity retrieval
- MMR retrieval
- cross-encoder reranking

### Reranking

The system supports second-stage reranking using HuggingFace cross-encoder models. This improves retrieval quality by:

- rescoring retrieved chunks
- improving ranking precision
- filtering noisy retrieval results

---

## Incremental Index Updates

The application supports incremental FAISS updates. When new PDFs are uploaded:

- only new PDFs are processed
- only new chunks are embedded
- the existing FAISS index is updated in place

This avoids rebuilding the entire vector index for every upload.

---

## Streamlit Application Features

The Streamlit application supports:

- PDF upload/remove directly from the UI
- automatic indexing of uploaded PDFs
- deletion of PDFs from the vector index
- configurable embedding model selection
- configurable LLM backend selection
- optional reranking
- retrieved chunk inspection
- grounded answer generation

----

## Logging

The project uses structured logging across ingestion, indexing, retrieval, and generation modules. 

Features include:

- module-level loggers
- per-run log files
- exception trace logging
- pipeline-stage visibility
- debugging support for ingestion and indexing workflows

Logs are stored under:

> outputs/logs/

---

## Core Modules
### `ingestion/`

Handles document ingestion and preprocessing.

#### `pdf_etl.py`

Runs the PDF ETL pipeline and manages processed artifacts.

#### `pdf_cleaning.py`

Cleans extracted PDF text and repairs section structure.

#### `loaders.py`

Loads processed PDF artifacts into LangChain Document objects.

#### `chunking.py`

Splits documents into retrieval-ready chunks.

---

### `embedding/`

Handles embedding generation and vector indexing.

#### `embeddings.py`

Builds configurable embedding backends.

Supported providers include:

- HuggingFace
- Cohere

#### `evectorstore.py`

Builds, saves, loads, updates, and modifies FAISS indexes.

---

### `retrieval/`

Handles retrieval and reranking logic.

#### `retriever.py`

Builds configurable retrievers for similarity and MMR search.

#### `reranking.py`

Builds cross-encoder reranking pipelines.

---

### `generation/`

Handles LLM orchestration and answer generation.

#### `llms.py`

Builds configurable LLM backends.

Supported providers include:

- OpenAI
- Cohere
- Ollama

#### `rag_chain.py`

Runs the full RAG orchestration pipeline:

- retrieve documents
- format context
- invoke the LLM
- return grounded answers

---

### `utils/`
#### `logging_config.py`

Initializes structured logging and per-run log files.

---

## Supported Model Backends
### Embedding Models

Examples:

- all-MiniLM
- MPNet
- BGE
- Cohere embeddings

### LLM Backends

Examples:

- Mistral
- Llama
- Qwen 
- OpenAI models
- Cohere chat models
- Ollama-hosted local models

---

## Experiment Tracking with MLflow

The project supports experiment tracking using MLflow.

Tracked experiment components may include:
- embedding model selection
- retriever configuration
- reranker usage
- chunking parameters
- evaluation metrics
- generated outputs

This enables reproducible comparison between retrieval and generation configurations.

---

## Tech Stack

Core technologies used in this project:

- Python
- LangChain
- FAISS
- Streamlit
- HuggingFace
- Cohere API
- OpenAI API
- Ollama
- MLflow
- sentence-transformers

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/pegahmansourian/RAG_Assistant
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

## Running the Application

From the project root:

```bash
streamlit run src/ResearchRAG/ui/app.py
```

This launches the Streamlit interface for querying the indexed PDF collection.

---

## Running Experiments

Example experiment run:

```bash
python run_experiment.py --config configs/exp_reranker.yaml
```

To inspect tracked runs locally with MLflow:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

---

## Future Improvements

Potential extensions include:

- hybrid retrieval with BM25 + dense search
- OCR support for scanned PDFs
- automatic answer grading
- metadata-aware retrieval
- conversational memory
- benchmark evaluation suites
- cloud deployment architecture
- async ingestion pipelines

---

## Author

**Pegah Mansourian**  
Applied ML / AI researcher focused on trustworthy AI, anomaly detection, and security-oriented machine learning systems.

---

## Note

ChatGPT AI tools were used as a development aid for code debugging and documentation support. The project structure, implementation, experimentation, and final validation were completed by the author.