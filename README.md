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

> What are common attacks on CAV networks?

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
│   ├── eval/
│   ├── processed/
│   └── raw/
├── notebooks/
│   ├── Evaluation.ipynb
│   └── RAG Test.ipynb
├── outputs/
│   ├── indexes/
│   └── experiments/
├── src/
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── llms.py
│   ├── loaders.py
│   ├── pdf_cleaning.py
│   ├── preprocessing.py
│   ├── rag_chain.py
│   ├── reranking.py
│   ├── retriever.py
│   └── vectorstore.py
│
├── app.py
├── eval_dashboard.py
├── experiment.py
└── README.md
```

---

## Folder and File Explanation

### `app.py`

Root-level Streamlit application for interactive question answering over the indexed PDF corpus.

Responsibilities:
- load or rebuild the vector index
- initialize embedding and LLM backends
- accept user questions
- display grounded answers and retrieved chunks


### `experiment.py`

Script for running structured experiments over the RAG pipeline using config files and MLflow tracking.

Typical responsibilities:
- load experiment config
- build retrieval/generation pipeline
- run evaluation over a labeled eval set
- compute retrieval metrics
- log params, metrics, and artifacts to MLflow

### `eval_dashboard.py`

Dashboard script for inspecting evaluation results and comparing experiment outputs.

---

### `outputs/`

Stores generated artifacts such as saved indexes, evaluation outputs, logs, and experiment results.

---

### `data/`

Stores project inputs and evaluation assets.

#### `data/raw_pdfs/`
Contains the original PDF files used as the document corpus.

#### `data/processed/`
Stores optional intermediate artifacts generated during development.


#### `data/eval/`
Stores evaluation datasets.

Examples:
- benchmark questions
- expected sources

---

### `notebooks/`

Contains notebooks for experimentation, debugging, and evaluation.

#### `RAG_Test.ipynb`
Used for interactive testing of the RAG pipeline, including document loading, chunking, indexing, retrieval, and answer generation.

#### `Evaluation.ipynb`
Used to inspect retrieval and answer-quality evaluation results, test metrics, and analyze experiment outputs.

---

### `outputs/`

Stores generated artifacts and experiment results.

#### `outputs/indexes/`
Stores saved FAISS indexes and related vector store files.

#### `outputs/logs/`
Stores logs from ingestion, indexing, retrieval, or app runs.

#### `outputs/samples/`
Stores sample question-answer outputs and other demo artifacts.

#### `outputs/eval_results/`
Stores evaluation summaries, metrics, and experiment outputs.

---

### `src/`

Contains the core reusable code for the system.

#### `config.py`
Central configuration file.

Responsibilities:
- define project paths
- define chunking defaults
- define embedding model registry
- define retrieval defaults
- load environment variables when needed

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


#### `reranking.py`
Reserved for second-stage reranking.

Possible responsibilities:
- rerank top retrieved chunks
- improve retrieval ordering with a cross-encoder or reranker model

#### `llms.py`
Builds the answer-generation LLM backend.

Responsibilities:
- initialize selected LLM provider
- support local or hosted LLM backends
- keep generation backend configurable

#### `prompts.py`
Stores prompt templates used in the RAG pipeline.

Examples:
- grounded-answering prompt
- citation instructions
- answer-style templates

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

#### `retriever.py`

Defines the retrieval logic for the system.

Typical responsibilities:
- build similarity retriever
- support retrieval configuration
- return top-k relevant chunks for a query


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
- reranking
- future hybrid retrieval

This structure makes it easy to compare retrieval quality systematically instead of relying only on manual inspection.

---

## Evaluation Plan

Evaluation is split into two parts.

### 1. Retrieval Evaluation
Measures whether relevant chunks are retrieved.

Possible metrics:
- Hit@k
- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)

### 2. Answer Evaluation
Measures the quality of the generated response.

Possible criteria:
- groundedness
- relevance
- completeness
- citation correctness
- faithfulness to retrieved context

The evaluation set should include technical questions paired with expected sources and expected answer cues.

---

## Suggested Tech Stack

Core technologies used in this project include:

- Python
- LangChain
- PyMuPDF / LangChain PDF loaders
- RecursiveCharacterTextSplitter
- sentence-transformers or other embedding backends
- FAISS
- Streamlit
- pandas / numpy / scikit-learn
- local or API-based LLM backends
- MLflow for experiment tracking

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

A `.env.example` file can be included as a template.

---

## Running the App

From the project root:

```bash
streamlit run app.py
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

## Typical Development Workflow

1. Add PDFs to the project data location
2. Build embeddings and vector store
3. Test retrieval quality
4. Run the RAG pipeline end to end
5. Launch the Streamlit app with `app.py`
6. Run `experiment.py` for reproducible evaluation and MLflow tracking
7. Use `eval_dashboard.py` to inspect and compare evaluation outputs

---

## Why This Project Structure Matters

This structure is designed to support both learning and professional presentation:

- `src/` contains reusable application logic
- `data/` holds inputs and evaluation assets
- `outputs/` stores generated artifacts
- `notebooks/` support experimentation and evaluation
- `app.py` provides a demo interface
- `config.py` keeps the system configurable
- `evaluation.py` makes the project measurable

This makes the project easier to debug, extend, compare experimentally, and present in interviews or on GitHub.

---

## Future Improvements

Potential extensions include:

- hybrid retrieval with BM25 + dense search
- stronger reranking with a cross-encoder
- upload support in the app
- OCR support for scanned PDFs
- AWS-aligned deployment architecture
- automatic answer grading
- benchmark comparison across local and hosted LLMs

---

## Author

**Pegah Mansourian**  
Applied ML / AI researcher focused on trustworthy AI, anomaly detection, and security-oriented machine learning systems.

---

## Note

ChatGPT AI tools were used as a development aid for code debugging and documentation support. The project structure, implementation, experimentation, and final validation were completed by the author.