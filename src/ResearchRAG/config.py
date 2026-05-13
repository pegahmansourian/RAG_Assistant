from pathlib import Path
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

OUTPUTS_DIR = ROOT_DIR / "outputs"
INDEX_DIR = OUTPUTS_DIR / "indexes"
LOG_DIR = OUTPUTS_DIR / "logs"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
EVAL_RESULTS_DIR = OUTPUTS_DIR / "eval_results"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODELS = {
    "miniLM": {
        "provider": "huggingface",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "mpnet": {
        "provider": "huggingface",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
    },
    "bge_base": {
        "provider": "huggingface",
        "model_name": "BAAI/bge-base-en-v1.5",
    },
    "cohere_v3": {
        "provider": "cohere",
        "model_name": "embed-english-v3.0",
    },
    "cohere_v4": {
        "provider": "cohere",
        "model_name": "embed-v4.0",
    },
}

DEFAULT_EMBEDDING_KEY = "miniLM"

LLM_MODELS = {
    "miniLM": {
        "provider": "openai",
        "model_name": "gpt-4o-mini",
    },
    "cohere": {
        "provider": "cohere",
        "model_name": "command-a-03-2025",
    },
    "mistral": {
        "provider": "ollama",
        "model_name": "mistral",
    },
    "llama3": {
        "provider": "ollama",
        "model_name": "llama3",
    },
    "qwen3": {
        "provider": "ollama",
        "model_name": "qwen3",
    },
}

DEFAULT_LLM_KEY = "mistral"

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

VECTORSTORE_NAME = "faiss"

RETRIEVAL_K = 3
RERANK_TOP_N = 3
RERANK_BASE_K = 15

LLM_PROVIDER = "openai"
LLM_MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_CONTEXT_DOCS = 5

DEFAULT_PROCESSED_FILE = PROCESSED_DIR / "parsed_documents.json"
DEFAULT_CHUNKS_FILE = PROCESSED_DIR / "chunked_documents.json"
DEFAULT_INDEX_PATH = INDEX_DIR / "faiss_index"

SUPPORTED_PDF_GLOB = "*.pdf"


