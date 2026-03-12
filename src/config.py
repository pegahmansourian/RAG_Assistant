from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"

OUTPUTS_DIR = ROOT_DIR / "outputs"
INDEX_DIR = OUTPUTS_DIR / "indexes"
LOG_DIR = OUTPUTS_DIR / "logs"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
EVAL_RESULTS_DIR = OUTPUTS_DIR / "eval_results"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_NAME = "faiss"

RETRIEVAL_K = 5
RERANK_TOP_K = 10

LLM_PROVIDER = "openai"
LLM_MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_CONTEXT_DOCS = 5

DEFAULT_PROCESSED_FILE = PROCESSED_DIR / "parsed_documents.json"
DEFAULT_CHUNKS_FILE = PROCESSED_DIR / "chunked_documents.json"
DEFAULT_INDEX_PATH = INDEX_DIR / "faiss_index"

SUPPORTED_PDF_GLOB = "*.pdf"