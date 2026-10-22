
import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Ollama configuration
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text:latest",
)

# Data directories
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Text chunking settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Retrieval settings
TOP_K = 4
MAX_RETRIEVAL_ATTEMPTS = 2
