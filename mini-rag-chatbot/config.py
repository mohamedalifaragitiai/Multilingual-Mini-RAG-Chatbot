# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment ---
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model Configuration ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# bloom-560m has strong support for Arabic and many other languages.
LLM_MODEL = "bigscience/bloom-560m"

# --- Data and Index Paths ---
DATA_DIR = "data"
ENGLISH_DOCS_PATH = os.path.join(DATA_DIR, "english_docs.json")
ARABIC_DOCS_PATH = os.path.join(DATA_DIR, "arabic_docs.json")

INDEX_DIR = "faiss_indices"
ENGLISH_INDEX_PATH = os.path.join(INDEX_DIR, "english.faiss")
ARABIC_INDEX_PATH = os.path.join(INDEX_DIR, "arabic.faiss")

# --- RAG Configuration ---
TOP_K = 3
