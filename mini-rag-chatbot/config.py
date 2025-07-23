# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Environment ---
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model Configuration ---
# Using a multilingual model for embeddings is crucial for supporting both languages[1].
# This model maps sentences to a 384-dimensional vector space[2].
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# A smaller, multilingual model suitable for CPU inference.
# Note: For production-grade performance, a larger model might be necessary,
# but this is a good starting point for memory-constrained environments.
# Using a smaller model like opt-350m is a trade-off between performance and resource usage.
LLM_MODEL = "facebook/opt-350m"

# --- Data and Index Paths ---
DATA_DIR = "data"
ENGLISH_DOCS_PATH = os.path.join(DATA_DIR, "english_docs.json")
ARABIC_DOCS_PATH = os.path.join(DATA_DIR, "arabic_docs.json")

INDEX_DIR = "faiss_indices"
ENGLISH_INDEX_PATH = os.path.join(INDEX_DIR, "english.faiss")
ARABIC_INDEX_PATH = os.path.join(INDEX_DIR, "arabic.faiss")

# --- RAG Configuration ---
TOP_K = 3 # Number of relevant documents to retrieve
