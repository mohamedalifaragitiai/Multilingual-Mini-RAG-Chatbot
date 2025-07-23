# retriever.py
import faiss
import numpy as np
import json
import os
import logging
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, ENGLISH_DOCS_PATH, ARABIC_DOCS_PATH, ENGLISH_INDEX_PATH, ARABIC_INDEX_PATH, INDEX_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    """
    Handles document retrieval using sentence embeddings and FAISS.
    """
    def __init__(self):
        logger.info("Initializing Retriever...")
        # Load the multilingual sentence transformer model[4].
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.documents = {}
        self.indices = {}
        self._load_and_index_all()

    def _load_documents(self, lang):
        """Loads documents from a JSON file."""
        path = ENGLISH_DOCS_PATH if lang == "en" else ARABIC_DOCS_PATH
        try:
            with open(path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                self.documents[lang] = {doc['id']: doc['text'] for doc in docs}
                logger.info(f"Loaded {len(docs)} documents for language: {lang}")
        except FileNotFoundError:
            logger.error(f"Document file not found at {path}")
            self.documents[lang] = {}

    def _create_index(self, lang):
        """Creates a FAISS index for the loaded documents."""
        doc_texts = list(self.documents[lang].values())
        if not doc_texts:
            logger.warning(f"No documents to index for language: {lang}")
            return

        logger.info(f"Creating embeddings for {lang} documents...")
        # The model encodes text into high-dimensional vectors[1].
        embeddings = self.embedding_model.encode(doc_texts, convert_to_tensor=False)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)
        
        doc_ids = np.array([int(id.split('_')[1]) for id in self.documents[lang].keys()])
        
        index.add_with_ids(embeddings, doc_ids)
        self.indices[lang] = index
        logger.info(f"FAISS index created for {lang} with {index.ntotal} vectors.")
        
        # Save the index to disk
        if not os.path.exists(INDEX_DIR):
            os.makedirs(INDEX_DIR)
        index_path = ENGLISH_INDEX_PATH if lang == "en" else ARABIC_INDEX_PATH
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")

    def _load_index(self, lang):
        """Loads a FAISS index from disk."""
        index_path = ENGLISH_INDEX_PATH if lang == "en" else ARABIC_INDEX_PATH
        try:
            self.indices[lang] = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index from {index_path}")
        except RuntimeError:
            logger.warning(f"Index file not found at {index_path}. Creating a new one.")
            self._create_index(lang)

    def _load_and_index_all(self):
        """Loads all documents and creates or loads indices."""
        for lang in ['en', 'ar']:
            self._load_documents(lang)
            index_path = ENGLISH_INDEX_PATH if lang == "en" else ARABIC_INDEX_PATH
            if os.path.exists(index_path):
                self._load_index(lang)
            else:
                self._create_index(lang)
    
    def retrieve(self, query: str, lang: str, top_k: int) -> list[str]:
        """
        Retrieves the top_k most relevant document chunks for a given query.
        """
        if lang not in self.indices:
            logger.error(f"No index available for language: {lang}")
            return []
            
        logger.info(f"Retrieving documents for query: '{query}' in language: {lang}")
        query_embedding = self.embedding_model.encode([query])
        
        index = self.indices[lang]
        distances, ids = index.search(query_embedding, top_k)
        
        retrieved_docs = []
        doc_map = self.documents[lang]
        
        # Determine the correct prefix based on the language.
        # The original IDs used "eng" and "ar".
        prefix = "eng" if lang == "en" else "ar"
        # --- FIX ENDS HERE ---

        for i in range(len(ids[0])):
            doc_id_num = ids[0][i]
            if doc_id_num != -1:
                # --- APPLY THE FIX HERE ---
                # Reconstruct the ID string with the correct prefix.
                doc_id_str = f"{prefix}_{str(doc_id_num).zfill(2)}"
                if doc_id_str in doc_map:
                    retrieved_docs.append(doc_map[doc_id_str])

        logger.info(f"Retrieved {len(retrieved_docs)} documents.")
        return retrieved_docs
# Instantiate retriever on startup
retriever_instance = Retriever()
