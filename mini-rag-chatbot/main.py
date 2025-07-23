# main.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langdetect import detect, LangDetectException

from retriever import retriever_instance
from generator import generator_instance
from config import TOP_K

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Multilingual Mini RAG Chatbot",
    description="A simple RAG chatbot supporting English and Arabic.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    response: str

# --- API Endpoints ---
@app.on_event("startup")
async def startup_event():
    """
    Log app startup and ensure models are loaded.
    This is a good place to warm up the models if needed.
    """
    logger.info("Application starting up...")
    if not retriever_instance or not generator_instance.generator_pipeline:
        logger.error("A critical component (Retriever or Generator) failed to initialize.")
        # In a real app, you might want to prevent startup.
    else:
        logger.info("Retriever and Generator are initialized.")
    logger.info("FastAPI application started successfully.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main endpoint to handle user questions.
    """
    question = request.question
    logger.info(f"Received new request with question: '{question}'")

    try:
        # 1. Detect Language
        try:
            lang = detect(question)
            logger.info(f"Detected language: {lang}")
            if lang not in ['en', 'ar']:
                raise HTTPException(status_code=400, detail=f"Unsupported language '{lang}'. Only 'en' and 'ar' are supported.")
        except LangDetectException:
            logger.warning("Language detection failed. Defaulting to English.")
            lang = 'en' # Default language if detection fails

        # 2. Retrieve Relevant Context
        retrieved_context = retriever_instance.retrieve(question, lang, top_k=TOP_K)
        if not retrieved_context:
            logger.warning("No relevant context found for the query.")
            # Fallback response if no context is found
            return ChatResponse(response="I could not find relevant information to answer your question.")

        # 3. Generate Response
        final_response = generator_instance.generate_response(question, retrieved_context)

        logger.info(f"Generated response: '{final_response}'")
        return ChatResponse(response=final_response)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# To run the app, use the command: uvicorn main:app --reload
