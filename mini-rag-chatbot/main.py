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
    description="A simple RAG chatbot supporting English and Arabic, powered by a multilingual LLM.",
    version="1.1.0"
)

# Configure logging to provide detailed insight into the application's flow
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API Data Validation ---
class ChatRequest(BaseModel):
    """Defines the structure for incoming chat requests."""
    question: str

class ChatResponse(BaseModel):
    """Defines the structure for outgoing chat responses."""
    response: str

# --- Application Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """
    This function runs when the application starts.
    It's used here to log the successful initialization of core components.
    """
    logger.info("Application starting up...")
    if not retriever_instance or not generator_instance.generator_pipeline:
        logger.critical("A CRITICAL COMPONENT (Retriever or Generator) FAILED TO INITIALIZE. The application may not function correctly.")
    else:
        logger.info("Retriever and Generator have been successfully initialized.")
    logger.info("FastAPI application started successfully. Awaiting requests...")

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    The main endpoint to handle user questions. It performs the full RAG pipeline:
    1.  Detects the language of the question.
    2.  Retrieves relevant context from the knowledge base.
    3.  Generates a final answer using the LLM and the retrieved context.
    """
    question = request.question
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    logger.info(f"Received new request with question: '{question}'")

    try:
        # --- Step 1: Detect Language ---
        # This step is crucial for selecting the correct index and prompt.
        try:
            lang = detect(question)
            logger.info(f"Detected language: {lang}")
            
            # Handle unsupported languages gracefully
            if lang not in ['en', 'ar']:
                logger.warning(f"Unsupported language '{lang}' detected. Defaulting to 'en' for processing.")
                lang = 'en'
        except LangDetectException:
            logger.warning("Language detection failed. Defaulting to English.")
            lang = 'en' # Default language if detection is uncertain.

        # --- Step 2: Retrieve Relevant Context ---
        # Use the retriever to find documents semantically similar to the question.
        retrieved_context = retriever_instance.retrieve(question, lang, top_k=TOP_K)
        
        # Provide a language-appropriate fallback response ---
        # If no context is found, the system can't answer. We inform the user in their own language.
        if not retrieved_context:
            logger.warning("No relevant context found for the query. Returning a fallback message.")
            response_text = ("لم أتمكن من العثور على معلومات ذات صلة للإجابة على سؤالك."
                           if lang == 'ar'
                           else "I could not find relevant information to answer your question.")
            return ChatResponse(response=response_text)

        # --- Step 3: Generate Response ---
        # The generator receives the question, the context, AND the language.
        # Pass the detected 'lang' to the generator ---
        # To ensure the LLM gets the correct prompt template.
        final_response = generator_instance.generate_response(question, retrieved_context, lang)

        logger.info(f"Generated response: '{final_response}'")
        return ChatResponse(response=final_response)

    except Exception as e:
        # Catch-all for any unexpected errors during the process.
        logger.error(f"An unexpected error occurred while processing the request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred. Please check the logs.")

# uvicorn main:app --reload
