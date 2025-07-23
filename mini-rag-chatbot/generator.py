
import torch
from transformers import pipeline
from config import LLM_MODEL, HF_TOKEN
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Generator:
    """
    Handles response generation using a pre-trained language model.
    """
    def __init__(self):
        logger.info("Initializing Generator...")
        try:
            self.generator_pipeline = pipeline(
                "text-generation",
                model=LLM_MODEL,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                token=HF_TOKEN
            )
            logger.info(f"Generator initialized with model: {LLM_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load the language model: {e}")
            self.generator_pipeline = None

    def generate_response(self, question: str, context: list[str], lang: str) -> str:
        """
        Generates a response based on the question, context, and language.
        """
        if not self.generator_pipeline:
            return "Error: Language model is not available."

        context_str = "\n".join(f"- {doc}" for doc in context)
        
        # Dynamic prompt based on language ---
        if lang == 'ar':
            prompt = f"""
            أنت مساعد ذكي. أجب على السؤال التالي باللغة العربية بناءً على السياق المقدم فقط.
            إذا كان السياق لا يحتوي على إجابة، قل "المعلومات غير متوفرة في السياق".

            السياق:
            {context_str}

            السؤال: {question}

            الجواب:
            """
        else: # Default to English
            prompt = f"""
            You are a helpful assistant. Answer the following question in English based only on the provided context.
            If the context does not contain the answer, say "The information is not available in the context".

            Context:
            {context_str}

            Question: {question}

            Answer:
            """
        
        logger.info(f"Generating response for language '{lang}' with the following prompt:")
        logger.info(prompt)
        
        try:
            generated = self.generator_pipeline(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                eos_token_id=self.generator_pipeline.tokenizer.eos_token_id,
                pad_token_id=self.generator_pipeline.tokenizer.pad_token_id
            )
            response = generated[0]['generated_text']
            
            # Clean up the response to only return the answer part
            answer_marker = "الجواب:" if lang == 'ar' else "Answer:"
            answer = response.split(answer_marker)[-1].strip()
            return answer
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return "I am sorry, but I encountered an error while generating a response."

# Instantiate generator on startup
generator_instance = Generator()
