# Multilingual Mini RAG Chatbot

This project implements a lightweight, multilingual (English and Arabic) Retrieval-Augmented Generation (RAG) chatbot that runs locally on a CPU. It uses FAISS for efficient document retrieval, Hugging Face Transformers for embeddings and language modeling, and FastAPI to serve the API.

## Features

-   **Multilingual Support**: Handles questions in both English and Arabic.
-   **RAG Pipeline**: Retrieves relevant documents before generating an answer for context-aware responses.
-   **Local & CPU-First**: Designed to run on a standard laptop with 12GB RAM without requiring a GPU.
-   **Modular Codebase**: Organized into logical components for retriever, generator, and API.
-   **Secure**: Uses `.env` to manage sensitive API tokens.
-   **Efficient Retrieval**: Leverages `faiss-cpu` for fast similarity search.

## Project Structure

mini-rag-chatbot/
├── data/
│ ├── arabic_docs.json
│ └── english_docs.json
├── .env
├── config.py
├── generator.py
├── main.py
├── README.md
├── requirements.txt
└── retriever.py

text

## Installation

1.  **Clone the repository:**
    ```
    git clone <your_repository_url>
    cd mini-rag-chatbot
    ```

2.  **Create a virtual environment:**
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

4.  **Set up your Hugging Face token:**
    Create a file named `.env` in the project root and add your Hugging Face API token. You can get a token from the [Hugging Face website](https://huggingface.co/settings/tokens).

    ```
    # .env
    HF_TOKEN="your_hugging_face_api_token_here"
    ```

## Running the Application

1.  **Start the FastAPI server:**
    The first time you run the application, it will download the language and embedding models, which may take some time. It will also create the FAISS indices and save them to the `faiss_indices/` directory for faster startup on subsequent runs.

    ```
    uvicorn main:app --reload
    ```

    The server will be available at `http://127.0.0.1:8000`.

2.  **Interact with the API:**
    You can use tools like `curl`, Postman, or the automatically generated Swagger UI to interact with the chatbot.

    -   **Swagger UI**: Open your browser and navigate to `http://127.0.0.1:8000/docs`.

    -   **`curl` Example (English):**
        ```
        curl -X POST "http://127.0.0.1:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{"question": "What is RAG?"}'
        ```

    -   **`curl` Example (Arabic):**
        ```
        curl -X POST "http://127.0.0.1:8000/chat" \
        -H "Content-Type: application/json" \
        -d '{"question": "ما هو الذكاء الاصطناعي؟"}'
        ```

    The expected response will be a JSON object:
    ```
    {
      "response": "Retrieval-Augmented Generation (RAG) is an advanced AI framework that combines a retriever and a generator to produce context-aware responses."
    }
    ```