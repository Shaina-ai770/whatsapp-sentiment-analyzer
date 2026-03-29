# Retrieval-Augmented Generation (RAG) Pipeline

This project implements a traditional RAG pipeline in Python. It ingests a .doc file, indexes its content, retrieves relevant context for a user query, and generates an answer using a language model.

## Features
- Ingests and processes .doc (Microsoft Word) documents
- Indexes document content for retrieval
- Retrieves relevant context for a user query
- Uses a generative model (e.g., OpenAI GPT) to answer questions
- CLI script to run the pipeline end-to-end

## Requirements
- Python 3.8+
- Install dependencies with `pip install -r requirements.txt`


## Usage
1. Place your .doc file in the project directory (optional).
2. Run the pipeline with a document, Google Search, or both, and select your LLM provider (OpenAI or Gemini):

   **With a document (OpenAI or Gemini):**
   ```bash
   # OpenAI
   python rag_pipeline.py --doc_path your_document.doc --query "Your question here" --api_key YOUR_OPENAI_KEY --provider openai
   # Gemini
   python rag_pipeline.py --doc_path your_document.doc --query "Your question here" --gemini_api_key YOUR_GEMINI_KEY --provider gemini
   ```

   **With Google Search (Serper AI):**
   ```bash
   # OpenAI
   python rag_pipeline.py --query "Your question here" --use_google --serper_api_key YOUR_SERPER_KEY --api_key YOUR_OPENAI_KEY --provider openai
   # Gemini
   python rag_pipeline.py --query "Your question here" --use_google --serper_api_key YOUR_SERPER_KEY --gemini_api_key YOUR_GEMINI_KEY --provider gemini
   ```

   **With both document and Google Search:**
   ```bash
   # OpenAI
   python rag_pipeline.py --doc_path your_document.doc --query "Your question here" --use_google --serper_api_key YOUR_SERPER_KEY --api_key YOUR_OPENAI_KEY --provider openai
   # Gemini
   python rag_pipeline.py --doc_path your_document.doc --query "Your question here" --use_google --serper_api_key YOUR_SERPER_KEY --gemini_api_key YOUR_GEMINI_KEY --provider gemini
   ```

   - `--provider` selects the LLM provider: `openai` or `gemini`
   - `--gemini_api_key` is your Google Gemini API key
   - `--api_key` is your OpenAI API key
   - `--use_google` enables Google Search via Serper AI
   - `--serper_api_key` is your Serper AI API key
   - `--top_k` and `--google_results` control the number of context chunks from doc and Google, respectively


## Project Structure
- `rag_pipeline.py`: Main script for the RAG pipeline
- `retriever.py`: Document indexing and retrieval logic
- `generator.py`: Generative model interface
- `google_search.py`: Google Search via Serper AI
- `requirements.txt`: Python dependencies

## Notes
- Replace the OpenAI API key placeholder in `generator.py` with your actual key.
- For other LLMs, modify `generator.py` accordingly.
