import argparse
from retriever import load_docx, chunk_text, Retriever
from generator import Generator

def main(doc_path: str, query: str, api_key: str = None, top_k: int = 3, use_google: bool = False, serper_api_key: str = None, google_results: int = 3, provider: str = "openai", gemini_api_key: str = None, model: str = None):
    context = []
    if doc_path:
        print("Loading document...")
        paragraphs = load_docx(doc_path)
        chunks = chunk_text(paragraphs)
        print(f"Indexed {len(chunks)} chunks.")
        retriever = Retriever(chunks)
        print(f"Retrieving top {top_k} relevant chunks...")
        results = retriever.retrieve(query, top_k=top_k)
        context.extend([chunk for chunk, _ in results])
    if use_google:
        from google_search import GoogleSerperSearch
        print(f"Searching Google via Serper AI for: {query}")
        searcher = GoogleSerperSearch(api_key=serper_api_key)
        google_context = searcher.search(query, num_results=google_results)
        context.extend(google_context)
    if not context:
        print("No context found. Please provide a document or enable Google Search.")
        return
    print("Generating answer...")
    gen = Generator(api_key=api_key, model=model or ("gpt-3.5-turbo" if provider=="openai" else "gemini-pro"), provider=provider, gemini_api_key=gemini_api_key)
    answer = gen.generate(query, context)
    print("\n---\nAnswer:\n", answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline on a .doc file and/or Google Search")
    parser.add_argument('--doc_path', type=str, default=None, help='Path to .doc file (optional)')
    parser.add_argument('--query', type=str, help='User question')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key (optional)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of chunks to retrieve from doc')
    parser.add_argument('--use_google', action='store_true', help='Use Serper AI Google Search for context')
    parser.add_argument('--serper_api_key', type=str, default=None, help='Serper AI API key (optional)')
    parser.add_argument('--google_results', type=int, default=3, help='Number of Google Search results to use')
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'gemini'], help='LLM provider (openai or gemini)')
    parser.add_argument('--gemini_api_key', type=str, default=None, help='Google Gemini API key (optional)')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    args = parser.parse_args()
    if not args.query:
        args.query = input("Please enter your query: ")
    main(args.doc_path, args.query, args.api_key, args.top_k, args.use_google, args.serper_api_key, args.google_results, args.provider, args.gemini_api_key, args.model)
