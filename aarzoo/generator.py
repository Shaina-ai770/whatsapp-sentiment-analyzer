

import os
from typing import List

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class Generator:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", provider: str = "openai", gemini_api_key: str = None):
        self.provider = provider
        self.model = model
        if provider == "openai":
            import openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            openai.api_key = self.api_key
            self.openai = openai
        elif provider == "gemini":
            self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(self, query: str, context: List[str]) -> str:
        prompt = f"Context: {' '.join(context)}\n\nQuestion: {query}\nAnswer:"
        if self.provider == "openai":
            response = self.openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.2,
            )
            return response.choices[0].message['content'].strip()
        elif self.provider == "gemini":
            import requests
            endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            headers = {"Content-Type": "application/json"}
            params = {"key": self.gemini_api_key}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 256, "temperature": 0.2}
            }
            resp = requests.post(endpoint, headers=headers, params=params, json=data)
            resp.raise_for_status()
            result = resp.json()
            try:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                return str(result)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generator Test")
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--context', type=str, nargs='+', required=True)
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--provider', type=str, default='openai', choices=['openai', 'gemini'], help='LLM provider')
    parser.add_argument('--gemini_api_key', type=str, default=None, help='Google Gemini API key')
    parser.add_argument('--model', type=str, default=None, help='Model name (optional)')
    args = parser.parse_args()
    model = args.model or ("gpt-3.5-turbo" if args.provider == "openai" else "gemini-pro")
    gen = Generator(api_key=args.api_key, model=model, provider=args.provider, gemini_api_key=args.gemini_api_key)
    answer = gen.generate(args.query, args.context)
    print("Generated Answer:\n", answer)
