from docx import Document
from typing import List
import re

def load_docx(doc_path: str) -> List[str]:
    doc = Document(doc_path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return paragraphs

def chunk_text(paragraphs: List[str], chunk_size: int = 5) -> List[str]:
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = " ".join(paragraphs[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Simple TF-IDF Retriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    def __init__(self, chunks: List[str]):
        self.vectorizer = TfidfVectorizer()
        self.chunks = chunks
        self.embeddings = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i], sims[i]) for i in top_indices]
