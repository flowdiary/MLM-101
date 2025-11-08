#!/usr/bin/env python3
"""
Lecture 83 - RAG with LangChain and Gradio Script

Usage:
    python 03_rag_langchain_gradio.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SAMPLE_DOCUMENTS = [
    "Deep learning uses neural networks with multiple layers for complex pattern recognition.",
    "Model deployment involves serializing trained models and serving them via APIs.",
    "FastAPI is a high-performance Python framework ideal for ML model serving.",
    "Docker containers ensure consistent deployment across different environments.",
    "RAG systems combine retrieval and generation for accurate, grounded responses.",
    "TensorFlow and PyTorch are the leading deep learning frameworks.",
    "Monitoring model performance in production is crucial for maintaining quality.",
    "API versioning allows smooth transitions between model updates."
]


class SimpleRAGSystem:
    """Simple RAG system using FAISS and sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        logger.info(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        self.dimension = self.embedder.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        
    def build_index(self, documents: List[str]):
        """Build FAISS index from documents."""
        logger.info(f"Building index for {len(documents)} documents...")
        self.documents = documents
        
        embeddings = self.embedder.encode(documents, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        logger.info(f"✓ Index built with {self.index.ntotal} vectors")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve top-k most relevant documents."""
        if self.index is None:
            raise ValueError("Index not built.")
        
        query_embedding = self.embedder.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))
        
        return results
    
    def generate_answer(self, query: str, context: List[str]) -> str:
        """Generate answer using context."""
        context_text = "\n\n".join(context[:3])
        
        answer = f"""**Question:** {query}

**Retrieved Context:**
{context_text}

**Generated Answer:**
Based on the retrieved information: {context[0]}

This demonstrates a RAG system. In production, integrate with GPT-4, Claude, or open-source LLMs.
"""
        return answer
    
    def query(self, question: str, top_k: int = 3) -> Tuple[str, List[str]]:
        """Complete RAG pipeline."""
        retrieved = self.retrieve(question, top_k)
        sources = [doc for doc, _ in retrieved]
        answer = self.generate_answer(question, sources)
        return answer, sources


def main():
    # Initialize RAG
    rag = SimpleRAGSystem()
    rag.build_index(SAMPLE_DOCUMENTS)
    
    # Interactive mode
    print("\n" + "="*60)
    print("RAG System Demo - Enter questions (or 'quit' to exit)")
    print("="*60 + "\n")
    
    while True:
        question = input("Question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        answer, sources = rag.query(question)
        print(f"\n{answer}\n")
        print("Sources:")
        for i, src in enumerate(sources, 1):
            print(f"{i}. {src}")
        print("\n" + "-"*60 + "\n")

if __name__ == "__main__":
    main()
