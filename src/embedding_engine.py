import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingEngine:
    """Handles embedding generation and storage"""
    def __init__(self, model_name = 'sentence-transformers/all-mpnet-base-v2'):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
    def create_embeddings(self, chunks):
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings
    def build_faiss_index(self, embeddings):
        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
    def search(self, query, k = 5):
        if self.index is None:
            raise ValueError("Index not built. Call build_faiss_index first.")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return distances[0], indices[0] 