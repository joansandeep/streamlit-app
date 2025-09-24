import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class Embedder:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.data_df = None

    def create_vector_store(self, data_df, faiss_save_path):
        self.data_df = data_df
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(self.data_df['text'].tolist())

        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        print(f"Saving FAISS index to {faiss_save_path}")
        os.makedirs(os.path.dirname(faiss_save_path), exist_ok=True)
        faiss.write_index(self.index, faiss_save_path)
        print("Vector store created and saved.")

    def load_vector_store(self, faiss_load_path, data_df):
        if not os.path.exists(faiss_load_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_load_path}.")
        
        print(f"Loading FAISS index from {faiss_load_path}")
        self.index = faiss.read_index(faiss_load_path)
        self.data_df = data_df
        print("Vector store loaded successfully.")

    def retrieve(self, query, top_k=5):
        if self.index is None:
            raise RuntimeError("Vector store not loaded. Call load_vector_store() first.")
        
        query_embedding = self.embedding_model.encode([query])
        D, I = self.index.search(query_embedding, top_k)
        retrieved_indices = I.flatten().tolist()
        return self.data_df.iloc[retrieved_indices]['text'].tolist()