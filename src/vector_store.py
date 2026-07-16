"""
vector_store.py
----------------
A thin wrapper around FAISS for storing ticket embeddings and retrieving
the most similar historical tickets to a new, incoming ticket.

Uses cosine similarity, implemented as inner product search over
L2-normalized vectors (a standard FAISS trick: IndexFlatIP + normalization
== cosine similarity).
"""

import pickle
import numpy as np
import faiss


class TicketVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []  # parallel list: metadata[i] describes vector i

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10  # avoid division by zero for empty vectors
        return vectors / norms

    def add(self, vectors: np.ndarray, metadata: list):
        """Add vectors + their associated metadata (e.g. text, category, id)."""
        vectors = self._normalize(vectors.astype("float32"))
        self.index.add(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Return the top_k most similar items as (metadata, score) tuples."""
        query_vector = self._normalize(query_vector.astype("float32"))
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, index_path: str, metadata_path: str):
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump({"dim": self.dim, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        with open(metadata_path, "rb") as f:
            data = pickle.load(f)
        store = cls(dim=data["dim"])
        store.index = faiss.read_index(index_path)
        store.metadata = data["metadata"]
        return store
