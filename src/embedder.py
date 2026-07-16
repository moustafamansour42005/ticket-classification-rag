"""
embedder.py
------------
Converts ticket text into numeric vector embeddings.

Two backends are provided:
  1. TFIDFEmbedder   -> default, 100% offline, no API key needed.
                        Great for demos, graduation projects, and offline dev.
  2. OpenAIEmbedder  -> uses OpenAI's embedding API (text-embedding-3-small).
                        Higher quality semantic embeddings, needs OPENAI_API_KEY.

Both expose the same interface: fit(texts), transform(texts) -> np.ndarray
so the rest of the pipeline (vector store, classifier) doesn't care which
backend is used.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFEmbedder:
    """Lightweight, offline embedder based on TF-IDF + character n-grams.

    Not as semantically rich as a neural embedding model, but works with
    zero external dependencies / API keys, which makes it ideal for
    coursework, demos, and quick prototyping.
    """

    def __init__(self, max_features: int = 2000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._fitted = False

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before transform().")
        vectors = self.vectorizer.transform(texts).toarray().astype("float32")
        return vectors

    def fit_transform(self, texts) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self._fitted = True
        return self

class SentenceTransformerEmbedder:
    """
    Local semantic embedder using Sentence Transformers.

    Uses:
        sentence-transformers/all-MiniLM-L6-v2

    Produces 384-dimensional semantic embeddings.
    """

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self._fitted = True

    def fit(self, texts):
        # Nothing to fit
        return self

    def transform(self, texts):
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return vectors.astype("float32")

    def fit_transform(self, texts):
        return self.transform(texts)

    def save(self, path):
        # Model is downloaded automatically,
        # no need to pickle it.
        pass

    def load(self, path):
        return self    


class OpenAIEmbedder:
    """Embedder backed by OpenAI's embedding API.

    Requires the OPENAI_API_KEY environment variable to be set.
    Produces higher-quality semantic embeddings than TF-IDF, useful once
    you move from a class demo to a more production-like system.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Export it or add it to a .env file "
                "to use OpenAIEmbedder, or use TFIDFEmbedder instead."
            )
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._fitted = True  # no local fitting needed

    def fit(self, texts):
        # Nothing to fit locally; kept for interface compatibility.
        return self

    def transform(self, texts) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        vectors = np.array([d.embedding for d in response.data], dtype="float32")
        return vectors

    def fit_transform(self, texts) -> np.ndarray:
        return self.transform(texts)


def get_embedder(backend: str = "sentence"):
    backend = backend.lower()

    if backend == "sentence":
        return SentenceTransformerEmbedder()

    elif backend == "tfidf":
        return TFIDFEmbedder()

    elif backend == "openai":
        return OpenAIEmbedder()

    else:
        raise ValueError(f"Unknown embedder backend: {backend}")