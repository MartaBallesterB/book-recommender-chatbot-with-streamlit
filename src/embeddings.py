import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MODEL = "all-MiniLM-L6-v2"


class BookEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_books(self, texts: list[str], cache_path: str = None) -> np.ndarray:
        """
        Encodes a list of book texts into embedding vectors.
        If cache_path is provided, loads from cache if it exists, otherwise computes and saves.
        """
        if cache_path:
            import os
            if os.path.exists(cache_path):
                print(f"Loading book vectors from cache: {cache_path}")
                return np.load(cache_path)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            vectors = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
            np.save(cache_path, vectors)
            print(f"Book vectors saved to cache: {cache_path}")
            return vectors
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into an embedding vector."""
        return self.model.encode([query], convert_to_numpy=True)

    def get_top_n(self, query: str, book_vectors: np.ndarray, n: int = 5, min_score: float = 0.1) -> tuple[list[int], list[float]]:
        """
        Returns indices and scores of the top N most similar books to the query.
        Falls back to individual keywords if no results meet the min_score threshold.
        """
        def search(text):
            query_vec = self.encode_query(text)
            scores = cosine_similarity(query_vec, book_vectors).flatten()
            top_indices = scores.argsort()[-n:][::-1]
            return [(i, scores[i]) for i in top_indices if scores[i] >= min_score]

        results = search(query)

        if not results:
            keywords = [w for w in query.lower().split() if len(w) > 3]
            for word in keywords:
                results = search(word)
                if results:
                    break

        indices = [i for i, _ in results]
        scores = [float(s) for _, s in results]
        return indices, scores
