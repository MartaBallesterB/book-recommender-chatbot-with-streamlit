import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FILLER_WORDS = {
    "book", "books", "novel", "novels", "story", "stories", "tale", "tales",
    "read", "reading", "looking", "find", "want",
    "recommend", "recommendation", "something", "give", "show", "suggest",
}

def preprocess_query(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in FILLER_WORDS)

class BookTFIDFRecommender:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
        )
        self.book_vectors = None

    def fit(self, books):
        self.book_vectors = self.vectorizer.fit_transform(books["combined"])

    def recommend(self, query, top_n, books, min_score = 0.05):
        query_vec = self.vectorizer.transform([preprocess_query(query)])
        scores = cosine_similarity(query_vec, self.book_vectors).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        top_indices = [i for i in top_indices if scores[i] >= min_score]
        if not top_indices:
            return pd.DataFrame()
        result = books.iloc[top_indices][["title", "author", "genres", "summary"]].reset_index(drop=True)
        result["score"] = scores[top_indices].round(3)
        return result
