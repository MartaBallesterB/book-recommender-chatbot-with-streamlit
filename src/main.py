import json
import kagglehub
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import BookEmbedder

def json_dict_to_str(raw):
    """ Function to parse a JSON dict to returns its values as a comma-separated string."""
    try:
        return ", ".join(json.loads(raw).values())
    except Exception:
        return "" # empty string for nan's and invalid formats

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # elimina puntuación
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_books_dataset() -> pd.DataFrame:
    path = kagglehub.dataset_download("ymaricar/cmu-book-summary-dataset")
    df = pd.read_csv(
        f"{path}/booksummaries.txt",
        # tab \t as column separator
        sep="\t",
        header=None,
        names=["wiki_id", "freebase_id", "title", "author", "pub_date", "genres", "summary"],
    )
    df = df[["title", "author", "genres", "summary"]].dropna(subset=["title", "summary"])
    
    # avoid dict format for genres in output
    df["genres"] = df["genres"].apply(json_dict_to_str)
    # convert nan to 'Unknown'
    df["author"] = df["author"].fillna("Unknown")
    df["combined"] = df["genres"] + " " + df["summary"]
    df["combined"] = df["combined"].apply(preprocess)
    return df

def build_tfidf(books: pd.DataFrame):
    """Builds and returns a fitted TF-IDF vectorizer and book vectors."""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2, #ignorar terminos que estan en <2 libros (reducir ruido)
        max_df=0.85, #ignorar terminos en >85% libros (muuy comunes)
        sublinear_tf=True, #usar log(1+tf) para reducir impacto de palabras super frecuentes
    )
    book_vectors = vectorizer.fit_transform(books["combined"])
    return vectorizer, book_vectors


def build_embeddings(books: pd.DataFrame):
    """Builds and returns a BookEmbedder and book vectors (cached to disk)."""
    embedder = BookEmbedder()
    book_vectors = embedder.encode_books(books["combined"].tolist(), cache_path="data/book_vectors.npy")
    return embedder, book_vectors


def recommend_tfidf(query: str, top_N: int, books: pd.DataFrame, vectorizer, book_vectors, min_score=0.05) -> pd.DataFrame:
    """Returns top N book recommendations using TF-IDF cosine similarity."""
    query_vec = vectorizer.transform([preprocess(query)])
    scores = cosine_similarity(query_vec, book_vectors).flatten()
    top_indices = scores.argsort()[-top_N:][::-1]
    top_indices = [i for i in top_indices if scores[i] >= min_score]  # filtro
    if not top_indices:
        return pd.DataFrame()
    result = books.iloc[top_indices][["title", "author", "genres", "summary"]].reset_index(drop=True)
    result["score"] = scores[top_indices].round(3)
    return result

def recommend_embeddings(query: str, top_N: int, books: pd.DataFrame, embedder: BookEmbedder, book_vectors) -> pd.DataFrame:
    """Returns top N book recommendations using sentence embeddings cosine similarity."""
    indices, scores = embedder.get_top_n(query, book_vectors, n=top_N)
    result = books.iloc[indices][["title", "author", "genres", "summary"]].reset_index(drop=True)
    result["score"] = [round(s, 3) for s in scores]
    return result


