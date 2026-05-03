import json
import kagglehub
import pandas as pd
import re
from src.tfidf import BookTFIDFRecommender
from src.embeddings import BookEmbedder
from src.chroma import BookChromaStore
from src.generator import BookResponseGenerator

def json_dict_to_str(raw):
    try:
        return ", ".join(json.loads(raw).values())
    except Exception:
        return ""

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_books_dataset():
    path = kagglehub.dataset_download("ymaricar/cmu-book-summary-dataset")
    df = pd.read_csv(
        f"{path}/booksummaries.txt",
        sep="\t",
        header=None,
        names=["wiki_id", "freebase_id", "title", "author", "pub_date", "genres", "summary"],
    )
    df = df[["title", "author", "genres", "summary"]].dropna(subset=["title", "summary"])
    df["genres"] = df["genres"].apply(json_dict_to_str)
    df["author"] = df["author"].fillna("Unknown")
    df["combined"] = df["genres"] + " " + df["summary"]
    df["combined"] = df["combined"].apply(preprocess)
    return df

# V1: bag of words + cosine similarity
def build_tfidf(books):
    recommender = BookTFIDFRecommender()
    recommender.fit(books)
    return recommender

# V2: sentence embeddings + cosine similarity
def build_embeddings(books, model_name = "all-MiniLM-L6-v2"):
    embedder = BookEmbedder(model_name)
    safe_name = model_name.replace("/", "_")
    cache_path = f"data/book_vectors_{safe_name}.npy"
    book_vectors = embedder.encode_books(books["combined"].tolist(), cache_path=cache_path)
    return embedder, book_vectors

def recommend_embeddings(query, top_n, books, embedder, book_vectors):
    indices, scores = embedder.get_top_n(query, book_vectors, n=top_n)
    result = books.iloc[indices][["title", "author", "genres", "summary"]].reset_index(drop=True)
    result["score"] = [round(s, 3) for s in scores]
    return result

# V3: ChromaDB vector store
def build_chroma(books, hf_token):
    store = BookChromaStore(hf_token=hf_token)
    if not store.is_indexed():
        store.index_books(books)
    return store

# V4: LLM response generator
def build_generator(hf_token):
    return BookResponseGenerator(hf_token=hf_token)

def generate_response(query, books, generator, llm_history =None):
    return generator.generate(query, books, llm_history=llm_history)

def recommend_chroma(query, top_n, store):
    results = store.query(query, n=top_n)
    if not results["metadatas"][0]:
        return pd.DataFrame()
    rows = [
        {
            "title": metadata["title"],
            "author": metadata["author"],
            "genres": metadata["genres"],
            "summary": metadata["summary"],
            "score": round(1 - distance, 3),
        }
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
    ]
    return pd.DataFrame(rows)

