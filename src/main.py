import json
import kagglehub
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def json_dict_to_str(raw):
    """ Function to parse a JSON dict to returns its values as a comma-separated string."""
    try:
        return ", ".join(json.loads(raw).values())
    except Exception:
        return "" # empty string for nan's and invalid formats


def load_books_dataset():
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
    return df


books = load_books_dataset()

vectorizer = TfidfVectorizer(stop_words="english")
book_vectors = vectorizer.fit_transform(books["combined"]) # book_vector ready to use!!


def top_N_book_recommender(query: str, top_N: int) -> pd.DataFrame:
    query_vec = vectorizer.transform([query]) # query_vector ready to use too!
    similarity = cosine_similarity(query_vec, book_vectors)
    top_indices = similarity.argsort()[0][-top_N:][::-1]
    return books.iloc[top_indices][["title", "author", "genres"]].reset_index(drop=True)
