import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

books = pd.read_csv("data/books.csv")

books["combined"] = (books["subjects"].fillna("") + " " + books["description"].fillna(""))

vectorizer = TfidfVectorizer(stop_words="english")
book_vectors = vectorizer.fit_transform(books["combined"])


def recommend_books(query):

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, book_vectors)
    top_indices = similarity.argsort()[0][-5:][::-1]

    return books.iloc[top_indices][["title", "author"]]