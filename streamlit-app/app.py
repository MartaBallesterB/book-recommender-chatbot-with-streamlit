import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from src.main import load_books_dataset, build_tfidf, build_embeddings, recommend_tfidf, recommend_embeddings
from src.embeddings import AVAILABLE_MODELS


# ── Cached setup functions (run once per session) ────────────────────────────

@st.cache_resource
def get_books():
    return load_books_dataset()

@st.cache_resource
def get_tfidf_setup():
    books = get_books()
    vectorizer, book_vectors = build_tfidf(books)
    return vectorizer, book_vectors

@st.cache_resource
def get_embeddings_setup(model_name: str):
    books = get_books()
    embedder, book_vectors = build_embeddings(books, model_name)
    return embedder, book_vectors


# Sidebar options:
st.sidebar.title("Recommender settings")

mode = st.sidebar.radio("Mode", ["TF-IDF", "Embeddings"])

model_name = None
if mode == "Embeddings":
    model_name = st.sidebar.selectbox("Model", AVAILABLE_MODELS)

top_N = st.sidebar.slider("Number of recommendations", min_value=1, max_value=10, value=5)


# Main UI:
st.title("Welcome to my Book Recommender! 📚🤖")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! Let me help you find your next great read. Just tell me what kind of story are you looking for."}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if query := st.chat_input("What kind of story are you looking for?"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    books = get_books()

    if mode == "Embeddings":
        embedder, book_vectors = get_embeddings_setup(model_name)
        results = recommend_embeddings(query, top_N, books, embedder, book_vectors)
    else:
        vectorizer, book_vectors = get_tfidf_setup()
        results = recommend_tfidf(query, top_N, books, vectorizer, book_vectors)

    if results.empty:
        response = "I couldn't find any books matching your query. Try different keywords please!"
    else:
        response = f"Here are my top {len(results)} recommendations for you *(mode: {mode}{f' · {model_name}' if model_name else ''})*:\n\n"
        for i, row in results.iterrows():
            response += f"**{i + 1}. {row['title']}** by {row['author']}\n"
            if row["genres"]:
                response += f"_{row['genres']}_\n"
            response += "\n"

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.write(response)
